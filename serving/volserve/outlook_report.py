"""변동성 전망 리포트 에이전트: 에이전트 실행 결과를 요약해 Claude API에 보내는 LangGraph 상태 그래프.

    불러온 실행 → 요약 → 요청 구성 → 차단 검사 → 호출 → 응답 검사 → 저장
                          ↑                               │
                          └── 숫자 대조 실패(최대 3회) ───┘

    차단 검사에 걸리거나 시험(dry)이면 호출하지 않고 끝난다. 응답 검사를 끝내 통과하지 못하면 저장하지 않는다.

- 요약에는 구간별 지금 국면·예측 국면 종목 수, 쓴 모델 칸 수, 예측/직전 변동성 배율의 사분위(참고), 배율 상위·하위 몇 종목의
  배율·국면만 넣는다.
  원시 시세와 가격을 거꾸로 계산할 수 있는 촘촘한 시계열은 보내지 않는다.
- 차단 검사(guard)는 보내기 직전 요청 본문에서 비밀값(환경 변수의 키·토큰 값), 키 모양 문자열, 원시 시세 항목 이름,
  가격 모양 큰 수, 촘촘한 연속 숫자, 지나친 크기를 찾고 하나라도 있으면 호출하지 않는다.
- 판단(유지·관망·축소)은 고정 규칙으로 코드가 계산한다(`stance_detail`). LLM은 설명 문장 칸만 쓰고 판단 칸이 없다.
  비교 기준은 바로 직전 구간이 아니라 평소다: 고정 GARCH-BIC 예측을 같은 모형이 학습 구간의 같은 시각에 낸 예측 분포와 비교한
  예측 국면(Q1~Q5)으로 판단한다. 직전 대비 배율은 평균 회귀 때문에 직전이 잔잔하면 기계적으로 1을 넘기 쉬워 참고로만 둔다.
- 응답 검사는 LLM이 쓴 문장의 숫자를 모두 입력 요약과 대조한다(`number_issues`). 맞지 않으면 지적 사항을 붙여 다시 생성하고
  (첫 호출 포함 최대 3회), 그래도 맞지 않으면 리포트를 저장하지 않는다. 리포트 표·판단 값은 코드가 조립한다.
- 가격 방향은 말하지 않는다. 투자 권유가 아닌 참고용이다.

선택 의존성: langgraph, anthropic. API 키는 ANTHROPIC_API_KEY 환경 변수에서 읽는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from itertools import combinations
from typing import Any, TypedDict

from .agent import AgentRun, horizon_word

for _k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
    os.environ[_k] = "false"  # 그래프 실행 내용이 외부 추적 서비스로 나가지 않게 한다

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
STANCES = ("유지", "관망", "축소")
TOP_N = 3
LIMITS = {"bytes": 30_000, "numbers": 400, "run": 8, "int_digits": 5}
SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|ACCESS", re.IGNORECASE)
KEY_SHAPE = re.compile(r"sk-ant-[\w-]+|ntn_\w+|secret_\w+|eyJ[\w-]{10,}\.[\w-]+|[A-Za-z0-9_\-+/=]{32,}")
RAW_MARKET = re.compile(r"\b(?:open|high|low|close|volume|ohlcv?|candles?|orderbooks?|ticks?|trade_price|"
                        r"ask_price|bid_price)\b", re.IGNORECASE)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

EXPAND, SHRINK = 1.1, 0.9  # 참고: 배율 중앙값이 1.1을 넘으면 "직전보다 커질 전망", 0.9 밑이면 "줄어들 전망"
MAX_ATTEMPTS = 3  # 숫자 대조에 걸리면 지적 사항을 붙여 다시 생성하는 최대 횟수(첫 호출 포함)

SYSTEM = """너는 암호화폐 변동성 예측 에이전트의 전망 리포트에서 설명 문장을 쓰는 사람이다. 입력은 에이전트가 만든 요약 집계와, 코드가 계산한 구간별 전망·판단("규칙 계산")뿐이다.
- 지금 국면 Q1~Q5: 직전 같은 길이 구간의 실현변동성이 학습 구간 분포의 5분위 중 어디인지다(Q1 잔잔, Q5 요동).
- 예측 국면 Q1~Q5: 고정 GARCH-BIC 예측이 같은 모형으로 학습 구간(평소)의 같은 시각에 낸 예측값 분포의 5분위 중 어디인지다. 평소라면 Q1~Q5가 각각 20% 안팎이다. 미정은 같은 시각 학습 예측이 모자라 정하지 못한 칸이다.
- 배율: 예측 실현변동성 ÷ 직전 같은 길이 구간의 실현변동성이다. 참고 정보일 뿐 판단 기준이 아니다. 직전이 잔잔하면 평균으로 되돌아가는 성질 때문에 기계적으로 1을 넘기 쉽다.
- 판단(유지·관망·축소)은 코드가 정했다. 판단을 바꾸거나 다른 판단을 권하지 않고, 왜 그 판단이 나왔는지 설명만 한다.
- 판단 기준(평소 대비): 1) 축소: 4시간 또는 12시간에서 예측 국면 Q4+Q5 종목이 과반이다. 2) 관망: 1이 아니고 예측 국면 Q4+Q5 종목이 과반인 구간이 2개 이상이다. 3) 유지: 그 밖.
- 칸: market_state(지금 시장 상태 한두 문장), outlook_conclusion(앞으로 전망 결론 한 문장), outlook_points(근거 2~4개), stance_explanation(판단이 어떤 수치로 어떤 기준을 거쳐 나왔는지 2~3문장), caveats(유의점 2~4개).
- 숫자는 입력 값을 그대로 쓴다. 반올림·새로 계산한 비율이나 평균은 쓰지 않는다. 같은 구간 국면 종목 수의 합, 종목 수에서 뺀 나머지만 예외다.
- 입력에 없는 수치·사건·가격을 만들지 않고, 가격 방향은 말하지 않는다. 한국어로 쓰고 문장을 서술어로 끝맺는다. 참고용이며 투자 권유가 아니다."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["market_state", "outlook_conclusion", "outlook_points", "stance_explanation", "caveats"],
          "properties": {"market_state": {"type": "string"}, "outlook_conclusion": {"type": "string"},
                         "outlook_points": {"type": "array", "items": {"type": "string"}},
                         "stance_explanation": {"type": "string"}, "caveats": {"type": "array", "items": {"type": "string"}}}}
TEXT_FIELDS = ("market_state", "outlook_conclusion", "stance_explanation")
LIST_FIELDS = ("outlook_points", "caveats")


class ReportState(TypedDict, total=False):
    run: AgentRun
    dry: bool
    save_dir: str  # 주면 성공한 리포트만 여기에 쓴다(실패하면 아무것도 쓰지 않는다)
    attempt: int
    feedback: list
    summary: dict
    request: dict
    guard: dict
    response: dict
    report: dict
    status: str  # dry | blocked | retry | ok | refused | error
    error: str
    saved: str


# ---------------------------------------------------------------- 판단 규칙(코드 계산)

def _median_ratio(d: dict) -> float | None:
    q = d.get("배율 사분위(25%·중앙·75%)") or []
    return q[1] if len(q) == 3 else None


def direction(d: dict) -> str:
    """참고: 직전 대비 배율 중앙값으로 본 방향."""
    m = _median_ratio(d)
    if m is None:
        return "알 수 없음"
    return "커질 전망" if m > EXPAND else "줄어들 전망" if m < SHRINK else "비슷할 전망"


def _forecast_groups(d: dict) -> tuple[int, int, int, int]:
    """예측 국면 종목 수 → (잔잔 Q1+Q2, 보통 Q3, 요동 Q4+Q5, 미정)."""
    c = d.get("예측 국면 종목 수") or {}
    return c.get("Q1", 0) + c.get("Q2", 0), c.get("Q3", 0), c.get("Q4", 0) + c.get("Q5", 0), c.get("미정", 0)


def level_word(d: dict) -> str:
    """평소 대비 전망: 예측 국면 Q4+Q5가 과반이면 높음, Q1+Q2가 과반이면 낮음, 그 밖은 평소 수준(미정도 분모에 넣는다)."""
    lo, _, hi, _ = g = _forecast_groups(d)
    n = sum(g)
    return "평소보다 높을 전망" if hi * 2 > n else "평소보다 낮을 전망" if lo * 2 > n else "평소 수준 전망"


def _regime_groups(d: dict) -> tuple[int, int, int]:
    """국면 종목 수 → (잔잔 Q1+Q2, 보통 Q3, 요동 Q4+Q5)."""
    c = d["국면 종목 수"]
    return c.get("Q1", 0) + c.get("Q2", 0), c.get("Q3", 0), c.get("Q4", 0) + c.get("Q5", 0)


def stance_detail(summary: dict) -> dict:
    """판단 기준(1 축소 → 2 관망 → 3 유지)을 위에서부터 적용한 결과와 근거. LLM은 이 값을 바꾸지 못한다.
    비교 기준은 평소다: 예측 국면 Q4+Q5가 과반인 구간을 센다(문턱은 결과를 보기 전에 고정)."""
    seg = summary["구간"]
    high = [h for h, d in seg.items() if _forecast_groups(d)[2] * 2 > sum(_forecast_groups(d))]
    hit = [h for h in ("4시간", "12시간") if h in high]
    stance, rule = ("축소", 1) if hit else ("관망", 2) if len(high) >= 2 else ("유지", 3)
    return {"판단": stance, "맞은 기준": rule, "요동 예측 우세 구간": high, "축소 기준에 걸린 구간": hit,
            "구간별 전망": {h: level_word(d) for h, d in seg.items()},
            "참고: 직전 대비 방향": {h: direction(d) for h, d in seg.items()}}


def rule_stance(summary: dict) -> str:
    return stance_detail(summary)["판단"]


# ---------------------------------------------------------------- 요약·요청

def summarize(run: AgentRun, top_n: int = TOP_N) -> dict:
    out: dict[str, Any] = {"기준 시각": str(run.as_of)[:16], "종목 수": len({c.ticker for c in run.cells}), "구간": {}}
    for h in sorted({c.route.horizon_minutes for c in run.cells}):
        cs = [c for c in run.cells if c.route.horizon_minutes == h]
        rs = sorted(((c.rv / c.prev_rv, c) for c in cs if c.prev_rv > 0), key=lambda x: x[0])
        vals = [r for r, _ in rs]
        q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else [statistics.median(vals)] * 3 if vals else []

        def pick(r, c):
            return {"종목": c.ticker, "배율": round(r, 2), "국면": f"Q{c.route.regime}"}

        out["구간"][horizon_word(h)] = {
            "국면 종목 수": dict(sorted(Counter(f"Q{c.route.regime}" for c in cs).items())),
            "예측 국면 종목 수": dict(sorted(Counter(f"Q{c.forecast_regime}" if c.forecast_regime else "미정" for c in cs).items())),
            "쓴 모델": dict(Counter(c.route.research_model for c in cs).most_common()),
            "배율 계산된 종목 수": len(vals),  # 직전 변동성이 0이면 배율을 못 구한다
            "배율 사분위(25%·중앙·75%)": [round(x, 2) for x in q],
            "배율 1.1 초과 종목 수": sum(r > EXPAND for r in vals),
            "배율 0.9 미만 종목 수": sum(r < SHRINK for r in vals),
            "배율 상위": [pick(r, c) for r, c in rs[::-1][:top_n]],
            "배율 하위": [pick(r, c) for r, c in rs[:top_n]],
        }
    out["규칙 계산"] = stance_detail(out)  # 판단은 코드가 정하고 LLM은 설명만 한다
    return out


def compose(summary: dict, feedback: list | None = None) -> dict:
    user = ("다음은 요약 집계와 코드가 계산한 구간별 전망·판단(\"규칙 계산\")이다. 이것만 근거로 설명 문장 칸을 채워라.\n\n```json\n"
            + json.dumps(summary, ensure_ascii=False) + "\n```")
    if feedback:
        user += ("\n\n직전 응답의 숫자가 입력과 맞지 않아 다시 요청한다. 다음을 고쳐서 처음부터 다시 써라: " + "; ".join(feedback)
                 + ". 입력에 있는 값만 그대로 쓴다. 이 지적 자체는 어느 칸에도 쓰지 않는다.")
    return {"model": MODEL, "system": SYSTEM, "user": user}


# ---------------------------------------------------------------- 숫자 전수 대조

DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|\b\d{2}:\d{2}\b")


def _numbers(text: str) -> list[float]:
    """글 속 숫자(날짜·시각은 빼고, 부호는 무시)."""
    return [abs(float(n)) for n in NUMBER.findall(DATETIME.sub(" ", text))]


def number_issues(text: str, summary: dict, system: str = SYSTEM) -> list[str]:
    """LLM이 쓴 글의 숫자가 모두 입력 요약(또는 지시문)에서 나온 값인지 본다. 같은 구간 지금·예측 국면 종목 수와 모델 칸 수의 부분합과
    종목 수에서 뺀 나머지는 허용한다. 빈 값 표기(null·None·NaN)도 잡는다."""
    allowed = set(_numbers(json.dumps(summary, ensure_ascii=False))) | set(_numbers(system))
    n_tk = float(summary.get("종목 수") or 0)
    for d in summary["구간"].values():
        for group in (list(d["국면 종목 수"].values()), list(d.get("예측 국면 종목 수", {}).values()), list(d["쓴 모델"].values())):
            for r in range(1, len(group) + 1):
                for c in combinations(group, r):
                    allowed |= {float(sum(c)), n_tk - sum(c)}
    issues = [f"빈 값 표기({m.group()})" for m in re.finditer(r"(?<![A-Za-z])(?:null|None|NaN|nan)(?![A-Za-z])", text)]
    issues += [f"입력에 없는 수 {n:g}" for n in sorted({x for x in _numbers(text) if not any(abs(x - a) < 1e-9 for a in allowed)})]
    return issues


# ---------------------------------------------------------------- 리포트 조립(코드)

def _table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + " :--- |" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render_markdown(summary: dict, rep: dict) -> str:
    """고정 형식: 예측값 요약 → 1 지금 시장 상태 → 2 현재 전망 → 3 판단(규칙 계산) → 4 유의점.
    표·판단 값은 코드가 쓰고, LLM 문장 칸만 끼워 넣는다."""
    seg, det = summary["구간"], summary["규칙 계산"]

    def med(d):
        m = _median_ratio(d)
        return f"{m:.2f}" if m is not None else "계산 안 됨"

    def groups(d):
        lo, mid, hi, und = _forecast_groups(d)
        return f"{lo} / {mid} / {hi}" + (f" (미정 {und})" if und else "")

    return "\n".join([
        "## 예측값 요약",
        _table(["구간", "예측 국면 잔잔/보통/요동", "평소 대비 전망", "직전 대비 배율(참고)"],
               [[h, groups(d), det["구간별 전망"][h], med(d)] for h, d in seg.items()]), "",
        "## 1. 지금 시장 상태", rep["market_state"].strip(), "",
        _table(["구간", "잔잔(Q1·Q2)", "보통(Q3)", "요동(Q4·Q5)"], [[h, *_regime_groups(d)] for h, d in seg.items()]), "",
        "## 2. 현재 전망", f"**{rep['outlook_conclusion'].strip()}**", "", *[f"- {x.strip()}" for x in rep["outlook_points"]], "",
        f"## 3. 판단: {det['판단']} (규칙 계산)",
        f"맞은 기준: {det['맞은 기준']}번 · 평소보다 높을 예측이 과반인 구간: {', '.join(det['요동 예측 우세 구간']) or '없음'}", "",
        rep["stance_explanation"].strip(), "",
        "## 4. 유의점", *[f"- {x.strip()}" for x in rep["caveats"]],
        "- 판단은 정해진 기준을 코드로 계산한 값이고, 문장은 LLM이 그 계산을 설명한 것이다. 가격 방향은 다루지 않으며 투자 권유가 아니다.",
    ]).strip()


def _long_runs(obj: Any, limit: int) -> list[int]:
    found: list[int] = []
    if isinstance(obj, dict):
        for v in obj.values():
            found += _long_runs(v, limit)
    elif isinstance(obj, list):
        run = best = 0
        for v in obj:
            run = run + 1 if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
            best = max(best, run)
            found += _long_runs(v, limit)
        if best > limit:
            found.append(best)
    return found


def guard(request: dict, summary: dict, environ: dict | None = None) -> dict:
    """보내기 직전 차단 검사. 하나라도 걸리면 호출하지 않는다."""
    env = os.environ if environ is None else environ
    body = json.dumps(request, ensure_ascii=False)
    reasons = []
    if any(len(v) >= 6 and v in body for k, v in env.items() if SECRET_NAME.search(k)):
        reasons.append("비밀값(키·토큰)이 요청 본문에 들어갔다")
    if m := KEY_SHAPE.search(body):
        reasons.append(f"키 모양 문자열({m.group()[:6]}…)")
    if m := RAW_MARKET.search(body):
        reasons.append(f"원시 시세 항목 이름({m.group()})")
    nums = NUMBER.findall(request["user"])
    if len(nums) > LIMITS["numbers"]:
        reasons.append(f"숫자 {len(nums)}개(한도 {LIMITS['numbers']})")
    if any(len(n.lstrip("-").split(".")[0]) >= LIMITS["int_digits"] for n in nums):
        reasons.append("가격 모양 큰 수")
    if runs := _long_runs(summary, LIMITS["run"]):
        reasons.append(f"연속 숫자 배열 길이 {max(runs)}(촘촘한 시계열)")
    if len(body.encode()) > LIMITS["bytes"]:
        reasons.append("요청 본문이 너무 크다")
    return {"ok": not reasons, "reasons": reasons, "bytes": len(body.encode()), "numbers": len(nums),
            "sha256": hashlib.sha256(body.encode()).hexdigest()}


def _node_summarize(st: ReportState) -> ReportState:
    return {"summary": summarize(st["run"])}


def _node_compose(st: ReportState) -> ReportState:
    return {"request": compose(st["summary"], st.get("feedback")), "attempt": int(st.get("attempt") or 0) + 1}


def _node_guard(st: ReportState) -> ReportState:
    g = guard(st["request"], st["summary"])
    if not g["ok"]:
        return {"guard": g, "status": "blocked", "error": "; ".join(g["reasons"])}
    return {"guard": g, "status": "dry" if st.get("dry") else "call"}


def call_claude(req: dict) -> dict:
    """Claude API를 한 번 부른다. 시험에서는 write_report(call=...)로 가짜 응답 함수를 넣어 네트워크를 쓰지 않는다."""
    import anthropic

    client = anthropic.Anthropic(timeout=300.0, max_retries=2)
    t0 = time.monotonic()
    try:
        resp = client.beta.messages.create(
            model=req["model"], max_tokens=MAX_TOKENS, betas=[FALLBACK_BETA], fallbacks="default",
            system=req["system"], messages=[{"role": "user", "content": req["user"]}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}})
    except anthropic.AuthenticationError as e:
        return {"error": f"인증 실패: {e.status_code}"}
    except anthropic.RateLimitError as e:
        return {"error": f"사용량 한도: {e.status_code}"}
    except anthropic.APIStatusError as e:
        return {"error": f"API 오류 {e.status_code}"}
    except anthropic.APIConnectionError as e:
        return {"error": f"연결 실패: {type(e).__name__}"}
    u = resp.usage
    return {"requested_model": req["model"], "served_model": resp.model, "stop_reason": resp.stop_reason,
            "usage": {k: getattr(u, k, None) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")},
            "elapsed_s": round(time.monotonic() - t0, 2), "request_id": getattr(resp, "_request_id", None),
            "text": next((b.text for b in resp.content if b.type == "text"), "")}


def _make_call_node(call):
    def _node_call(st: ReportState) -> ReportState:
        req = st["request"]
        if guard(req, st["summary"])["sha256"] != st["guard"]["sha256"]:
            return {"status": "blocked", "error": "차단 검사 뒤 요청 본문이 바뀌었다"}
        r = call(req)
        if "error" in r:
            return {"status": "error", "error": r["error"]}
        return {"response": r}
    return _node_call


def _node_check(st: ReportState) -> ReportState:
    r = st.get("response")
    if r is None:
        return {}
    if r["stop_reason"] in ("refusal", "max_tokens"):
        return {"status": "refused" if r["stop_reason"] == "refusal" else "error", "error": r["stop_reason"]}
    try:
        rep = json.loads(r["text"])
    except json.JSONDecodeError:
        return {"status": "error", "error": "응답이 JSON이 아니다"}
    if (any(not str(rep.get(k) or "").strip() for k in TEXT_FIELDS)
            or any(not isinstance(rep.get(k), list) or not rep[k] for k in LIST_FIELDS)):
        return {"status": "error", "error": "설명 칸이 비었다"}
    text = "\n".join([*(rep[k] for k in TEXT_FIELDS), *rep["outlook_points"], *rep["caveats"]])
    if issues := number_issues(text, st["summary"]):
        if int(st.get("attempt") or 1) < MAX_ATTEMPTS:
            return {"status": "retry", "feedback": issues[:12], "response": None}
        return {"status": "error", "error": f"{MAX_ATTEMPTS}번 생성해도 숫자가 입력과 맞지 않음: " + ", ".join(issues[:12])}
    det = st["summary"]["규칙 계산"]
    report = {**rep, "stance": det["판단"], "stance_source": "규칙 계산", "stance_rule": det["맞은 기준"],
              "report_markdown": render_markdown(st["summary"], rep)}
    return {"report": report, "status": "ok", "feedback": []}


def _node_save(st: ReportState) -> ReportState:
    """성공한 리포트만 쓴다. 차단·거절·숫자 대조 실패면 아무것도 쓰지 않는다."""
    if st.get("status") != "ok" or not st.get("save_dir"):
        return {}
    from pathlib import Path

    p = Path(st["save_dir"]) / f"report_{str(st['summary']['기준 시각']).replace(' ', '_').replace(':', '')}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# 변동성 전망 리포트 ({st['summary']['기준 시각']} 기준)\n\n" + st["report"]["report_markdown"] + "\n")
    return {"saved": str(p)}


def build_graph(call=None):
    """불러온 실행 → 요약 → 요청 구성 → 차단 검사 → 호출 → 응답 검사(숫자 대조 실패면 요청 구성으로 돌아가 재생성) → 저장."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(ReportState)
    for name, fn in (("summarize", _node_summarize), ("compose", _node_compose), ("guard", _node_guard),
                     ("call", _make_call_node(call or call_claude)), ("check", _node_check), ("save", _node_save)):
        g.add_node(name, fn)
    g.add_edge(START, "summarize")
    g.add_edge("summarize", "compose")
    g.add_edge("compose", "guard")
    g.add_conditional_edges("guard", lambda st: "call" if st.get("status") == "call" else END, {"call": "call", END: END})
    g.add_conditional_edges("call", lambda st: "check" if st.get("response") else END, {"check": "check", END: END})
    g.add_conditional_edges("check", lambda st: "compose" if st.get("status") == "retry" else "save",
                            {"compose": "compose", "save": "save"})
    g.add_edge("save", END)
    return g.compile()


def write_report(run: AgentRun, dry: bool = True, call=None, save_dir: str | None = None) -> ReportState:
    """dry=True면 요약·차단 검사까지만 하고 보내지 않는다. call은 요청 → 응답 사전 함수(기본: Claude API).
    save_dir를 주면 응답 검사를 통과한 리포트만 그 폴더에 쓴다."""
    return build_graph(call).invoke({"run": run, "dry": dry, "save_dir": save_dir or ""})
