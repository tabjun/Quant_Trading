"""변동성 전망 리포트 에이전트: 에이전트 실행 결과를 요약해 Claude API에 보내는 LangGraph 상태 그래프.

    불러온 실행 → 요약 → 요청 구성 → 차단 검사 ─┬─ 막힘 ──────────────┐
                                               ├─ 시험(dry) ─────────┤
                                               └─ 호출 → 응답 검사 ──┴→ 끝

- 요약에는 구간별 국면 종목 수, 쓴 모델 칸 수, 예측/직전 변동성 배율의 사분위, 배율 상위·하위 몇 종목의 배율·국면만 넣는다.
  원시 시세와 가격을 거꾸로 계산할 수 있는 촘촘한 시계열은 보내지 않는다.
- 차단 검사(guard)는 보내기 직전 요청 본문에서 비밀값(환경 변수의 키·토큰 값), 키 모양 문자열, 원시 시세 항목 이름,
  가격 모양 큰 수, 촘촘한 연속 숫자, 지나친 크기를 찾고 하나라도 있으면 호출하지 않는다.
- 판단은 위험 노출에 대한 셋 중 하나(유지·관망·축소)이고, 가격 방향은 말하지 않는다. 투자 권유가 아닌 참고용이다.

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

SYSTEM = """너는 암호화폐 변동성 예측 에이전트의 전망 리포트 작성자다. 입력은 에이전트가 만든 요약 집계뿐이다.
- 국면 Q1~Q5: 직전 같은 길이 구간의 실현변동성이 학습 구간 분포의 5분위 중 어디인지다(Q1 잔잔, Q5 요동).
- 배율: 예측 실현변동성 ÷ 직전 같은 길이 구간의 실현변동성이다.
- 입력에 없는 수치·사건·가격을 만들지 않고, 가격 방향은 말하지 않는다.
- 판단(stance)은 위험 노출에 대한 유지·관망·축소 중 하나다. 변동성 확대가 넓게 예상되면 보수적으로 고른다.
- 한국어로 쓰고 참고용임을 밝힌다. report_markdown은 "## 한 줄 요약", "## 구간별 전망", "## 눈여겨볼 종목", "## 판단과 근거", "## 해석 유의점"으로 쓴다."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["stance", "stance_reason", "report_markdown", "caveats"],
          "properties": {"stance": {"type": "string", "enum": list(STANCES)}, "stance_reason": {"type": "string"},
                         "report_markdown": {"type": "string"}, "caveats": {"type": "array", "items": {"type": "string"}}}}


class ReportState(TypedDict, total=False):
    run: AgentRun
    dry: bool
    summary: dict
    request: dict
    guard: dict
    response: dict
    report: dict
    status: str  # dry | blocked | ok | refused | error
    error: str


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
            "쓴 모델": dict(Counter(c.route.research_model for c in cs).most_common()),
            "배율 사분위(25%·중앙·75%)": [round(x, 2) for x in q],
            "배율 1.1 초과 종목 수": sum(r > 1.1 for r in vals),
            "배율 0.9 미만 종목 수": sum(r < 0.9 for r in vals),
            "배율 상위": [pick(r, c) for r, c in rs[::-1][:top_n]],
            "배율 하위": [pick(r, c) for r, c in rs[:top_n]],
        }
    return out


def compose(summary: dict) -> dict:
    user = "다음 요약만 근거로 변동성 전망 리포트와 위험 노출 판단을 써라.\n\n```json\n" + json.dumps(summary, ensure_ascii=False) + "\n```"
    return {"model": MODEL, "system": SYSTEM, "user": user}


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
    s = summarize(st["run"])
    return {"summary": s, "request": compose(s)}


def _node_guard(st: ReportState) -> ReportState:
    g = guard(st["request"], st["summary"])
    if not g["ok"]:
        return {"guard": g, "status": "blocked", "error": "; ".join(g["reasons"])}
    return {"guard": g, "status": "dry" if st.get("dry") else "call"}


def _node_call(st: ReportState) -> ReportState:
    import anthropic

    req = st["request"]
    if guard(req, st["summary"])["sha256"] != st["guard"]["sha256"]:
        return {"status": "blocked", "error": "차단 검사 뒤 요청 본문이 바뀌었다"}
    client = anthropic.Anthropic(timeout=300.0, max_retries=2)
    t0 = time.monotonic()
    try:
        resp = client.beta.messages.create(
            model=req["model"], max_tokens=MAX_TOKENS, betas=[FALLBACK_BETA], fallbacks="default",
            system=req["system"], messages=[{"role": "user", "content": req["user"]}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}})
    except anthropic.AuthenticationError as e:
        return {"status": "error", "error": f"인증 실패: {e.status_code}"}
    except anthropic.RateLimitError as e:
        return {"status": "error", "error": f"사용량 한도: {e.status_code}"}
    except anthropic.APIStatusError as e:
        return {"status": "error", "error": f"API 오류 {e.status_code}"}
    except anthropic.APIConnectionError as e:
        return {"status": "error", "error": f"연결 실패: {type(e).__name__}"}
    u = resp.usage
    return {"response": {
        "requested_model": req["model"], "served_model": resp.model, "stop_reason": resp.stop_reason,
        "usage": {k: getattr(u, k, None) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")},
        "elapsed_s": round(time.monotonic() - t0, 2), "request_id": getattr(resp, "_request_id", None),
        "text": next((b.text for b in resp.content if b.type == "text"), "")}}


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
    if rep.get("stance") not in STANCES:
        return {"status": "error", "error": "판단 값이 없다"}
    return {"report": rep, "status": "ok"}


def build_graph():
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(ReportState)
    for name, fn in (("summarize", _node_summarize), ("guard", _node_guard), ("call", _node_call), ("check", _node_check)):
        g.add_node(name, fn)
    g.add_edge(START, "summarize")
    g.add_edge("summarize", "guard")
    g.add_conditional_edges("guard", lambda st: "call" if st.get("status") == "call" else END, {"call": "call", END: END})
    g.add_edge("call", "check")
    g.add_edge("check", END)
    return g.compile()


def write_report(run: AgentRun, dry: bool = True) -> ReportState:
    """dry=True면 요약·차단 검사까지만 하고 보내지 않는다."""
    return build_graph().invoke({"run": run, "dry": dry})
