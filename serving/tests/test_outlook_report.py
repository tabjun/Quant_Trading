"""전망 리포트 품질 장치(판단 코드 계산, 숫자 전수 대조, 재생성·저장 안 함)를 합성 자료와 가짜 응답으로 시험한다.

네트워크를 쓰지 않는다: Claude 호출은 `write_report(call=...)`에 넣은 가짜 응답 함수로 대신한다.
"""
import json

import pandas as pd
import pytest
from volserve.agent import AgentRun, Cell
from volserve.outlook_report import SCHEMA, number_issues, rule_stance, stance_detail, summarize
from volserve.routing import route

HORIZONS = (15, 30, 60, 240, 720)


def make_run(ratio_by_h: dict[int, float], regimes: tuple[int, ...], forecast: dict[int, tuple] | None = None) -> AgentRun:
    """종목 수 = len(regimes). 구간 h의 모든 종목 배율을 ratio_by_h[h] 근처(±0.05)로, 예측 국면을 forecast[h]로 둔 합성 실행."""
    cells = []
    for h in HORIZONS:
        fq = (forecast or {}).get(h, regimes)
        for i, q in enumerate(regimes):
            prev = 0.01 + 0.001 * i
            r = ratio_by_h[h] + (i - (len(regimes) - 1) / 2) * 0.05 / max(len(regimes) - 1, 1)
            cells.append(Cell(ticker=f"T{i}", route=route(h, q), rv=prev * r, prev_rv=prev, forecast_regime=fq[i]))
    return AgentRun(as_of=pd.Timestamp("2024-01-01 06:00"), cells=cells)


CALM, WILD = (1, 1, 2, 2, 3), (5, 5, 4, 4, 3)
HIGH = (5, 5, 4, 3, 2)  # 예측 국면 Q4+Q5가 5종목 중 3(0.6, 관망 문턱과 같아 넘지 않는다)
WATCH = (4, 4, 4, 5, 2)  # Q4+Q5 4/5 = 0.8, Q5 1/5
SHRINK_Q5 = (5, 5, 5, 1, 1)  # Q5 3/5 = 0.6 > 0.5
UNDECIDED = (5, 5, None, None, None)  # 미정도 분모에 들어가 Q5 2/5는 과반이 아니다


FLAT = {h: 1.0 for h in HORIZONS}
UP = {h: 1.5 for h in HORIZONS}


@pytest.mark.parametrize("ratios, forecast, stance, rule", [
    (FLAT, {15: CALM, 30: CALM, 60: CALM, 240: SHRINK_Q5, 720: CALM}, "축소", 1),  # 4시간 표 Q5 과반
    (FLAT, {15: CALM, 30: CALM, 60: CALM, 240: WATCH, 720: CALM}, "관망", 2),      # 4시간 Q4+Q5 0.8, Q5는 과반 아님
    (FLAT, {15: WATCH, 30: WATCH, 60: WATCH, 240: CALM, 720: CALM}, "관망", 2),    # 단기 표 12/15
    (FLAT, {15: WATCH, 30: WATCH, 60: CALM, 240: CALM, 720: CALM}, "유지", 3),     # 짧은 두 구간만 높으면 단기 표 8/15 = 0.53
    (FLAT, {15: CALM, 30: CALM, 60: CALM, 240: HIGH, 720: CALM}, "유지", 3),       # 0.6은 관망 문턱을 넘지 않는다
    (UP, {h: CALM for h in HORIZONS}, "유지", 3),                                  # 배율이 모두 1.5여도 판단에는 쓰지 않는다
    (FLAT, {15: CALM, 30: CALM, 60: CALM, 240: UNDECIDED, 720: CALM}, "유지", 3),  # 미정이 섞여 Q5 2/5
])
def test_stance_is_computed_by_code(ratios, forecast, stance, rule):
    s = summarize(make_run(ratios, CALM, forecast))
    det = stance_detail(s)
    assert (det["판단"], det["맞은 기준"]) == (stance, rule) and rule_stance(s) == stance
    assert s["규칙 계산"]["판단"] == stance  # LLM에 보내는 요약에도 코드 판단이 들어간다
    assert "stance" not in SCHEMA["properties"]  # LLM 응답에는 판단 칸이 없다


def test_number_check_allows_input_values_and_sums_only():
    s = summarize(make_run(UP, CALM, {h: HIGH for h in HORIZONS}))
    seg = s["구간"]["15분"]
    calm = seg["국면 종목 수"].get("Q1", 0) + seg["국면 종목 수"].get("Q2", 0)
    high = seg["예측 국면 종목 수"].get("Q4", 0) + seg["예측 국면 종목 수"].get("Q5", 0)
    ok = f"15분 배율 중앙값은 {seg['배율 사분위(25%·중앙·75%)'][1]}이고 잔잔한 종목은 {calm}개, 평소보다 높을 예측은 {high}개다."
    assert number_issues(ok, s) == []
    for bad in ("배율 중앙값은 7.77배다.", "구분 안 됨 수는 null이다."):
        assert number_issues(bad, s), bad


def fake_response(text_fields: dict) -> dict:
    return {"requested_model": "fake", "served_model": "fake", "stop_reason": "end_turn", "usage": {}, "elapsed_s": 0.0,
            "request_id": None, "text": json.dumps(text_fields, ensure_ascii=False)}


GOOD = {"market_state": "대부분 종목이 잔잔하다.", "outlook_conclusion": "짧은 세 구간에서 평소보다 높을 전망이다.",
        "outlook_points": ["15분·30분·1시간에서 예측 국면 Q4·Q5가 과반이다."], "stance_explanation": "기준 2에 맞아 관망이다.",
        "caveats": ["합성 자료다."]}
BAD = {**GOOD, "stance_explanation": "배율이 7.77배라서 관망이다."}


class FakeClaude:
    """미리 정한 응답을 차례로 돌려주는 가짜 호출. 받은 요청을 남긴다."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, req):
        self.requests.append(req)
        return fake_response(self.replies[min(len(self.requests), len(self.replies)) - 1])


@pytest.fixture
def run():
    pytest.importorskip("langgraph")
    return make_run(FLAT, CALM, {15: WATCH, 30: WATCH, 60: WATCH, 240: CALM, 720: CALM})


def test_ok_report_uses_code_stance_and_is_saved(run, tmp_path):
    from volserve.outlook_report import write_report

    fake = FakeClaude(GOOD)
    st = write_report(run, dry=False, call=fake, save_dir=str(tmp_path))
    assert st["status"] == "ok" and st["attempt"] == 1 and len(fake.requests) == 1
    assert st["report"]["stance"] == "관망" and st["report"]["stance_source"] == "규칙 계산"
    heads = [ln for ln in st["report"]["report_markdown"].splitlines() if ln.startswith("## ")]
    assert [h[:5] for h in heads] == ["## 예측", "## 1.", "## 2.", "## 3.", "## 4."] and "(규칙 계산)" in heads[3]
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_bad_number_triggers_regeneration(run, tmp_path):
    from volserve.outlook_report import write_report

    fake = FakeClaude(BAD, GOOD)
    st = write_report(run, dry=False, call=fake, save_dir=str(tmp_path))
    assert st["status"] == "ok" and st["attempt"] == 2 and len(fake.requests) == 2
    assert "7.77" in fake.requests[1]["user"] and "7.77" not in fake.requests[0]["user"]  # 지적 사항을 붙여 다시 요청
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_three_failures_save_nothing(run, tmp_path):
    from volserve.outlook_report import MAX_ATTEMPTS, write_report

    fake = FakeClaude(BAD)
    st = write_report(run, dry=False, call=fake, save_dir=str(tmp_path))
    assert st["status"] == "error" and len(fake.requests) == MAX_ATTEMPTS == 3
    assert "report" not in st and not list(tmp_path.iterdir())


def test_dry_run_calls_nothing(run, tmp_path):
    from volserve.outlook_report import write_report

    fake = FakeClaude(GOOD)
    st = write_report(run, dry=True, call=fake, save_dir=str(tmp_path))
    assert st["status"] == "dry" and not fake.requests and not list(tmp_path.iterdir())
