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


def make_run(ratio_by_h: dict[int, float], regimes: tuple[int, ...]) -> AgentRun:
    """종목 수 = len(regimes). 구간 h의 모든 종목 배율을 ratio_by_h[h] 근처(±0.05)로 둔 합성 실행."""
    cells = []
    for h in HORIZONS:
        for i, q in enumerate(regimes):
            prev = 0.01 + 0.001 * i
            r = ratio_by_h[h] + (i - (len(regimes) - 1) / 2) * 0.05 / max(len(regimes) - 1, 1)
            cells.append(Cell(ticker=f"T{i}", route=route(h, q), rv=prev * r, prev_rv=prev))
    return AgentRun(as_of=pd.Timestamp("2024-01-01 06:00"), cells=cells)


CALM, WILD = (1, 1, 2, 2, 3), (5, 5, 4, 4, 3)


@pytest.mark.parametrize("ratios, regimes, stance, rule", [
    ({15: 1.0, 30: 1.0, 60: 1.0, 240: 1.5, 720: 1.0}, WILD, "축소", 1),   # 4시간 확대 + 요동 우세
    ({15: 1.5, 30: 1.5, 60: 1.5, 240: 1.5, 720: 1.0}, CALM, "관망", 2),   # 확대 4구간, 잔잔 우세라 축소 아님
    ({15: 1.5, 30: 1.0, 60: 1.0, 240: 1.0, 720: 0.8}, CALM, "유지", 3),   # 확대 1구간
])
def test_stance_is_computed_by_code(ratios, regimes, stance, rule):
    s = summarize(make_run(ratios, regimes))
    det = stance_detail(s)
    assert (det["판단"], det["맞은 기준"]) == (stance, rule) and rule_stance(s) == stance
    assert s["규칙 계산"]["판단"] == stance  # LLM에 보내는 요약에도 코드 판단이 들어간다
    assert "stance" not in SCHEMA["properties"]  # LLM 응답에는 판단 칸이 없다


def test_number_check_allows_input_values_and_sums_only():
    s = summarize(make_run({h: 1.5 for h in HORIZONS}, CALM))
    seg = s["구간"]["15분"]
    calm = seg["국면 종목 수"].get("Q1", 0) + seg["국면 종목 수"].get("Q2", 0)
    ok = f"15분 배율 중앙값은 {seg['배율 사분위(25%·중앙·75%)'][1]}이고 잔잔한 종목은 {calm}개다."
    assert number_issues(ok, s) == []
    for bad in ("배율 중앙값은 7.77배다.", "구분 안 됨 수는 null이다."):
        assert number_issues(bad, s), bad


def fake_response(text_fields: dict) -> dict:
    return {"requested_model": "fake", "served_model": "fake", "stop_reason": "end_turn", "usage": {}, "elapsed_s": 0.0,
            "request_id": None, "text": json.dumps(text_fields, ensure_ascii=False)}


GOOD = {"market_state": "대부분 종목이 잔잔하다.", "outlook_conclusion": "다섯 구간 중 네 구간이 커질 전망이다.",
        "outlook_points": ["짧은 구간 배율 중앙값이 기준을 넘는다."], "stance_explanation": "기준 2에 맞아 관망이다.",
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
    return make_run({15: 1.5, 30: 1.5, 60: 1.5, 240: 1.5, 720: 1.0}, CALM)


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
