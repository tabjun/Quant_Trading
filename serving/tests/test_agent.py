"""국면 라우팅 에이전트·공개 집계·노션 블록·리포트 차단 검사를 합성 자료로 시험한다(네트워크 없음)."""
import json

import numpy as np
import pytest
from test_synthetic import synthetic_returns
from volserve.agent import RegimeRoutingAgent, check_public, public_summary
from volserve.notion_page import build_blocks
from volserve.routing import IMPLEMENTED, ROUTE_TABLE, SUBSTITUTE, route

HORIZONS = (60, 720)


def test_route_table_covers_every_cell():
    for h in ROUTE_TABLE:
        for q in range(1, 6):
            r = route(h, q)
            assert r.served_model in IMPLEMENTED.values()
            assert r.substituted == (r.research_model in SUBSTITUTE)


@pytest.fixture(scope="module")
def fitted():
    data = {f"T{i}": synthetic_returns(n=9_000, seed=i) for i in range(3)}
    train = {k: v.iloc[:8_000] for k, v in data.items()}
    return RegimeRoutingAgent(horizons=HORIZONS, gru_kwargs={"epochs": 2, "device": "cpu"}).fit(train), data, train


@pytest.fixture(scope="module")
def run(fitted):
    agent, data, _ = fitted
    return agent.run(data)


def test_forecast_regime_compares_with_same_hour_training_forecasts(fitted, run):
    """판단 기준(평소 대비): 학습 구간의 같은 시각 GARCH-BIC 예측을 그 경계에 다시 넣으면 각 5분위가 20% 안팎이어야 한다."""
    agent, _, train = fitted
    svc = agent.services_["T0"]
    for h in HORIZONS:
        bars = h // 15
        pos = np.arange(673, len(train["T0"]) - bars + 1)
        rv = np.sqrt(svc.garch_.predict_variance_path(train["T0"], bars, pos))
        hours = train["T0"].index[pos].hour
        hr = 6
        edges = svc.forecast_edges_[h][hr]
        share = np.bincount(np.digitize(rv[hours == hr], edges), minlength=5) / (hours == hr).sum()
        assert np.allclose(share, 0.2, atol=0.03), share
    assert all(c.forecast_regime is None or 1 <= c.forecast_regime <= 5 for c in run.cells)
    assert any(c.forecast_regime is not None for c in run.cells)


def test_agent_runs_every_cell(run):
    assert len(run.cells) == 3 * len(HORIZONS)
    for c in run.cells:
        assert np.isfinite(c.rv) and c.rv > 0 and 1 <= c.route.regime <= 5
        assert c.explanation.startswith(c.ticker)
    assert list(run.stage_seconds)[:5] == ["입력 확인", "국면 판정", "모델 라우터", "예측", "설명 생성"]


def test_public_summary_has_no_tickers_or_values(run):
    s = public_summary(run)
    text = json.dumps(s, ensure_ascii=False)
    assert not any(c.ticker in text for c in run.cells)
    assert sum(s["regime_counts"]["60"].values()) == 3
    blocks = build_blocks(s)
    assert all(b["type"] in ("paragraph", "heading_2", "table") for b in blocks)
    with pytest.raises(ValueError):
        check_public({"x": "예측 0.012345"})


def test_report_guard_blocks_secrets_and_dense_series(run):
    from volserve.outlook_report import compose, guard, summarize

    s = summarize(run)
    env = {"EXCHANGE_ACCESS_KEY": "exchange-secret-9"}
    assert guard(compose(s), s, env)["ok"]
    for bad in ({**s, "메모": "exchange-secret-9"}, {**s, "rv": [0.01] * 20}, {**s, "가격": 143250000},
                {**s, "close": 1.0}):
        assert not guard(compose(bad), bad, env)["ok"]


def test_report_graph_dry_run_sends_nothing(run):
    pytest.importorskip("langgraph")
    from volserve.outlook_report import write_report

    st = write_report(run, dry=True)
    assert st["status"] == "dry" and st["guard"]["ok"] and "response" not in st
