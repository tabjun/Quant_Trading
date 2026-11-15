"""합성 자료(GARCH(1,1) 수익률에 가격 정지를 섞은 15분 계열)로 전체 흐름을 시험한다. 실제 시세는 쓰지 않는다."""
import numpy as np
import pandas as pd
import pytest

from volserve import DEFAULT_ROUTE, VolatilityService
from volserve.features import future_rv, horizon_bars, past_rv, regime_of, tabular_features


def synthetic_returns(n=12_000, seed=0, halt_share=0.15):
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-7, 0.08, 0.90
    var, r = omega / (1 - alpha - beta), np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(var) * rng.standard_t(5) / np.sqrt(5 / 3)
        var = omega + alpha * r[i] ** 2 + beta * var
    r[rng.random(n) < halt_share] = 0.0                     # 체결이 없던 봉
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.Series(r, index=idx)


@pytest.fixture(scope="module")
def returns():
    return synthetic_returns()


def test_targets_do_not_look_ahead(returns):
    h = horizon_bars(60)
    fwd, back = future_rv(returns, h), past_rv(returns, h)
    t = 1000
    assert np.isclose(fwd.iloc[t], np.sqrt((returns.iloc[t:t + h] ** 2).sum()))
    assert np.isclose(back.iloc[t], np.sqrt((returns.iloc[t - h:t] ** 2).sum()))
    feats = tabular_features(returns)
    changed = returns.copy()
    changed.iloc[t:] *= 10                                   # t 이후를 바꿔도 t의 특징은 그대로여야 한다
    assert np.allclose(tabular_features(changed).iloc[t].to_numpy(), feats.iloc[t].to_numpy(), equal_nan=True)


def test_regime_upper_cell_on_boundary():
    edges = np.array([1.0, 2.0, 3.0, 4.0])
    assert regime_of(0.5, edges) == 1 and regime_of(2.0, edges) == 3 and regime_of(9.0, edges) == 5


def test_service_routes_and_predicts(returns):
    train, live = returns.iloc[:10_000], returns.iloc[:10_500]
    svc = VolatilityService(regime_route={(60, 5): "gru"}, gru_kwargs={"epochs": 3, "device": "cpu"}).fit(train)
    for h_min, name in DEFAULT_ROUTE.items():
        fc = svc.predict(live, h_min)
        assert fc.model in (name, "gru")
        assert np.isfinite(fc.rv) and fc.rv > 0
        assert 1 <= fc.regime <= 5
        if fc.model != "garch_bic":
            assert 0 <= fc.halt_prob <= 1
    log_model = svc.parts_[60]["garch_lgbm"]
    assert log_model.c_pos_ > 0                              # 보정 배율은 양수
