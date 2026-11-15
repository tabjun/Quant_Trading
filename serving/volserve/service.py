"""예측 구간마다 정한 모델로 다음 H분 실현변동성을 예측한다.

기본 경로는 연구에서 전체 기간 1위였던 모델이다.
  15분·30분 → GARCH-BIC, 1시간·4시간 → GARCH-BIC+LightGBM, 12시간 → Nystroem+Ridge
연구에서 전체 기간 1위는 가장 요동하는 국면(Q5)에서도 Q5 1위와 통계적으로 구분되지 않았으므로,
국면에 따라 모델을 바꾸는 경로는 기본으로 쓰지 않는다. 필요하면 regime_route로 켤 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import (Periodicity, future_rv, horizon_bars, past_rv, regime_edges, regime_of,
                       sequence_input, tabular_features)
from .garch_bic import GarchBIC
from .models import GRURV, HaltClassifier, LightGBMRV, NystroemRidgeRV, Split

DEFAULT_ROUTE = {15: "garch_bic", 30: "garch_bic", 60: "garch_lgbm", 240: "garch_lgbm", 720: "nystroem_ridge"}
SEQ_LEN = 96
WARMUP = 672 + 1          # 가장 긴 특징 창(1주)이 찰 때까지는 학습 표본으로 쓰지 않는다


@dataclass
class Forecast:
    horizon_minutes: int
    model: str
    rv: float                 # 다음 H분 실현변동성 예측(수익률 단위)
    variance: float           # rv²
    halt_prob: float | None   # 로그 타깃 모델만
    regime: int               # 1(잔잔) – 5(요동), 직전 H분 RV 기준


@dataclass
class VolatilityService:
    route: dict = field(default_factory=lambda: dict(DEFAULT_ROUTE))
    regime_route: dict = field(default_factory=dict)      # 예: {(60, 5): "gru"}; 기본은 비어 있음
    gru_kwargs: dict = field(default_factory=dict)

    def fit(self, returns: pd.Series) -> "VolatilityService":
        """returns: 15분 격자 로그 수익률(체결이 없던 봉은 0). 이 구간 전체가 학습 구간이다."""
        if returns.isna().any():
            raise ValueError("수익률에 결측이 있다. 체결이 없던 봉은 0으로 채운다")
        self.periodicity_ = Periodicity().fit(returns)
        d = self.periodicity_.transform(returns)
        self.garch_ = GarchBIC().fit(returns)
        self.parts_, self.edges_ = {}, {}
        needed = {(h, m) for h, m in self.route.items()} | {(h, m) for (h, _), m in self.regime_route.items()}
        for h_min in sorted({h for h, _ in needed}):
            h = horizon_bars(h_min)
            self.edges_[h_min] = regime_edges(past_rv(returns, h).iloc[WARMUP:])
            names = {m for hh, m in needed if hh == h_min and m != "garch_bic"}
            if names:
                self.parts_[h_min] = self._fit_log_models(returns, d, h, names)
        return self

    def _fit_log_models(self, returns, d, h, names):
        tgt = future_rv(d, h).to_numpy()
        positions = np.arange(WARMUP, len(d) - h + 1, h)          # 창이 겹치지 않게 h봉 간격
        positions = positions[np.isfinite(tgt[positions])]
        split = Split.of(positions)
        is_val = np.isin(positions, split.val)
        tab = tabular_features(d).iloc[positions]
        halt = HaltClassifier().fit(tab[~is_val], tgt[positions][~is_val] == 0)
        out = {"halt": halt}
        for name in names:
            x = self._inputs(name, returns, d, h, positions, tab)
            out[name] = self._new(name).fit(x, tgt[positions], is_val)
        return out

    def _inputs(self, name, returns, d, h, positions, tab):
        if name == "garch_lgbm":
            g = self.garch_.predict_variance_path(returns, h, positions)
            return tab.assign(garch_logvar=np.log(g))
        if name == "nystroem_ridge":
            return tab
        if name == "lightgbm":
            return tab
        if name == "gru":
            return sequence_input(d, positions, SEQ_LEN)
        raise KeyError(name)

    def _new(self, name):
        return {"garch_lgbm": LightGBMRV, "lightgbm": LightGBMRV, "nystroem_ridge": NystroemRidgeRV,
                "gru": lambda: GRURV(**self.gru_kwargs)}[name]()

    def predict(self, returns: pd.Series, horizon_minutes: int) -> Forecast:
        """returns 끝까지의 정보로 다음 horizon_minutes 동안의 RV를 예측한다(returns 다음 봉부터)."""
        h = horizon_bars(horizon_minutes)
        prev = float(np.sqrt((returns.iloc[-h:] ** 2).sum()))
        regime = regime_of(prev, self.edges_[horizon_minutes])
        name = self.regime_route.get((horizon_minutes, regime), self.route[horizon_minutes])
        if name == "garch_bic":
            var = self.garch_.predict_variance(returns, h)
            return Forecast(horizon_minutes, name, float(np.sqrt(var)), var, None, regime)
        # 로그 타깃 모델: 다음 봉 위치를 하나 덧붙여 '그 시점 직전까지의 정보'로 특징을 만든다
        step = returns.index[1] - returns.index[0]
        ext = pd.concat([returns, pd.Series([0.0], index=[returns.index[-1] + step])])
        d = self.periodicity_.transform(ext)
        t = np.array([len(ext) - 1])
        tab = tabular_features(d).iloc[t]
        parts = self.parts_[horizon_minutes]
        x = self._inputs(name, ext, d, h, t, tab)
        pi = parts["halt"].predict(tab)
        var_d = float(parts[name].predict_variance(x, pi)[0])
        future_idx = pd.date_range(ext.index[-1], periods=h, freq=step)
        var = var_d * float(np.mean(self.periodicity_.factors(future_idx) ** 2))
        return Forecast(horizon_minutes, name, float(np.sqrt(var)), var, float(pi[0]), regime)
