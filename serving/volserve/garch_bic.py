"""GARCH-BIC: 10개 GARCH 사양을 적합하고 BIC가 가장 작은 사양으로 H분 분산을 예측한다.

15분 수익률의 한 봉 앞 조건부 분산을 재귀로 H/15봉 앞까지 이어 더해 H분 분산을 만든다.
수익률 0(가격 정지)도 그대로 받아 분산을 직접 예측하므로 정지 확률 결합을 쓰지 않는다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

SCALE = 100.0   # 수익률을 % 단위로 바꿔 최적화를 안정시킨다
SPECS = [
    dict(vol="GARCH", p=1, o=0, q=1),
    dict(vol="GARCH", p=2, o=0, q=1),
    dict(vol="GARCH", p=1, o=0, q=2),
    dict(vol="GARCH", p=1, o=1, q=1),   # GJR-GARCH
    dict(vol="FIGARCH", p=1, q=1),
]
DISTS = ("t", "skewt")


class GarchBIC:
    def fit(self, returns: pd.Series) -> "GarchBIC":
        y = returns.to_numpy() * SCALE
        best, failures = None, []
        for spec in SPECS:
            for dist in DISTS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = arch_model(y, mean="Zero", dist=dist, **spec).fit(disp="off")
                except Exception as exc:          # 실패한 사양은 기록하고 후보에서 뺀다
                    failures.append((spec, dist, repr(exc)))
                    continue
                if res.convergence_flag != 0 or not np.isfinite(res.bic):
                    failures.append((spec, dist, f"수렴 실패(flag={res.convergence_flag})"))
                    continue
                if best is None or res.bic < best[0].bic:
                    best = (res, spec, dist)
        if best is None:
            raise RuntimeError(f"GARCH 사양이 하나도 적합되지 않았다: {failures}")
        self.result_, self.spec_, self.dist_ = best
        self.failures_ = failures
        return self

    def _fixed(self, returns: pd.Series):
        model = arch_model(returns.to_numpy() * SCALE, mean="Zero", dist=self.dist_, **self.spec_)
        return model.fix(self.result_.params)

    def predict_variance(self, returns: pd.Series, h: int) -> float:
        """returns 끝까지의 정보로 다음 h개 봉의 분산 합(RV² 예측)을 낸다."""
        fc = self._fixed(returns).forecast(horizon=h, reindex=False)
        return float(fc.variance.to_numpy()[-1].sum()) / SCALE**2

    def predict_variance_path(self, returns: pd.Series, h: int, positions: np.ndarray) -> np.ndarray:
        """학습용: 각 위치 t에서 t 직전까지의 정보로 낸 h봉 분산 합. 혼합 모델의 입력 특징으로 쓴다."""
        fc = self._fixed(returns).forecast(horizon=h, start=0, reindex=True)
        var = fc.variance.to_numpy().sum(axis=1) / SCALE**2   # 행 t: t까지 알고 t+1…t+h 예측
        prev = np.concatenate([[np.nan], var[:-1]])           # 위치 t의 예측은 t−1까지 정보로 낸 값
        return prev[positions]
