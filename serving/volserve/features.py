"""15분 수익률에서 예측 대상과 입력 특징을 만든다.

입력은 15분 격자에 맞춘 로그 수익률 Series(DatetimeIndex, 체결이 없던 봉은 0)다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MINUTES = 15
RV_WINDOWS = (1, 4, 16, 96, 672)      # 15분, 1시간, 4시간, 1일, 1주
ABS_LAGS = (1, 2, 4, 8, 16)
MEAN_WINDOWS = (4, 16, 96)


def horizon_bars(horizon_minutes: int) -> int:
    if horizon_minutes % BAR_MINUTES:
        raise ValueError("예측 구간은 15분의 배수여야 한다")
    return horizon_minutes // BAR_MINUTES


def future_rv(returns: pd.Series, h: int) -> pd.Series:
    """시점 t 기준 다음 h개 봉(t, …, t+h−1)의 실현변동성. 끝부분은 NaN."""
    sq = returns.pow(2)
    fwd = sq[::-1].rolling(h, min_periods=h).sum()[::-1]
    return np.sqrt(fwd)


def past_rv(returns: pd.Series, h: int) -> pd.Series:
    """시점 t 직전 h개 봉(t−h, …, t−1)의 실현변동성. 예측 시점 t에 이미 알려져 있다."""
    return np.sqrt(returns.pow(2).rolling(h, min_periods=h).sum().shift(1))


class Periodicity:
    """요일×시간대(168칸)의 평균 변동 수준. 수익률을 이 값으로 나눠 하루·한 주의 주기를 없앤다."""

    def fit(self, returns: pd.Series) -> "Periodicity":
        sq = returns.pow(2)
        key = _cell(returns.index)
        overall = sq.mean()
        self.factor_ = np.sqrt(sq.groupby(key).mean() / overall).replace(0, np.nan).fillna(1.0)
        return self

    def factors(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.factor_.reindex(_cell(index)).fillna(1.0).to_numpy()

    def transform(self, returns: pd.Series) -> pd.Series:
        return returns / self.factors(returns.index)


def _cell(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index.dayofweek * 24 + index.hour)


def tabular_features(d: pd.Series) -> pd.DataFrame:
    """특징 기반 모델의 입력 16개: 여러 척도의 RV, RV 비율, |d| 시차, d 평균. 모두 시점 t 직전 봉까지만 쓴다."""
    cols = {}
    for w in RV_WINDOWS:
        cols[f"rv_{w}"] = past_rv(d, w)
    cols["ratio_1_16"] = cols["rv_1"] / cols["rv_16"]
    cols["ratio_4_96"] = cols["rv_4"] / cols["rv_96"]
    cols["ratio_16_672"] = cols["rv_16"] / cols["rv_672"]
    absd = d.abs()
    for lag in ABS_LAGS:
        cols[f"absd_lag{lag}"] = absd.shift(lag)
    for w in MEAN_WINDOWS:
        cols[f"dmean_{w}"] = d.rolling(w, min_periods=w).mean().shift(1)
    out = pd.DataFrame(cols, index=d.index)
    for c in out.columns:
        if c.startswith("rv_"):
            out[c] = np.log(out[c].where(out[c] > 0))
    return out.replace([np.inf, -np.inf], np.nan)


def sequence_input(d: pd.Series, t_positions: np.ndarray, length: int = 96) -> np.ndarray:
    """순환 신경망 입력: 시점마다 직전 length개 봉의 (d, |d|). 모양은 (표본, length, 2)."""
    arr = d.to_numpy()
    idx = t_positions[:, None] - length + np.arange(length)[None, :]
    x = arr[idx]
    return np.stack([x, np.abs(x)], axis=-1).astype(np.float32)


def regime_edges(prev_rv_train: pd.Series) -> np.ndarray:
    """학습 구간 직전 RV 분포의 20·40·60·80% 분위수. 국면 경계로 쓴다."""
    return np.nanquantile(prev_rv_train.to_numpy(), [0.2, 0.4, 0.6, 0.8])


def regime_of(prev_rv: float, edges: np.ndarray) -> int:
    """1(가장 잔잔) – 5(가장 요동). 경계값과 같으면 위 칸에 넣는다."""
    return int(np.digitize(prev_rv, edges)) + 1
