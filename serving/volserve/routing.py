"""국면 × 예측 구간 칸마다 연구 1위 모델을 고르는 라우팅 표.

국면은 직전 H분 실현변동성이 학습 구간 분포의 어느 5분위에 드는지다(1 잔잔 – 5 요동).
표의 모델은 연구에서 그 칸의 QLIKE 1위이고, ties는 그 1위와 통계적으로 구분되지 않은 다른 모델 수다(Holm 보정).
ties가 크면 "1위"는 여러 동등한 후보 중 하나라는 뜻이다. 12시간은 국면별 표가 없어 전체 기간 1위를 모든 국면에 쓴다.

이 패키지가 직접 구현한 모델은 GARCH-BIC, GARCH-BIC+LightGBM, Nystroem+Ridge, GRU 넷이다. 표의 1위가 그 밖의
모델(LSTM, TimesFM, PatchTST, TimeXer)이면 가장 가까운 구현으로 대신하고 substituted=True로 표시한다.
"""
from __future__ import annotations

from dataclasses import dataclass

SCOPES = ("Q1", "Q2", "Q3", "Q4", "Q5")

# 구간(분) → 국면 → (연구 1위 모델, 구분 안 되는 다른 모델 수)
ROUTE_TABLE: dict[int, dict[str, tuple[str, int | None]]] = {
    15: {"Q1": ("GARCH-BIC", 1), "Q2": ("GARCH-BIC", 1), "Q3": ("GARCH-BIC", 4), "Q4": ("GARCH-BIC", 2),
         "Q5": ("LSTM", 8)},
    30: {"Q1": ("GARCH-BIC", 5), "Q2": ("GARCH-BIC", 8), "Q3": ("GARCH-BIC+LightGBM", 8), "Q4": ("GARCH-BIC", 9),
         "Q5": ("LSTM", 6)},
    60: {"Q1": ("GARCH-BIC+LightGBM", 16), "Q2": ("TimesFM", 22), "Q3": ("TimesFM", 22),
         "Q4": ("GARCH-BIC+LightGBM", 6), "Q5": ("GRU", 6)},
    240: {"Q1": ("TimesFM", 15), "Q2": ("TimeXer", 24), "Q3": ("Nystroem+Ridge", 19), "Q4": ("Nystroem+Ridge", 9),
          "Q5": ("GRU", 4)},
    720: {q: ("Nystroem+Ridge", None) for q in SCOPES},
}

# 연구 모델 이름 → 이 패키지의 구현 이름
IMPLEMENTED = {"GARCH-BIC": "garch_bic", "GARCH-BIC+LightGBM": "garch_lgbm", "Nystroem+Ridge": "nystroem_ridge",
               "GRU": "gru"}
# 구현하지 않은 1위 모델의 대신: 순환 신경망은 GRU, 사전학습·어텐션 계열은 GARCH-BIC+LightGBM
SUBSTITUTE = {"LSTM": "gru", "TimesFM": "garch_lgbm", "PatchTST": "garch_lgbm", "TimeXer": "garch_lgbm"}


@dataclass(frozen=True)
class Route:
    horizon_minutes: int
    regime: int
    research_model: str
    ties: int | None
    served_model: str
    substituted: bool


def route(horizon_minutes: int, regime: int) -> Route:
    model, ties = ROUTE_TABLE[horizon_minutes][f"Q{regime}"]
    served = IMPLEMENTED.get(model) or SUBSTITUTE[model]
    return Route(horizon_minutes, regime, model, ties, served, model not in IMPLEMENTED)


def regime_route(horizons=tuple(ROUTE_TABLE)) -> dict[tuple[int, int], str]:
    """VolatilityService(regime_route=...)에 넘길 (구간, 국면) → 구현 이름 표."""
    return {(h, q): route(h, q).served_model for h in horizons for q in range(1, 6)}
