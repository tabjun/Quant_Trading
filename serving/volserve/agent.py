"""국면 라우팅 에이전트: 여러 종목의 다섯 구간 실현변동성을 국면별 1위 모델로 예측한다.

단계: 입력 확인 → 국면 판정 → 모델 라우터 → 예측 → 설명 생성 → 공개 집계

공개 집계(`public_summary`)에는 구간별 국면 종목 수, 모델별 사용 칸 수, 단계별 시간만 담는다. 종목 이름과 예측값은
넣지 않고, `check_public`이 다시 확인한다. 공개 화면(노션 등)에는 이 집계만 올린다.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import horizon_bars, regime_of
from .routing import ROUTE_TABLE, Route, regime_route, route
from .service import DEFAULT_ROUTE, VolatilityService

HORIZONS = (15, 30, 60, 240, 720)
STAGES = ("입력 확인", "국면 판정", "모델 라우터", "예측", "설명 생성", "공개 집계")
REGIME_WORD = {1: "가장 잔잔한 하위 20%", 2: "하위 20~40%", 3: "중간 40~60%", 4: "상위 20~40%", 5: "가장 요동치는 상위 20%"}


def horizon_word(h: int) -> str:
    return f"{h}분" if h < 60 else f"{h // 60}시간"


@dataclass
class Cell:
    ticker: str
    route: Route
    rv: float            # 다음 H분 실현변동성 예측
    prev_rv: float       # 직전 H분 실현변동성
    explanation: str = ""


@dataclass
class AgentRun:
    as_of: pd.Timestamp
    cells: list[Cell]
    stage_seconds: dict[str, float] = field(default_factory=dict)


def explain(cell: Cell) -> str:
    r, h = cell.route, horizon_word(cell.route.horizon_minutes)
    used = f"{r.research_model}" if not r.substituted else f"{r.research_model} 대신 {r.served_model}(이 패키지 구현)"
    ties = ("이 칸의 구분 안 되는 모델 수는 연구 표에 없습니다" if r.ties is None
            else f"이 칸 1위는 다른 모델 {r.ties}개와 통계적으로 구분되지 않습니다")
    ratio = cell.rv / cell.prev_rv if cell.prev_rv > 0 else float("nan")
    move = ("커질" if ratio > 1.1 else "줄어들" if ratio < 0.9 else "비슷할") if np.isfinite(ratio) else "판단할 수 없을"
    return (f"{cell.ticker} {h}: 직전 {h} 실현변동성이 학습 구간의 {REGIME_WORD[r.regime]}(Q{r.regime})에 들어 {used}로 예측했습니다. "
            f"{ties}. 예측 변동성은 직전의 {ratio:.2f}배로 {move} 것으로 봅니다.")


class RegimeRoutingAgent:
    def __init__(self, horizons=HORIZONS, gru_kwargs: dict | None = None):
        self.horizons = tuple(horizons)
        self.gru_kwargs = gru_kwargs or {}

    def fit(self, returns_by_ticker: dict[str, pd.Series]) -> "RegimeRoutingAgent":
        """종목마다 학습 구간 수익률로 국면 경계와 표에 필요한 모델을 적합한다."""
        self.services_ = {
            tk: VolatilityService(route={h: DEFAULT_ROUTE[h] for h in self.horizons},
                                  regime_route=regime_route(self.horizons), gru_kwargs=self.gru_kwargs).fit(r)
            for tk, r in returns_by_ticker.items()}
        return self

    def run(self, returns_by_ticker: dict[str, pd.Series]) -> AgentRun:
        sec, t0 = {}, time.perf_counter()

        def lap(name):
            nonlocal t0
            now = time.perf_counter()
            sec[name], t0 = round(now - t0, 3), now

        for tk, r in returns_by_ticker.items():             # 1 입력 확인
            if tk not in self.services_:
                raise KeyError(f"적합하지 않은 종목: {tk}")
            if r.isna().any():
                raise ValueError(f"{tk}: 수익률에 결측이 있다")
        as_of = min(r.index[-1] for r in returns_by_ticker.values())
        lap(STAGES[0])
        regimes = {}                                         # 2 국면 판정
        for tk, r in returns_by_ticker.items():
            for h in self.horizons:
                prev = float(np.sqrt((r.iloc[-horizon_bars(h):] ** 2).sum()))
                regimes[tk, h] = (regime_of(prev, self.services_[tk].edges_[h]), prev)
        lap(STAGES[1])
        routes = {k: route(k[1], q) for k, (q, _) in regimes.items()}   # 3 모델 라우터
        lap(STAGES[2])
        cells = []                                           # 4 예측
        for (tk, h), rt in routes.items():
            fc = self.services_[tk].predict(returns_by_ticker[tk], h)
            if fc.model != rt.served_model or fc.regime != rt.regime:
                raise RuntimeError(f"{tk} {h}분: 라우터와 예측 모델이 어긋났다")
            cells.append(Cell(tk, rt, fc.rv, regimes[tk, h][1]))
        lap(STAGES[3])
        for c in cells:                                      # 5 설명 생성
            c.explanation = explain(c)
        lap(STAGES[4])
        run = AgentRun(as_of, cells, sec)
        public_summary(run)                                  # 6 공개 집계(시간 재기용)
        lap(STAGES[5])
        return run


def public_summary(run: AgentRun) -> dict:
    """공개해도 되는 집계만. 종목 이름·예측값·설명 문장은 넣지 않는다."""
    hs = sorted({c.route.horizon_minutes for c in run.cells})
    by_h = {h: [c for c in run.cells if c.route.horizon_minutes == h] for h in hs}
    out = {
        "as_of": str(run.as_of)[:16],
        "tickers": len({c.ticker for c in run.cells}),
        "horizons_min": hs,
        "regime_counts": {str(h): {f"Q{q}": sum(c.route.regime == q for c in cs) for q in range(1, 6)} for h, cs in by_h.items()},
        "model_counts": {str(h): dict(Counter(c.route.research_model + (" (대신 구현)" if c.route.substituted else "")
                                              for c in cs).most_common()) for h, cs in by_h.items()},
        "route_table": {str(h): {q: {"model": m, "ties": t} for q, (m, t) in ROUTE_TABLE[h].items()} for h in hs},
        "stage_seconds": dict(run.stage_seconds),
    }
    check_public(out, [c.ticker for c in run.cells])
    return out


LONG_DECIMAL = re.compile(r"\d\.\d{4,}")


def check_public(obj, tickers=()) -> None:
    """공개 자료에 종목 이름이나 소수 넷째 자리 이상의 값(예측값 모양)이 들어가면 멈춘다."""
    text = repr(obj)
    for tk in set(tickers):
        if tk and tk in text:
            raise ValueError(f"공개 자료에 종목 이름이 들어갔다: {tk}")
    if m := LONG_DECIMAL.search(text):
        raise ValueError(f"공개 자료에 값 모양 수가 들어갔다: {m.group()}")
