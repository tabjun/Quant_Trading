"""로그 RV를 학습하는 모델과 두 부분 결합(정지 확률 × 보정 배율).

모든 로그 타깃 모델은 같은 순서로 예측한다.
1) 정지가 아닐 때의 log RV 예측 → 2) 지수 변환 → 3) 학습 범위 밖이면 학습 평균으로 대체
→ 4) 보정 배율 c₊ 곱하기 → 5) (1 − 정지 확률) 곱하기 → 6) 요일×시간대 주기를 되돌림.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_HALT_SAMPLES = 200   # 정지 표본이 이보다 적으면 정지 확률을 0으로 둔다


@dataclass
class Split:
    """학습 위치를 내부학습(앞 85%)과 내부검증(뒤 15%)으로 나눈 결과."""
    fit: np.ndarray
    val: np.ndarray

    @classmethod
    def of(cls, positions: np.ndarray, val_share: float = 0.15) -> "Split":
        cut = int(len(positions) * (1 - val_share))
        return cls(positions[:cut], positions[cut:])


class HaltClassifier:
    """다음 구간이 가격 정지(RV = 0)일 확률 π."""

    def fit(self, x: pd.DataFrame, halted: np.ndarray) -> "HaltClassifier":
        self.constant_ = None
        if halted.sum() < MIN_HALT_SAMPLES:
            self.constant_ = 0.0
            return self
        self.model_ = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
        self.model_.fit(x, halted.astype(int))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(x), self.constant_)
        return self.model_.predict_proba(x)[:, 1]


class LogTargetModel:
    """하위 클래스는 _fit_raw(x_fit, y_fit, x_val, y_val)와 _predict_log(x)만 구현한다."""

    def fit(self, x: pd.DataFrame | np.ndarray, rv: np.ndarray, split_val: np.ndarray) -> "LogTargetModel":
        """x·rv는 학습 위치 순서. split_val은 그중 내부검증 위치를 가리키는 불리언 배열."""
        pos = rv > 0
        fit_m, val_m = pos & ~split_val, pos & split_val
        self._fit_raw(_take(x, fit_m), np.log(rv[fit_m]), _take(x, val_m), np.log(rv[val_m]))
        self.lo_, self.hi_ = rv[fit_m].min(), rv[fit_m].max()
        self.mean_ = rv[fit_m].mean()
        raw = self._raw_rv(_take(x, val_m))
        self.c_pos_ = float(np.mean(rv[val_m] ** 2 / raw ** 2))    # 보정 배율 c₊(분산 축)
        return self

    def _raw_rv(self, x) -> np.ndarray:
        raw = np.exp(self._predict_log(x))
        return np.where((raw < self.lo_) | (raw > self.hi_), self.mean_, raw)

    def predict_variance(self, x, halt_prob: np.ndarray) -> np.ndarray:
        """주기를 제거한 단위의 분산 예측 c₊(1 − π)p_raw²."""
        return self.c_pos_ * (1 - halt_prob) * self._raw_rv(x) ** 2


def _take(x, mask):
    return x[mask] if isinstance(x, np.ndarray) else x.loc[mask]


class LightGBMRV(LogTargetModel):
    """특징 기반 트리. GARCH-BIC 예측을 특징에 더하면 GARCH-BIC+LightGBM 혼합 모델이 된다."""

    def _fit_raw(self, xf, yf, xv, yv):
        self.model_ = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                                        min_child_samples=50, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, verbose=-1)
        self.model_.fit(xf, yf, eval_set=[(xv, yv)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])

    def _predict_log(self, x):
        return self.model_.predict(x)


class NystroemRidgeRV(LogTargetModel):
    """RBF 커널을 Nystroem 근사로 펼친 뒤 Ridge 회귀. 결측이 있으면 학습 중앙값으로 채운다."""

    def __init__(self, n_components: int = 300, alpha: float = 1.0):
        self.n_components, self.alpha = n_components, alpha

    def _fit_raw(self, xf, yf, xv, yv):
        self.fill_ = xf.median()
        self.model_ = make_pipeline(StandardScaler(), Nystroem(n_components=self.n_components, random_state=0),
                                    Ridge(alpha=self.alpha))
        self.model_.fit(xf.fillna(self.fill_), yf)

    def _predict_log(self, x):
        return self.model_.predict(x.fillna(self.fill_))


class _GRUNet(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gru = torch.nn.GRU(2, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


class GRURV(LogTargetModel):
    """직전 96개 15분봉의 (d, |d|)를 시간 순서대로 읽는 GRU. 검증 오차가 가장 작던 가중치를 복원한다."""

    def __init__(self, hidden: int = 32, epochs: int = 60, patience: int = 5, lr: float = 1e-3,
                 batch: int = 2048, seed: int = 0, device: str | None = None):
        self.hidden, self.epochs, self.patience, self.lr = hidden, epochs, patience, lr
        self.batch, self.seed = batch, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _fit_raw(self, xf, yf, xv, yv):
        torch.manual_seed(self.seed)
        self.scale_ = float(np.std(xf[..., 0])) or 1.0
        self.ymean_ = float(np.mean(yf))
        net = _GRUNet(self.hidden).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        xf_t, yf_t = self._tensor(xf), torch.tensor(yf - self.ymean_, dtype=torch.float32, device=self.device)
        xv_t, yv_t = self._tensor(xv), torch.tensor(yv - self.ymean_, dtype=torch.float32, device=self.device)
        best, best_state, wait = np.inf, None, 0
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            net.train()
            for idx in torch.randperm(len(xf_t), generator=gen).split(self.batch):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(xf_t[idx.to(self.device)]), yf_t[idx.to(self.device)])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = torch.nn.functional.mse_loss(net(xv_t), yv_t).item()
            if val < best:
                best, best_state, wait = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net.eval()

    def _tensor(self, x):
        return torch.tensor(x / self.scale_, dtype=torch.float32, device=self.device)

    def _predict_log(self, x):
        with torch.no_grad():
            return self.net_(self._tensor(x)).cpu().numpy() + self.ymean_
