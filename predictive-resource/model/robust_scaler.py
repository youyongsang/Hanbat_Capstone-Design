# -*- coding: utf-8 -*-
import pickle
import warnings

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class AdaptiveScaler:
    def __init__(self, ood_threshold: float = 3.0):
        self.ood_threshold = ood_threshold
        self._robust = RobustScaler()
        self._minmax = MinMaxScaler(clip=True)
        self._fitted = False
        self._train_q1 = None
        self._train_q3 = None
        self._train_iqr = None

    def fit(self, x: np.ndarray) -> 'AdaptiveScaler':
        self._robust.fit(x)
        x_robust = self._robust.transform(x)
        self._minmax.fit(x_robust)

        self._train_q1 = np.percentile(x, 25, axis=0)
        self._train_q3 = np.percentile(x, 75, axis=0)
        self._train_iqr = self._train_q3 - self._train_q1
        self._fitted = True
        return self

    def transform(self, x: np.ndarray, check_ood: bool = True) -> np.ndarray:
        assert self._fitted, "fit() 먼저 호출하세요."
        if check_ood:
            self._check_ood(x)
        x_robust = self._robust.transform(x)
        return self._minmax.transform(x_robust)

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        assert self._fitted, "fit() 먼저 호출하세요."
        x_robust = self._minmax.inverse_transform(x_scaled)
        return self._robust.inverse_transform(x_robust)

    def _check_ood(self, x: np.ndarray):
        rps_col = x[:, 0]
        lower_bound = self._train_q1[0] - self.ood_threshold * self._train_iqr[0]
        upper_bound = self._train_q3[0] + self.ood_threshold * self._train_iqr[0]

        ood_low = np.sum(rps_col < lower_bound)
        ood_high = np.sum(rps_col > upper_bound)

        if ood_low > 0 or ood_high > 0:
            pct = (ood_low + ood_high) / len(rps_col) * 100
            warnings.warn(
                f"\n⚠️  [OOD 경고] 입력 데이터의 {pct:.1f}%가 학습 분포를 벗어났습니다.\n"
                f"   학습 RPS 범위: [{lower_bound:.0f}, {upper_bound:.0f}]\n"
                f"   현재 입력 범위: [{rps_col.min():.0f}, {rps_col.max():.0f}]\n"
                f"   입력값: 하한 미달={ood_low}개, 상한 초과={ood_high}개\n"
                f"   → 예측 정확도가 저하될 수 있습니다. 가능하면 현재 데이터로 재학습을 권장합니다.",
                UserWarning,
                stacklevel=3,
            )

    def save(self, path_prefix: str):
        with open(f'{path_prefix}_adaptive_scaler.pkl', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path_prefix: str) -> 'AdaptiveScaler':
        with open(f'{path_prefix}_adaptive_scaler.pkl', 'rb') as f:
            return pickle.load(f)
