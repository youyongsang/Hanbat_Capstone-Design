# -*- coding: utf-8 -*-
"""
robust_scaler.py — 일반화 문제 해결 (단점 4, 5)
────────────────────────────────────────────────────────────
- RobustScaler: 이상치에 강한 스케일러 (IQR 기반)
- OOD(Out-of-Distribution) 탐지: 학습 범위 벗어난 입력 경고
- 스케일 클램핑: MinMaxScaler 범위 외 입력값 왜곡 방지
"""

import pickle
import warnings
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler


class AdaptiveScaler:
    """
    RobustScaler(이상치 내성) + MinMaxScaler(0~1 정규화) 2단계 스케일링.
    OOD 입력 탐지 및 경고 기능 포함.

    사용 예:
        scaler = AdaptiveScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X_test)          # OOD 자동 경고
        X_orig   = scaler.inverse_transform(X_scaled)
    """

    def __init__(self, ood_threshold: float = 3.0):
        """
        ood_threshold: 학습 데이터 IQR 대비 몇 배 벗어나면 OOD로 판단 (기본 3.0)
        """
        self.ood_threshold  = ood_threshold
        self._robust        = RobustScaler()
        self._minmax        = MinMaxScaler(clip=True)   # clip=True → 범위 외 값 0~1로 강제 클램핑
        self._fitted        = False
        # OOD 탐지용 통계 (학습 데이터 기준)
        self._train_q1      = None
        self._train_q3      = None
        self._train_iqr     = None

    def fit(self, X: np.ndarray) -> 'AdaptiveScaler':
        """학습 데이터로 스케일러 적합"""
        self._robust.fit(X)
        X_robust = self._robust.transform(X)
        self._minmax.fit(X_robust)

        # OOD 경계 계산 (원본 기준)
        self._train_q1  = np.percentile(X, 25, axis=0)
        self._train_q3  = np.percentile(X, 75, axis=0)
        self._train_iqr = self._train_q3 - self._train_q1
        self._fitted    = True
        return self

    def transform(self, X: np.ndarray, check_ood: bool = True) -> np.ndarray:
        """스케일 변환 + (선택적) OOD 검사"""
        assert self._fitted, "fit() 먼저 호출하세요."

        if check_ood:
            self._check_ood(X)

        X_robust = self._robust.transform(X)
        return self._minmax.transform(X_robust)

    def inverse_transform_y(self, y_scaled: np.ndarray,
                             scaler_y: 'AdaptiveScaler') -> np.ndarray:
        """y 전용 역변환 (단일 컬럼)"""
        return scaler_y.inverse_transform(y_scaled)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """역변환"""
        assert self._fitted, "fit() 먼저 호출하세요."
        X_robust = self._minmax.inverse_transform(X_scaled)
        return self._robust.inverse_transform(X_robust)

    def _check_ood(self, X: np.ndarray):
        """
        RPS 컬럼(인덱스 0) 기준 OOD 탐지.
        학습 IQR의 ood_threshold 배 이상 벗어나면 경고.
        """
        rps_col = X[:, 0]
        lower_bound = self._train_q1[0] - self.ood_threshold * self._train_iqr[0]
        upper_bound = self._train_q3[0] + self.ood_threshold * self._train_iqr[0]

        ood_low  = np.sum(rps_col < lower_bound)
        ood_high = np.sum(rps_col > upper_bound)

        if ood_low > 0 or ood_high > 0:
            pct = (ood_low + ood_high) / len(rps_col) * 100
            warnings.warn(
                f"\n⚠️  [OOD 경고] 입력 데이터의 {pct:.1f}%가 학습 분포를 벗어났습니다.\n"
                f"   학습 RPS 범위: [{lower_bound:.0f}, {upper_bound:.0f}]\n"
                f"   현재 입력 범위: [{rps_col.min():.0f}, {rps_col.max():.0f}]\n"
                f"   입력값: 하한 미달={ood_low}개, 상한 초과={ood_high}개\n"
                f"   → 예측 정확도가 저하될 수 있습니다. "
                f"     가능하면 현재 데이터로 재학습을 권장합니다.",
                UserWarning,
                stacklevel=3
            )

    def save(self, path_prefix: str):
        """스케일러 저장 (path_prefix_adaptive_scaler.pkl)"""
        with open(f'{path_prefix}_adaptive_scaler.pkl', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path_prefix: str) -> 'AdaptiveScaler':
        """스케일러 로드"""
        with open(f'{path_prefix}_adaptive_scaler.pkl', 'rb') as f:
            return pickle.load(f)


def make_scalers(X_train: np.ndarray, y_train: np.ndarray):
    """
    X, y 각각 AdaptiveScaler 생성 및 적합.
    반환: (scaler_x, scaler_y)
    """
    sx = AdaptiveScaler()
    sy = AdaptiveScaler()
    sx.fit(X_train)
    # y는 1D → reshape 후 fit
    sy.fit(y_train.reshape(-1, 1))
    return sx, sy
