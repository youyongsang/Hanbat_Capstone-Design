# -*- coding: utf-8 -*-
"""
baseline.py — 분석 및 비교 실험 (단점 7)
────────────────────────────────────────────────────────────
Baseline 모델 3종 구현 + LSTM과 성능 비교표 자동 출력
  1. Naive          : 직전 값을 그대로 예측 (Last-Value)
  2. MovingAverage  : 최근 N개 평균
  3. LinearRegression: 윈도우 내 선형 추세 외삽

evaluate_baselines(y_true) 호출 시 → 비교 DataFrame 반환
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ── Baseline 예측 함수들 ──────────────────────────────────────────────────

def naive_predict(series: np.ndarray) -> np.ndarray:
    """직전 값 = 예측값 (Persistence Model)"""
    return series[:-1]


def moving_average_predict(series: np.ndarray, window: int = 10) -> np.ndarray:
    """최근 window개 평균으로 예측"""
    preds = []
    for i in range(window, len(series)):
        preds.append(series[i - window : i].mean())
    return np.array(preds)


def linear_regression_predict(series: np.ndarray, window: int = 20) -> np.ndarray:
    """window 내 선형 추세를 외삽하여 다음 값 예측"""
    preds = []
    for i in range(window, len(series)):
        X = np.arange(window).reshape(-1, 1)
        y = series[i - window : i]
        reg = LinearRegression().fit(X, y)
        preds.append(reg.predict([[window]])[0])
    return np.array(preds)


# ── 평가 지표 계산 ───────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    mae   = np.mean(np.abs(y_true - y_pred))
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # MAPE (0 분모 방지)
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    # 피크 MAE (상위 20%)
    thresh    = np.quantile(y_true, 0.80)
    peak_mask = y_true >= thresh
    peak_mae  = np.mean(np.abs(y_true[peak_mask] - y_pred[peak_mask])) if peak_mask.sum() > 0 else np.nan
    return {'모델': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
            'MAPE(%)': round(mape, 2), '피크MAE': round(peak_mae, 2)}


def evaluate_baselines(y_test_raw: np.ndarray,
                        lstm_pred: np.ndarray = None) -> pd.DataFrame:
    """
    y_test_raw : 역변환된 실제 RPS 배열
    lstm_pred  : (optional) LSTM 예측값 배열 — 있으면 비교표에 포함

    반환: 각 모델 성능 비교 DataFrame
    """
    rows = []

    # Naive
    n = naive_predict(y_test_raw)
    rows.append(compute_metrics(y_test_raw[1:], n, 'Naive (Last-Value)'))

    # Moving Average
    w_ma = 10
    ma   = moving_average_predict(y_test_raw, window=w_ma)
    rows.append(compute_metrics(y_test_raw[w_ma:], ma, f'MovingAvg (w={w_ma})'))

    # Linear Regression
    w_lr = 20
    lr   = linear_regression_predict(y_test_raw, window=w_lr)
    rows.append(compute_metrics(y_test_raw[w_lr:], lr, f'LinearRegression (w={w_lr})'))

    # LSTM (있을 때)
    if lstm_pred is not None:
        # 길이 맞추기
        min_len = min(len(y_test_raw), len(lstm_pred))
        rows.append(compute_metrics(y_test_raw[:min_len], lstm_pred[:min_len], 'LSTM (우리 모델)'))

    df = pd.DataFrame(rows)

    print("\n" + "="*65)
    print("📊 Baseline vs LSTM 성능 비교")
    print("="*65)
    print(df.to_string(index=False))

    if lstm_pred is not None:
        lstm_row = df[df['모델'] == 'LSTM (우리 모델)']
        best_base = df[df['모델'] != 'LSTM (우리 모델)']['MAE'].min()
        lstm_mae  = float(lstm_row['MAE'].values[0])
        improvement = (best_base - lstm_mae) / best_base * 100

        print(f"\n  최고 Baseline MAE : {best_base:.2f}")
        print(f"  LSTM MAE          : {lstm_mae:.2f}")
        if improvement > 0:
            print(f"  ✅ LSTM이 최고 Baseline 대비 MAE {improvement:.1f}% 개선")
        else:
            print(f"  ⚠️  LSTM이 Baseline보다 {abs(improvement):.1f}% 낮습니다 → 추가 학습 권장")
    print("="*65 + "\n")

    return df
