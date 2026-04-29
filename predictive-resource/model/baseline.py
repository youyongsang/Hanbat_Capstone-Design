# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def naive_predict(series: np.ndarray) -> np.ndarray:
    return series[:-1]


def moving_average_predict(series: np.ndarray, window: int = 10) -> np.ndarray:
    preds = []
    for i in range(window, len(series)):
        preds.append(series[i - window : i].mean())
    return np.array(preds)


def linear_regression_predict(series: np.ndarray, window: int = 20) -> np.ndarray:
    preds = []
    for i in range(window, len(series)):
        x = np.arange(window).reshape(-1, 1)
        y = series[i - window : i]
        reg = LinearRegression().fit(x, y)
        preds.append(reg.predict([[window]])[0])
    return np.array(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    thresh = np.quantile(y_true, 0.80)
    peak_mask = y_true >= thresh
    peak_mae = np.mean(np.abs(y_true[peak_mask] - y_pred[peak_mask])) if peak_mask.sum() > 0 else np.nan
    return {
        '모델': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE(%)': round(mape, 2),
        '피크MAE': round(peak_mae, 2),
    }


def evaluate_baselines(y_test_raw: np.ndarray, lstm_pred: np.ndarray = None) -> pd.DataFrame:
    rows = []

    naive = naive_predict(y_test_raw)
    rows.append(compute_metrics(y_test_raw[1:], naive, 'Naive (Last-Value)'))

    ma_window = 10
    ma = moving_average_predict(y_test_raw, window=ma_window)
    rows.append(compute_metrics(y_test_raw[ma_window:], ma, f'MovingAvg (w={ma_window})'))

    lr_window = 20
    lr = linear_regression_predict(y_test_raw, window=lr_window)
    rows.append(compute_metrics(y_test_raw[lr_window:], lr, f'LinearRegression (w={lr_window})'))

    if lstm_pred is not None:
        min_len = min(len(y_test_raw), len(lstm_pred))
        rows.append(compute_metrics(y_test_raw[:min_len], lstm_pred[:min_len], 'LSTM (우리 모델)'))

    df = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("Baseline vs LSTM 성능 비교")
    print("=" * 65)
    print(df.to_string(index=False))

    if lstm_pred is not None:
        lstm_row = df[df['모델'] == 'LSTM (우리 모델)']
        best_base = df[df['모델'] != 'LSTM (우리 모델)']['MAE'].min()
        lstm_mae = float(lstm_row['MAE'].values[0])
        improvement = (best_base - lstm_mae) / best_base * 100

        print(f"\n  최고 Baseline MAE : {best_base:.2f}")
        print(f"  LSTM MAE          : {lstm_mae:.2f}")
        if improvement > 0:
            print(f"  LSTM이 최고 Baseline 대비 MAE {improvement:.1f}% 개선")
        else:
            print(f"  LSTM이 Baseline보다 {abs(improvement):.1f}% 낮습니다 -> 추가 학습 권장")
    print("=" * 65 + "\n")

    return df
