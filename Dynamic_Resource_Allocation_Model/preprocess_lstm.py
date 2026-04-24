# -*- coding: utf-8 -*-
"""
preprocess_lstm.py
- week_traffic.csv  : time_sec, target_rps, day_of_week, is_event, is_weekend
- sale_event_traffic.csv : time_sec, target_rps, scenario, phase
두 CSV 모두 자동 처리. 요일/이벤트 피처를 명시적으로 반영.
"""
import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 60
# 학습에 사용할 피처 컬럼 순서 (6개) — 저장 시 메타데이터로 함께 보관
FEATURE_COLS = ['target_rps', 'diff', 'diff2', 'day_of_week', 'is_event', 'is_weekend']


def build_features(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """CSV 종류에 관계없이 6개 피처 컬럼을 보장한다."""

    # ── 1. 요일/이벤트 컬럼 처리 ──────────────────────────────────────────
    # week_traffic.csv 처럼 이미 컬럼이 있으면 그대로 사용
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 0          # sale_event 는 요일 정보 없음 → 0으로 패딩
    else:
        # 값이 이미 0~6 정수인지 확인, 아니면 변환
        df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)

    if 'is_event' not in df.columns:
        # 파일명이나 scenario 컬럼에 'sale'이 있으면 이벤트로 간주
        is_evt = 1 if ('sale' in os.path.basename(csv_path).lower()
                       or ('scenario' in df.columns)) else 0
        df['is_event'] = is_evt
    else:
        df['is_event'] = pd.to_numeric(df['is_event'], errors='coerce').fillna(0).astype(int)

    if 'is_weekend' not in df.columns:
        # day_of_week 5,6 이 주말(토,일)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    else:
        df['is_weekend'] = pd.to_numeric(df['is_weekend'], errors='coerce').fillna(0).astype(int)

    # ── 2. 변화량 피처 ────────────────────────────────────────────────────
    df['diff']  = df['target_rps'].diff().fillna(0)
    df['diff2'] = df['diff'].diff().fillna(0)

    # ── 3. 이상치 클리핑 (99.5 퍼센타일 기준) ─────────────────────────────
    cap = df['target_rps'].quantile(0.995)
    df['target_rps'] = df['target_rps'].clip(upper=cap)

    return df


def create_windows(scaled_x: np.ndarray, scaled_y: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(scaled_x) - window_size):
        X.append(scaled_x[i : i + window_size])
        y.append(scaled_y[i + window_size])
    return np.array(X), np.array(y)


def preprocess_ultimate(csv_path: str, window_size: int = WINDOW_SIZE):
    print(f"🛠️  [{csv_path}] 전처리 시작...")

    if not os.path.exists(csv_path):
        print(f"❌ 에러: {csv_path} 파일이 없습니다.")
        return False

    df = pd.read_csv(csv_path)
    print(f"   원본 shape: {df.shape}, 컬럼: {list(df.columns)}")

    df = build_features(df, csv_path)

    # 요일별 평균 RPS 출력 (week_traffic 전용 정보)
    if df['day_of_week'].nunique() > 1:
        print("\n📅 요일별 평균 RPS:")
        day_names = ['월', '화', '수', '목', '금', '토', '일']
        for d, grp in df.groupby('day_of_week'):
            print(f"   {day_names[d]}요일 : avg={grp['target_rps'].mean():.0f}, "
                  f"max={grp['target_rps'].max():.0f}, is_event={int(grp['is_event'].iloc[0])}")
        print()

    data_x = df[FEATURE_COLS].values
    data_y = df['target_rps'].values.reshape(-1, 1)

    # 8:2 분할 (Data Leakage 방지 — train 구간만으로 scaler fit)
    split_idx = int(len(df) * 0.8)
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_x.fit(data_x[:split_idx])
    scaler_y.fit(data_y[:split_idx])

    scaled_x = scaler_x.transform(data_x)
    scaled_y = scaler_y.transform(data_y)

    X_train, y_train = create_windows(scaled_x[:split_idx],           scaled_y[:split_idx],           window_size)
    X_test,  y_test  = create_windows(scaled_x[split_idx-window_size:], scaled_y[split_idx-window_size:], window_size)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy',  X_test)
    np.save('y_test.npy',  y_test)
    with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)

    # 피처 컬럼 목록도 저장 → predict_and_allocate.py 에서 재사용
    with open('feature_cols.pkl', 'wb') as f: pickle.dump(FEATURE_COLS, f)

    print(f"✅ 전처리 완료")
    print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"   X_test : {X_test.shape}   y_test : {y_test.shape}")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv'
    preprocess_ultimate(target)
