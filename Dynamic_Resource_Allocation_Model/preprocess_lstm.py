# -*- coding: utf-8 -*-
"""
preprocess_lstm.py — 입력 문제 완전 해결 (단점 1, 2, 3, 5)
────────────────────────────────────────────────────────────
- SmartCSVLoader: 어떤 CSV든 자동 스키마 변환
- AdaptiveScaler: 이상치 내성 + OOD 탐지
- feature_cols.pkl, metadata.pkl 저장 → 파이프라인 일관성 보장

사용법:
  python preprocess_lstm.py                          # week_traffic.csv
  python preprocess_lstm.py sale_event_traffic.csv
  python preprocess_lstm.py my_custom_traffic.csv    # 어떤 CSV든 OK
"""
import os, sys, pickle
import numpy as np

from schema_adapter import SmartCSVLoader, STANDARD_FEATURES
from robust_scaler import AdaptiveScaler

WINDOW_SIZE = 60


def create_windows(scaled_x: np.ndarray, scaled_y: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(scaled_x) - window_size):
        X.append(scaled_x[i : i + window_size])
        y.append(scaled_y[i + window_size])
    return np.array(X), np.array(y)


def preprocess_ultimate(csv_path: str, window_size: int = WINDOW_SIZE) -> bool:
    print(f"\n{'='*55}")
    print(f"🛠️  전처리 시작: {csv_path}")
    print(f"{'='*55}")

    # 1. SmartCSVLoader → 어떤 CSV든 표준 스키마로 변환
    loader = SmartCSVLoader(csv_path, verbose=True)
    try:
        df = loader.load()
    except Exception as e:
        print(f"❌ CSV 로드/변환 실패: {e}")
        return False

    # 2. 요일별 RPS 통계 출력
    if df['day_of_week'].nunique() > 1:
        day_names = ['월', '화', '수', '목', '금', '토', '일']
        print("📅 요일별 RPS 통계:")
        for d in sorted(df['day_of_week'].unique()):
            grp = df[df['day_of_week'] == d]
            lbl = day_names[d] if d < 7 else str(d)
            print(f"   {lbl}요일: avg={grp['target_rps'].mean():.0f}"
                  f"  max={grp['target_rps'].max():.0f}"
                  f"  is_event={int(grp['is_event'].mode()[0])}")
        print()

    # 3. 피처 / 타깃 분리
    data_x = df[STANDARD_FEATURES].values
    data_y = df['target_rps'].values.reshape(-1, 1)
    split_idx = int(len(df) * 0.8)

    # 4. AdaptiveScaler (train 구간만 fit → Data Leakage 방지)
    scaler_x = AdaptiveScaler()
    scaler_y = AdaptiveScaler()
    scaler_x.fit(data_x[:split_idx])
    scaler_y.fit(data_y[:split_idx])

    scaled_x = scaler_x.transform(data_x, check_ood=False)
    scaled_y = scaler_y.transform(data_y, check_ood=False)

    # 5. 슬라이딩 윈도우 생성
    X_train, y_train = create_windows(scaled_x[:split_idx],
                                      scaled_y[:split_idx], window_size)
    X_test,  y_test  = create_windows(scaled_x[split_idx - window_size:],
                                      scaled_y[split_idx - window_size:], window_size)

    # 6. 저장
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy',  X_test)
    np.save('y_test.npy',  y_test)

    with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)

    metadata = {
        'feature_cols': STANDARD_FEATURES,
        'window_size':  window_size,
        'csv_path':     csv_path,
        'rps_max_train': float(data_y[:split_idx].max()),
        'rps_min_train': float(data_y[:split_idx].min()),
    }
    with open('feature_cols.pkl', 'wb') as f: pickle.dump(STANDARD_FEATURES, f)
    with open('metadata.pkl',     'wb') as f: pickle.dump(metadata, f)

    print(f"✅ 전처리 완료")
    print(f"   X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"   X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    print(f"   학습 RPS 범위: [{metadata['rps_min_train']:.0f}, {metadata['rps_max_train']:.0f}]")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv'
    preprocess_ultimate(target)
