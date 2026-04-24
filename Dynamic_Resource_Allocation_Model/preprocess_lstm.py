# -*- coding: utf-8 -*-
import os, pickle, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_ultimate(csv_path, window_size=60):
    print(f"🛠️ [{csv_path}] 범용 데이터 전처리 시작...")
    if not os.path.exists(csv_path):
        print(f"❌ 에러: {csv_path} 파일이 존재하지 않습니다."); return

    df = pd.read_csv(csv_path)
    
    # [핵심] 부족한 컬럼 자동 생성 (표준 6개 피처 규격 맞춤)
    if 'day_of_week' not in df.columns: df['day_of_week'] = 0
    if 'is_event' not in df.columns: 
        # 시나리오에 'sale'이 포함되거나 컬럼이 있으면 이벤트로 간주
        df['is_event'] = 1 if ('scenario' in df.columns or 'sale' in csv_path) else 0
    if 'is_weekend' not in df.columns: df['is_weekend'] = 0
    
    # 변화 속도 및 가속도 피처 생성
    df['diff'] = df['target_rps'].diff().fillna(0)
    df['diff2'] = df['diff'].diff().fillna(0)
    
    feature_cols = ['target_rps', 'diff', 'diff2', 'day_of_week', 'is_event', 'is_weekend']
    data_x = df[feature_cols].values
    data_y = df['target_rps'].values.reshape(-1, 1)

    # 8:2 분할 및 스케일링 (Data Leakage 방지)
    split_idx = int(len(df) * 0.8)
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_x.fit(data_x[:split_idx])
    scaler_y.fit(data_y[:split_idx])

    scaled_x = scaler_x.transform(data_x)
    scaled_y = scaler_y.transform(data_y)

    def create_window(raw_x, raw_y):
        X, y = [], []
        for i in range(len(raw_x) - window_size):
            X.append(raw_x[i : i + window_size])
            y.append(raw_y[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_window(scaled_x[:split_idx], scaled_y[:split_idx])
    X_test, y_test = create_window(scaled_x[split_idx-window_size:], scaled_y[split_idx-window_size:])

    # 파일 저장
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
    
    print(f"✅ 전처리 완료 (X_train: {X_train.shape}, X_test: {X_test.shape})")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv'
    preprocess_ultimate(target_file)
