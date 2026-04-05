import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess(csv_path, window_size=12, train_ratio=0.7, val_ratio=0.15):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 파일 없음: {csv_path}")

    df = pd.read_csv(csv_path)
    data = df['target_rps'].values.reshape(-1, 1)

    total_len = len(data)
    val_start = int(total_len * train_ratio)
    test_start = int(total_len * (train_ratio + val_ratio))

    # 1. 시계열 순서 유지하며 분리 (Train, Validation, Test)
    # [핵심] Test와 Val이 첫 예측을 할 때 필요한 이전 과거 데이터(Tail)를 포함시키기 위해 앞으로 당겨서 자름
    train_raw = data[:val_start]
    val_raw = data[val_start - window_size : test_start]
    test_raw = data[test_start - window_size :]

    # 2. Scaler: 오직 Train 데이터로만 fit (Test 정보 누설 완벽 차단)
    scaler = MinMaxScaler()
    scaler.fit(train_raw)
    
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    # 3. 윈도우 생성 함수
    def create_window(dataset):
        X, y = [], []
        if len(dataset) <= window_size:
            return np.array([]), np.array([])
        for i in range(len(dataset) - window_size):
            X.append(dataset[i : i + window_size])
            y.append(dataset[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_window(train_scaled)
    X_val, y_val = create_window(val_scaled)
    X_test, y_test = create_window(test_scaled)

    # 4. 데이터 저장
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"✅ 전처리 완료 (Train/Val/Test 분리 & Train Tail 포함)")
    print(f" - Train: {X_train.shape}")
    print(f" - Val  : {X_val.shape}")
    print(f" - Test : {X_test.shape}\n")

if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else 'sale_event_traffic.csv'
    preprocess(target_csv)
