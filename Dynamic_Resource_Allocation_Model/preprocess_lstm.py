import os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_ultimate(csv_path='sale_event_traffic.csv', window_size=10): # 과거 10초만 기억
    print("🛠️ 데이터 전처리 시작 (극한의 과적합 버전)...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # --- [핵심 수정] 모든 스무딩 피처 제거, 원본 데이터만 사용 ---
    # LSTM이 원본 데이터의 굴곡 자체를 암기하도록 유도
    feature_cols = ['target_rps']
    data_x = df[feature_cols].values
    data_y = df['target_rps'].values.reshape(-1, 1)

    # --- [2] 전략적 데이터 분할 (기존 유지) ---
    spike_idx = df['target_rps'].idxmax()
    split_idx = max(int(len(df) * 0.8), spike_idx + window_size)

    min_test_size = int(len(df) * 0.2)
    if len(df) - split_idx < min_test_size:
        split_idx = len(df) - min_test_size
        
    print(f"🚩 데이터 피크 지점: {spike_idx}s | 최종 분할 지점: {split_idx}s")

    full_train_x_raw = data_x[:split_idx]
    full_train_y_raw = data_y[:split_idx]
    test_x_raw = data_x[split_idx - window_size:]
    test_y_raw = data_y[split_idx - window_size:]

    val_split_idx = int(len(full_train_x_raw) * 0.8)
    train_x_raw = full_train_x_raw[:val_split_idx]
    train_y_raw = full_train_y_raw[:val_split_idx]
    val_x_raw = full_train_x_raw[val_split_idx - window_size:]
    val_y_raw = full_train_y_raw[val_split_idx - window_size:]

    # --- [3] 스케일링 ---
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    train_x_scaled = scaler_x.fit_transform(train_x_raw)
    val_x_scaled = scaler_x.transform(val_x_raw)
    test_x_scaled = scaler_x.transform(test_x_raw)
    
    train_y_scaled = scaler_y.fit_transform(train_y_raw)
    val_y_scaled = scaler_y.transform(val_y_raw)
    test_y_scaled = scaler_y.transform(test_y_raw)

    # --- [4] 윈도우 생성 ---
    def create_window(x_data, y_data):
        X, y = [], []
        for i in range(len(x_data) - window_size):
            X.append(x_data[i : i + window_size])
            y.append(y_data[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_window(train_x_scaled, train_y_scaled)
    X_val, y_val = create_window(val_x_scaled, val_y_scaled)
    X_test, y_test = create_window(test_x_scaled, test_y_scaled)

    # --- [5] 파일 저장 ---
    np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val); np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test); np.save('y_test.npy', y_test)
    with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
    
    print(f"✅ 전처리 완료! 피처 {len(feature_cols)}개, 윈도우 {window_size}로 데이터가 생성되었습니다.")

if __name__ == "__main__":
    preprocess_ultimate()
