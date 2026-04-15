import os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_ultimate(csv_path='sale_event_traffic.csv', window_size=60): # 윈도우 60으로 확장
    print("🛠️ 데이터 전처리 시작 (분석 피드백 반영 버전)...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # --- [핵심] 입력 정보 부족 해결: 3중 피처 구성 ---
    df['diff'] = df['target_rps'].diff().fillna(0) # 급변 감지
    df['rolling_mean'] = df['target_rps'].rolling(window=10).mean().bfill() # 흐름 감지
    
    feature_cols = ['target_rps', 'diff', 'rolling_mean']
    data_x = df[feature_cols].values
    data_y = df['target_rps'].values.reshape(-1, 1)

    # 데이터 분할
    split_idx = int(len(df) * 0.8)
    train_x_raw = data_x[:split_idx]
    train_y_raw = data_y[:split_idx]
    test_x_raw = data_x[split_idx - window_size:]
    test_y_raw = data_y[split_idx - window_size:]

    # 스케일링
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    train_x_scaled = scaler_x.fit_transform(train_x_raw)
    test_x_scaled = scaler_x.transform(test_x_raw)
    train_y_scaled = scaler_y.fit_transform(train_y_raw)
    test_y_scaled = scaler_y.transform(test_y_raw)

    def create_window(x_data, y_data):
        X, y = [], []
        for i in range(len(x_data) - window_size):
            X.append(x_data[i : i + window_size])
            y.append(y_data[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_window(train_x_scaled, train_y_scaled)
    X_test, y_test = create_window(test_x_scaled, test_y_scaled)

    # 파일 저장
    np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test); np.save('y_test.npy', y_test)
    with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
    
    print(f"✅ 전처리 완료: 피처 {feature_cols}, 윈도우 {window_size}")

if __name__ == "__main__":
    preprocess_ultimate()
