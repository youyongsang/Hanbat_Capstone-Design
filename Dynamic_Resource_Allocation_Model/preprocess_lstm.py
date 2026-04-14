import os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_ultimate(csv_path='sale_event_traffic.csv', window_size=60):
    print("🛠️ 데이터 전처리 시작...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # --- [1] Feature Engineering (파생 변수 생성) ---
    # 10초 이동평균선: 전체적인 흐름 파악
    df['moving_avg'] = df['target_rps'].rolling(window=10).mean()
    # 변화량(diff) 및 3초 평활화: 가속도 파악 및 노이즈 제거
    df['diff'] = df['target_rps'].diff().fillna(0).rolling(window=3).mean()
    
    # --- [2] 동적 Clipping 적용 (하드코딩 제거) ---
    diff_upper = df['diff'].quantile(0.99)
    diff_lower = df['diff'].quantile(0.01)
    df['diff'] = df['diff'].clip(lower=diff_lower, upper=diff_upper)
    
    # 결측치 처리 (최신 Pandas 권장 문법)
    df = df.ffill().bfill()

    feature_cols = ['target_rps', 'moving_avg', 'diff']
    data_x = df[feature_cols].values
    data_y = df['target_rps'].values.reshape(-1, 1)

    # --- [3] 전략적 데이터 분할 (Spike 보장 & Test 크기 보장) ---
    spike_idx = df['target_rps'].idxmax()
    # 기본적으로 80%를 자르되, Spike가 뒤에 있다면 Spike + window_size까지 Train으로 확보
    split_idx = max(int(len(df) * 0.8), spike_idx + window_size)

    # Test 데이터 최소 20% 보장 (평가의 객관성 확보)
    min_test_size = int(len(df) * 0.2)
    if len(df) - split_idx < min_test_size:
        split_idx = len(df) - min_test_size
    
    print(f"🚩 데이터 피크 지점: {spike_idx}s | 최종 분할 지점: {split_idx}s")

    # 분할 수행
    full_train_x_raw = data_x[:split_idx]
    full_train_y_raw = data_y[:split_idx]
    test_x_raw = data_x[split_idx - window_size:]
    test_y_raw = data_y[split_idx - window_size:]

    # Train / Val 분리 (내부에서 8:2)
    val_split_idx = int(len(full_train_x_raw) * 0.8)
    train_x_raw = full_train_x_raw[:val_split_idx]
    train_y_raw = full_train_y_raw[:val_split_idx]
    val_x_raw = full_train_x_raw[val_split_idx - window_size:]
    val_y_raw = full_train_y_raw[val_split_idx - window_size:]

    # --- [4] 스케일링 (StandardScaler로 Outlier 대응 강화) ---
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    train_x_scaled = scaler_x.fit_transform(train_x_raw)
    val_x_scaled = scaler_x.transform(val_x_raw)
    test_x_scaled = scaler_x.transform(test_x_raw)
    
    train_y_scaled = scaler_y.fit_transform(train_y_raw)
    val_y_scaled = scaler_y.transform(val_y_raw)
    test_y_scaled = scaler_y.transform(test_y_raw)

    # --- [5] 윈도우 생성 ---
    def create_window(x_data, y_data):
        X, y = [], []
        for i in range(len(x_data) - window_size):
            X.append(x_data[i : i + window_size])
            y.append(y_data[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_window(train_x_scaled, train_y_scaled)
    X_val, y_val = create_window(val_x_scaled, val_y_scaled)
    X_test, y_test = create_window(test_x_scaled, test_y_scaled)

    print(f"📊 최종 데이터셋 -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- [6] 파일 저장 ---
    np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val); np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test); np.save('y_test.npy', y_test)
    with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
    
    print("✅ 전처리 및 파일 저장 완료!")

if __name__ == "__main__":
    preprocess_ultimate()
