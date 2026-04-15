import os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_ultimate(csv_path='sale_event_traffic.csv', window_size=5): # 윈도우를 더 줄여 반응성 극대화
    print("🛠️ 초정밀 추종을 위한 전처리 시작...")
    df = pd.read_csv(csv_path)
    
    # [핵심] 트래픽의 변화 속도(diff)와 가속도(diff2)를 추가하여 모델에게 '움직임'을 가르침
    df['diff'] = df['target_rps'].diff().fillna(0)
    df['diff2'] = df['diff'].diff().fillna(0)
    
    feature_cols = ['target_rps', 'diff', 'diff2']
    data_x = df[feature_cols].values
    data_y = df['target_rps'].values.reshape(-1, 1)

    # 8:2 분할 (일반화 성능 검증용)
    split_idx = int(len(df) * 0.8)
    
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_x.fit(data_x)
    scaler_y.fit(data_y)

    def create_window(raw_x, raw_y):
        scaled_x = scaler_x.transform(raw_x)
        scaled_y = scaler_y.transform(raw_y)
        X, y = [], []
        for i in range(len(scaled_x) - window_size):
            X.append(scaled_x[i : i + window_size])
            y.append(scaled_y[i + window_size])
        return np.array(X), np.array(y)

    X_train, y_train = create_window(data_x[:split_idx], data_y[:split_idx])
    X_test, y_test = create_window(data_x[split_idx-window_size:], data_y[split_idx-window_size:])

    np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test); np.save('y_test.npy', y_test)
    with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
    print(f"✅ 데이터 준비 완료! (Train: {len(X_train)}, Test: {len(X_test)})")

if __name__ == "__main__":
    preprocess_ultimate()
