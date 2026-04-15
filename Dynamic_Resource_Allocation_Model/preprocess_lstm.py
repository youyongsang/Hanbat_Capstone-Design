import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def preprocess_ultimate(csv_path='sale_event_traffic.csv', window_size=60):
    print("🛠️ [1/3] 데이터 전처리 중...")
    df = pd.read_csv(csv_path)
    df['diff'] = df['target_rps'].diff().fillna(0)
    df['rolling_mean'] = df['target_rps'].rolling(window=10).mean().bfill()
    
    feature_cols = ['target_rps', 'diff', 'rolling_mean']
    data_x = df[feature_cols].values
    data_y = df['target_rps'].values.reshape(-1, 1)

    split_idx = int(len(df) * 0.8)
    train_x_raw, test_x_raw = data_x[:split_idx], data_x[split_idx - window_size:]
    train_y_raw, test_y_raw = data_y[:split_idx], data_y[split_idx - window_size:]

    sx, sy = StandardScaler(), StandardScaler()
    train_x_s = sx.fit_transform(train_x_raw)
    test_x_s = sx.transform(test_x_raw)
    train_y_s = sy.fit_transform(train_y_raw)
    test_y_s = sy.transform(test_y_raw)

    def create_window(x, y):
        X_list, y_list = [], []
        for i in range(len(x) - window_size):
            X_list.append(x[i:i+window_size]); y_list.append(y[i+window_size])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = create_window(train_x_s, train_y_s)
    X_test, y_test = create_window(test_x_s, test_y_s)

    np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test); np.save('y_test.npy', y_test)
    with open('scaler_x.pkl', 'wb') as f: pickle.dump(sx, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(sy, f)
    print("✅ 데이터 준비 완료!")

if __name__ == "__main__":
    preprocess_ultimate()
