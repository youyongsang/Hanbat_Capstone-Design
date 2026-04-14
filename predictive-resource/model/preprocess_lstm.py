import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# =========================
# 경로 설정
# =========================

BASE_DIR   = Path(__file__).resolve().parent.parent   # predictive-resource/
OUTPUT_DIR = Path(__file__).resolve().parent          # predictive-resource/model/

WINDOW_SIZE = 12

def preprocess(csv_path: Path, train_ratio=0.7, val_ratio=0.15):
    if not csv_path.exists():
        raise FileNotFoundError(f"파일 없음: {csv_path}")

    df   = pd.read_csv(csv_path)
    data = df["target_rps"].values.reshape(-1, 1)

    total_len  = len(data)
    val_start  = int(total_len * train_ratio)
    test_start = int(total_len * (train_ratio + val_ratio))

    # 시계열 순서 유지하며 분리
    # val/test는 window 크기만큼 앞으로 당겨서 자름 (과거 데이터 연속성 보장)
    train_raw = data[:val_start]
    val_raw   = data[val_start - WINDOW_SIZE : test_start]
    test_raw  = data[test_start - WINDOW_SIZE :]

    # scaler는 train 데이터로만 fit (데이터 누설 방지)
    scaler = MinMaxScaler()
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw)
    val_scaled   = scaler.transform(val_raw)
    test_scaled  = scaler.transform(test_raw)

    def create_window(dataset):
        X, y = [], []
        if len(dataset) <= WINDOW_SIZE:
            return np.array([]), np.array([])
        for i in range(len(dataset) - WINDOW_SIZE):
            X.append(dataset[i : i + WINDOW_SIZE])
            y.append(dataset[i + WINDOW_SIZE])
        return np.array(X), np.array(y)

    X_train, y_train = create_window(train_scaled)
    X_val,   y_val   = create_window(val_scaled)
    X_test,  y_test  = create_window(test_scaled)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_val.npy",   X_val)
    np.save(OUTPUT_DIR / "y_val.npy",   y_val)
    np.save(OUTPUT_DIR / "X_test.npy",  X_test)
    np.save(OUTPUT_DIR / "y_test.npy",  y_test)

    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"전처리 완료")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")


if __name__ == "__main__":
    target_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        BASE_DIR / "data" / "input" / "sale_event_traffic.csv"
    )
    preprocess(target_csv)
