# preprocessor.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config

def create_dataset_from_csv():
    df = pd.read_csv(config.RAW_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    pivot = df.pivot(index="timestamp", columns="node_id", values="bytes").fillna(0)
    sorted_cols = sorted(pivot.columns, key=lambda x: int(x.split("_")[1]))
    raw_values = pivot[sorted_cols].values

    scaled = np.log1p(raw_values)

    X, Y = [], []
    limit = len(scaled) - config.WINDOW_SIZE - config.PRED_HORIZON + 1

    for i in range(limit):
        X.append(scaled[i:i + config.WINDOW_SIZE])
        Y.append(scaled[i + config.WINDOW_SIZE])

    X = np.array(X)                  # (S, T, N)
    Y = np.array(Y)                  # (S, N)

    X = np.transpose(X, (0, 2, 1))    # (S, N, T)
    X = X[..., np.newaxis]            # (S, N, T, F)

    split = int(len(X) * config.TRAIN_RATIO)
    x_train, x_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    np.savez(f"{config.PROCESSED_DATA_DIR}/traffic_data_train.npz",
             x_data=x_train, y_data=y_train)
    np.savez(f"{config.PROCESSED_DATA_DIR}/traffic_data_test.npz",
             x_data=x_test, y_data=y_test)

    print("✅ Preprocessing complete")
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

if __name__ == "__main__":
    create_dataset_from_csv()
