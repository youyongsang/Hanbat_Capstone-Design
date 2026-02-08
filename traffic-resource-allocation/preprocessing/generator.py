# generator.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config

def generate_synthetic_traffic():
    np.random.seed(config.RANDOM_SEED)

    timestamps = pd.date_range(
        start="2026-01-19 09:00:00",
        periods=config.TOTAL_SAMPLES,
        freq=f"{config.INTERVAL_SEC}s"
    )

    data = []
    time_idx = np.arange(config.TOTAL_SAMPLES)
    base_pattern = 1000 * (np.sin(time_idx * 0.01) + 1) + 500

    for node_idx in range(config.NUM_NODES):
        node_id = f"node_{node_idx}"
        usage_factor = 2.0 if node_idx % 2 == 0 else 0.5

        noise = np.random.normal(0, 100, size=config.TOTAL_SAMPLES)
        traffic = base_pattern * usage_factor + noise

        burst_idx = np.random.choice(
            config.TOTAL_SAMPLES,
            size=int(config.TOTAL_SAMPLES * 0.01),
            replace=False
        )
        traffic[burst_idx] *= 5.0
        traffic = np.maximum(traffic, 0).astype(int)

        for ts, val in zip(timestamps, traffic):
            data.append([ts, node_id, val])

    df = pd.DataFrame(data, columns=["timestamp", "node_id", "bytes"])
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)

    print(f"✅ Raw traffic generated: {config.RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_traffic()
