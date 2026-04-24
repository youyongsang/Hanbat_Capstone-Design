# -*- coding: utf-8 -*-
"""
allocate_resources.py — 전체 파이프라인 통합 실행
────────────────────────────────────────────────────────────
사용법:
  python allocate_resources.py                              # week_traffic.csv
  python allocate_resources.py sale_event_traffic.csv
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

np.random.seed(42)

CSV_PATH    = sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv'
WINDOW_SIZE = 60
FEATURE_COLS = ['target_rps', 'diff', 'diff2', 'day_of_week', 'is_event', 'is_weekend']


# ────────────────────────────────────────────────────────────────
# 1. 데이터 로드 & 피처 생성
# ────────────────────────────────────────────────────────────────
def build_features(df, csv_path):
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 0
    if 'is_event' not in df.columns:
        df['is_event'] = 1 if 'sale' in os.path.basename(csv_path).lower() else 0
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['diff']  = df['target_rps'].diff().fillna(0)
    df['diff2'] = df['diff'].diff().fillna(0)
    cap = df['target_rps'].quantile(0.995)
    df['target_rps'] = df['target_rps'].clip(upper=cap)
    return df

print(f"📂 CSV 로드: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df = build_features(df, CSV_PATH)

data_x = df[FEATURE_COLS].values
data_y = df['target_rps'].values.reshape(-1, 1)
split_idx = int(len(df) * 0.8)

scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
scaler_x.fit(data_x[:split_idx])
scaler_y.fit(data_y[:split_idx])
scaled_x = scaler_x.transform(data_x)
scaled_y = scaler_y.transform(data_y)

def create_windows(sx, sy):
    X, y = [], []
    for i in range(len(sx) - WINDOW_SIZE):
        X.append(sx[i : i + WINDOW_SIZE])
        y.append(sy[i + WINDOW_SIZE])
    return np.array(X), np.array(y)

X_train, y_train = create_windows(scaled_x[:split_idx], scaled_y[:split_idx])
X_test,  y_test  = create_windows(scaled_x[split_idx - WINDOW_SIZE:], scaled_y[split_idx - WINDOW_SIZE:])
print(f"   X_train={X_train.shape}  X_test={X_test.shape}")


# ────────────────────────────────────────────────────────────────
# 2. 피크 가중 손실 + 모델 정의
# ────────────────────────────────────────────────────────────────
peak_threshold = float(np.quantile(y_train, 0.70))

def peak_weighted_loss(y_true, y_pred):
    error  = tf.square(y_true - y_pred)
    weight = tf.where(y_true > peak_threshold,
                      tf.ones_like(y_true) * 2.5,
                      tf.ones_like(y_true))
    return tf.reduce_mean(error * weight)

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(WINDOW_SIZE, len(FEATURE_COLS))),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss=peak_weighted_loss)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint('lstm_model.h5', monitor='val_loss', save_best_only=True, verbose=0),
]

print("\n🧠 모델 학습 시작 (EarlyStopping 적용, max 100 epochs)...")
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=512,
    validation_split=0.15,
    shuffle=True, callbacks=callbacks, verbose=1
)
print(f"✅ 학습 완료 ({len(history.history['loss'])} epochs)\n")


# ────────────────────────────────────────────────────────────────
# 3. 예측 & 평가
# ────────────────────────────────────────────────────────────────
predicted_scaled = model.predict(X_test)
predicted_rps    = scaler_y.inverse_transform(predicted_scaled).flatten()
actual_rps       = scaler_y.inverse_transform(y_test).flatten()
predicted_rps    = np.maximum(predicted_rps, 0)

mae  = mean_absolute_error(actual_rps, predicted_rps)
rmse = np.sqrt(mean_squared_error(actual_rps, predicted_rps))
peak_thresh = np.quantile(actual_rps, 0.80)
peak_mask   = actual_rps >= peak_thresh
peak_mae    = mean_absolute_error(actual_rps[peak_mask], predicted_rps[peak_mask])

print("=== 모델 평가 ===")
print(f"  MAE      : {mae:.2f} RPS")
print(f"  RMSE     : {rmse:.2f} RPS")
print(f"  피크 MAE : {peak_mae:.2f} RPS (상위 20%)")
print("=================\n")


# ────────────────────────────────────────────────────────────────
# 4. 자원 할당 시뮬레이션
# ────────────────────────────────────────────────────────────────
allocated_resources = []
current_containers  = 25
smoothed_capacity   = current_containers * 80

for p in predicted_rps:
    safe_pred        = p * 1.1
    target_containers = max(5, int(np.ceil(safe_pred / 80)))
    if target_containers > current_containers:
        current_containers = target_containers
    elif target_containers < current_containers - 1:
        current_containers = target_containers
    current_containers = min(current_containers, 35)
    smoothed_capacity  = 0.7 * smoothed_capacity + 0.3 * (current_containers * 80)
    allocated_resources.append(smoothed_capacity)


# ────────────────────────────────────────────────────────────────
# 5. 시각화
# ────────────────────────────────────────────────────────────────
font_candidates = ['NanumGothic', 'AppleGothic', 'Malgun Gothic']
for fc in font_candidates:
    if fc in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams['font.family'] = fc
        break
plt.rcParams['axes.unicode_minus'] = False

plot_range = min(1000, len(actual_rps))
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle(f'Production-Ready Auto-Scaling  [{os.path.basename(CSV_PATH)}]',
             fontsize=13, fontweight='bold')

# 상단: 예측 vs 실제
ax1 = axes[0]
ax1.plot(actual_rps[:plot_range],       color='lightgray', alpha=0.8, linewidth=1.0,  label='실제 RPS')
ax1.plot(predicted_rps[:plot_range],    color='royalblue', linestyle='--', linewidth=1.5,
         label=f'LSTM 예측 (MAE={mae:.1f})')
ax1.fill_between(range(plot_range),
                 predicted_rps[:plot_range], actual_rps[:plot_range],
                 where=(actual_rps[:plot_range] > predicted_rps[:plot_range]),
                 alpha=0.15, color='orange', label='예측 부족')
ax1.set_title(f'트래픽 예측  |  MAE={mae:.1f}  RMSE={rmse:.1f}  피크MAE={peak_mae:.1f}')
ax1.set_ylabel('RPS')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# 하단: 자원 할당
ax2 = axes[1]
ax2.plot(actual_rps[:plot_range],          color='lightgray', alpha=0.7, linewidth=1.0, label='실제 RPS')
ax2.plot(allocated_resources[:plot_range], color='crimson',   linewidth=2.0,
         label='할당 자원 (컨테이너×80)')
ax2.set_title('스마트 자원 할당 (Production Logic)')
ax2.set_ylabel('용량 (RPS)')
ax2.set_xlabel('시간 (초)')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
out_path = f'plots/production_{os.path.splitext(os.path.basename(CSV_PATH))[0]}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"✅ 그래프 저장: {out_path}")
