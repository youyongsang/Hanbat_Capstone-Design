# -*- coding: utf-8 -*-
"""
allocate_resources.py — 전체 파이프라인 통합 자동화 (단점 6)
────────────────────────────────────────────────────────────
전처리 → 학습 → 예측 → 자원할당 → 시각화를 단일 진입점에서 실행.
사용자가 순서·중간 파일을 직접 관리할 필요 없음.

사용법:
  python allocate_resources.py                              # week_traffic.csv
  python allocate_resources.py sale_event_traffic.csv
  python allocate_resources.py my_custom.csv               # 어떤 CSV든 OK
"""
import sys, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from schema_adapter import SmartCSVLoader, STANDARD_FEATURES
from robust_scaler import AdaptiveScaler
from baseline import evaluate_baselines

warnings.filterwarnings('ignore', category=UserWarning)
np.random.seed(42)

CSV_PATH    = sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv'
WINDOW_SIZE = 60


# ── 슬라이딩 윈도우 ────────────────────────────────────────────────────────
def create_windows(sx, sy, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(sx) - window_size):
        X.append(sx[i : i + window_size])
        y.append(sy[i + window_size])
    return np.array(X), np.array(y)


# ── 피크 가중 손실 ─────────────────────────────────────────────────────────
def make_peak_loss(y_train, quantile=0.70, weight=2.5):
    threshold = float(np.quantile(y_train, quantile))
    def loss_fn(y_true, y_pred):
        err = tf.square(y_true - y_pred)
        w   = tf.where(y_true > threshold,
                       tf.ones_like(y_true) * weight,
                       tf.ones_like(y_true))
        return tf.reduce_mean(err * w)
    loss_fn.__name__ = 'peak_weighted_mse'
    return loss_fn


# ════════════════════════════════════════════════════════════════
# STEP 1: 데이터 로드 & 전처리
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"📂 STEP 1: 데이터 로드  [{CSV_PATH}]")
print(f"{'='*55}")

loader = SmartCSVLoader(CSV_PATH, verbose=True)
df = loader.load()

data_x    = df[STANDARD_FEATURES].values
data_y    = df['target_rps'].values.reshape(-1, 1)
split_idx = int(len(df) * 0.8)

scaler_x = AdaptiveScaler()
scaler_y = AdaptiveScaler()
scaler_x.fit(data_x[:split_idx])
scaler_y.fit(data_y[:split_idx])

scaled_x = scaler_x.transform(data_x, check_ood=False)
scaled_y = scaler_y.transform(data_y, check_ood=False)

X_train, y_train = create_windows(scaled_x[:split_idx],           scaled_y[:split_idx])
X_test,  y_test  = create_windows(scaled_x[split_idx-WINDOW_SIZE:], scaled_y[split_idx-WINDOW_SIZE:])
print(f"\n   X_train={X_train.shape}  X_test={X_test.shape}")


# ════════════════════════════════════════════════════════════════
# STEP 2: 학습
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"🧠 STEP 2: 모델 학습 (EarlyStopping 적용)")
print(f"{'='*55}\n")

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(WINDOW_SIZE, len(STANDARD_FEATURES))),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss=make_peak_loss(y_train))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint('lstm_model.h5', monitor='val_loss', save_best_only=True, verbose=0),
]

history = model.fit(X_train, y_train, epochs=100, batch_size=512,
                    validation_split=0.15, shuffle=True, callbacks=callbacks, verbose=1)

# 스케일러 저장 (predict_and_allocate 재사용용)
with open('scaler_x.pkl', 'wb') as f: pickle.dump(scaler_x, f)
with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
with open('feature_cols.pkl', 'wb') as f: pickle.dump(STANDARD_FEATURES, f)

actual_epochs = len(history.history['loss'])
print(f"\n✅ 학습 완료: {actual_epochs} epochs\n")


# ════════════════════════════════════════════════════════════════
# STEP 3: 예측 & 평가
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"📊 STEP 3: 예측 & 평가")
print(f"{'='*55}")

def inv_y(arr):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

predicted_rps = np.maximum(inv_y(model.predict(X_test).flatten()), 0)
actual_rps    = inv_y(y_test.flatten())

mae      = mean_absolute_error(actual_rps, predicted_rps)
rmse     = np.sqrt(mean_squared_error(actual_rps, predicted_rps))
nonzero  = actual_rps != 0
mape     = float(np.mean(np.abs((actual_rps[nonzero]-predicted_rps[nonzero])/actual_rps[nonzero]))*100)
p_thresh = np.quantile(actual_rps, 0.80)
p_mask   = actual_rps >= p_thresh
peak_mae = mean_absolute_error(actual_rps[p_mask], predicted_rps[p_mask])

print(f"\n  MAE      : {mae:.2f} RPS")
print(f"  RMSE     : {rmse:.2f} RPS")
print(f"  MAPE     : {mape:.2f} %")
print(f"  피크 MAE : {peak_mae:.2f} RPS")

# Baseline 비교
evaluate_baselines(actual_rps, lstm_pred=predicted_rps)


# ════════════════════════════════════════════════════════════════
# STEP 4: 자원 할당 시뮬레이션
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"⚙️  STEP 4: 자원 할당 시뮬레이션")
print(f"{'='*55}\n")

allocated = []
cur_cont  = 25
sm_cap    = cur_cont * 80

for p in predicted_rps:
    safe_pred   = p * 1.1
    target_cont = max(5, int(np.ceil(safe_pred / 80)))
    if target_cont > cur_cont:
        cur_cont = target_cont
    elif target_cont < cur_cont - 1:
        cur_cont = target_cont
    cur_cont = min(cur_cont, 35)
    sm_cap   = 0.7 * sm_cap + 0.3 * (cur_cont * 80)
    allocated.append(sm_cap)


# ════════════════════════════════════════════════════════════════
# STEP 5: 시각화
# ════════════════════════════════════════════════════════════════
print(f"🎨 STEP 5: 시각화 저장")

font_candidates = ['NanumGothic', 'AppleGothic', 'Malgun Gothic']
for fc in font_candidates:
    if fc in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams['font.family'] = fc
        break
plt.rcParams['axes.unicode_minus'] = False

plot_range = min(1000, len(actual_rps))
fig, axes  = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle(f'Production-Ready Auto-Scaling  [{os.path.basename(CSV_PATH)}]',
             fontsize=13, fontweight='bold')

ax1 = axes[0]
ax1.plot(actual_rps[:plot_range],    color='lightgray', alpha=0.8, lw=1.0, label='실제 RPS')
ax1.plot(predicted_rps[:plot_range], color='royalblue', ls='--', lw=1.5,
         label=f'LSTM 예측 (MAE={mae:.1f})')
ax1.fill_between(range(plot_range),
                 predicted_rps[:plot_range], actual_rps[:plot_range],
                 where=(actual_rps[:plot_range] > predicted_rps[:plot_range]),
                 alpha=0.15, color='orange', label='예측 부족')
ax1.set_title(f'트래픽 예측  |  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%  피크MAE={peak_mae:.1f}')
ax1.set_ylabel('RPS')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(actual_rps[:plot_range],  color='lightgray', alpha=0.7, lw=1.0, label='실제 RPS')
ax2.plot(allocated[:plot_range],   color='crimson',   lw=2.0, label='할당 자원 (컨테이너×80)')
ax2.set_title('스마트 자원 할당')
ax2.set_ylabel('용량 (RPS)')
ax2.set_xlabel('시간 (초)')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
out_path = f"plots/production_{os.path.splitext(os.path.basename(CSV_PATH))[0]}.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 그래프 저장: {out_path}")
print(f"\n{'='*55}")
print(f"🎉 전체 파이프라인 완료!")
print(f"   CSV: {CSV_PATH}")
print(f"   MAE: {mae:.2f} RPS  |  피크MAE: {peak_mae:.2f} RPS")
print(f"{'='*55}\n")
