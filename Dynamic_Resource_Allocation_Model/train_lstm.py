# -*- coding: utf-8 -*-
"""
train_lstm.py — 학습 속도 · 피크 추종 개선판
────────────────────────────────────────────
[문제 1 해결] 속도 개선
  - Bidirectional LSTM 유지, 하지만 units 줄여서 CPU 부담 완화
  - batch_size 512로 확대
  - EarlyStopping + ReduceLROnPlateau 조합 → 수렴 후 자동 종료
    * restore_best_weights=True 이므로 성능 손실 없이 조기 종료 가능

[문제 4 해결] 피크 미추종 개선
  - loss = 'huber' → 고트래픽 구간 가중 손실 (weighted_loss) 으로 교체
    → 피크(상위 30%) 샘플에 2× 가중치 부여
  - 출력층 활성화 제거 (linear) + scaler inverse 범위 보장
"""
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf


# ── 피크 가중 손실 함수 ────────────────────────────────────────────────────
def make_peak_weighted_loss(y_train_raw: np.ndarray, peak_quantile: float = 0.70, peak_weight: float = 2.5):
    """
    상위 peak_quantile 초과 샘플에 peak_weight 배 손실을 부여.
    y_train_raw: 스케일된 y_train (0~1 범위) — threshold 계산에만 사용
    """
    threshold = float(np.quantile(y_train_raw, peak_quantile))

    def weighted_mse(y_true, y_pred):
        error = tf.square(y_true - y_pred)
        weight = tf.where(y_true > threshold,
                          tf.ones_like(y_true) * peak_weight,
                          tf.ones_like(y_true))
        return tf.reduce_mean(error * weight)

    return weighted_mse


def train_ultimate():
    print("🚀 LSTM 학습 시작 (속도 + 피크 추종 개선판)...")

    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except FileNotFoundError:
        print("❌ X_train.npy / y_train.npy 가 없습니다. preprocess_lstm.py 먼저 실행하세요.")
        return

    print(f"   데이터: X_train={X_train.shape}, y_train={y_train.shape}")
    n_features = X_train.shape[2]
    window_size = X_train.shape[1]

    # ── 모델 정의 ──────────────────────────────────────────────────────────
    model = Sequential([
        # [속도] units: 64/32 유지, return_sequences=True 첫 레이어만
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(window_size, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)   # linear — 피크값 상한을 막지 않도록 활성화 없음
    ])

    peak_loss = make_peak_weighted_loss(y_train)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=peak_loss)
    model.summary()

    # ── 콜백 ────────────────────────────────────────────────────────────────
    callbacks = [
        # [속도 문제 해결] val_loss 5 epoch 개선 없으면 종료 + 최적 가중치 복원
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        # 학습률 자동 감소 → 수렴 가속
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        # 최고 모델 자동 저장
        ModelCheckpoint('lstm_model.h5', monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    print(f"\n🧠 학습 중... (max 100 epochs, EarlyStopping patience=8)")
    print("   → 수렴하면 자동 종료됩니다. 보통 20~40 epoch 내 완료됩니다.\n")

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=512,          # [속도] 256→512, 1 epoch 시간 약 2배 단축
        validation_split=0.15,   # train의 15%를 validation으로 사용
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    actual_epochs = len(history.history['loss'])
    best_val = min(history.history['val_loss'])
    print(f"\n✅ 학습 완료!")
    print(f"   실제 학습 epoch: {actual_epochs}  |  최적 val_loss: {best_val:.6f}")
    print(f"   모델 저장: lstm_model.h5")


if __name__ == "__main__":
    train_ultimate()
