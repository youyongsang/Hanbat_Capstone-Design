# -*- coding: utf-8 -*-
"""
train_lstm.py — 학습 속도 + 피크 추종 개선 (단점 1, 4)
────────────────────────────────────────────────────────────
[속도 개선]
  - batch_size 512, EarlyStopping(patience=8) + ReduceLROnPlateau
  - restore_best_weights=True → 성능 손실 없이 조기 종료

[피크 추종 개선]
  - peak_weighted_loss: 상위 30% 샘플에 2.5× 가중 손실
  - 출력층 linear 활성화 → 피크 상한 제거

사용법:
  python train_lstm.py
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def make_peak_weighted_loss(y_train: np.ndarray,
                             peak_quantile: float = 0.70,
                             peak_weight: float = 2.5):
    """
    상위 peak_quantile 초과 샘플에 peak_weight 배 손실 부여.
    → 피크 구간 예측 정확도 집중 향상
    """
    threshold = float(np.quantile(y_train, peak_quantile))

    def weighted_mse(y_true, y_pred):
        error  = tf.square(y_true - y_pred)
        weight = tf.where(y_true > threshold,
                          tf.ones_like(y_true) * peak_weight,
                          tf.ones_like(y_true))
        return tf.reduce_mean(error * weight)

    weighted_mse.__name__ = 'peak_weighted_mse'
    return weighted_mse


def train_ultimate():
    print("🚀 LSTM 학습 시작 (속도 + 피크 추종 개선판)...")

    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except FileNotFoundError:
        print("❌ X_train.npy / y_train.npy 없음. preprocess_lstm.py 먼저 실행하세요.")
        return

    print(f"   데이터: X_train={X_train.shape}, y_train={y_train.shape}")
    window_size, n_features = X_train.shape[1], X_train.shape[2]

    # 모델 정의
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(window_size, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)   # linear — 피크값 상한 없음
    ])

    peak_loss = make_peak_weighted_loss(y_train)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=peak_loss)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint('lstm_model.h5', monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    print("\n🧠 학습 중... (max 100 epochs, EarlyStopping patience=8)")
    print("   → 수렴하면 자동 종료. 보통 20~40 epoch 내 완료됩니다.\n")

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=512,
        validation_split=0.15,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    actual_epochs = len(history.history['loss'])
    best_val = min(history.history['val_loss'])
    print(f"\n✅ 학습 완료: {actual_epochs} epochs  |  최적 val_loss: {best_val:.6f}")
    print("   모델 저장: lstm_model.h5")


if __name__ == "__main__":
    train_ultimate()
