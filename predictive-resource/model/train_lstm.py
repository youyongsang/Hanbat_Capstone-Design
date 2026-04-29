# -*- coding: utf-8 -*-
"""
train_lstm.py — 학습 속도 + 피크 추종 개선 (단점 1, 4)
────────────────────────────────────────────────────────────
[사용자 최적화 설정 반영]
  - batch_size: 64 (학습 속도와 정밀도의 최적 균형)
  - peak_quantile: 0.65 (상위 35% 피크 구간 집중 학습)
  - peak_weight: 3.0 (피크 오차에 대한 강력한 패널티)
  - patience: 10 (충분한 학습 기회 보장)
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

MODEL_DIR = Path(__file__).resolve().parent


def make_peak_weighted_loss(y_train: np.ndarray,
                             peak_quantile: float = 0.65, # 사용자 제안 수치
                             peak_weight: float = 3.0):    # 사용자 제안 수치
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
    print("🚀 LSTM 학습 시작 (사용자 최적화 밸런스판)...")

    try:
        X_train = np.load(MODEL_DIR / 'X_train.npy')
        y_train = np.load(MODEL_DIR / 'y_train.npy')
    except FileNotFoundError:
        print("❌ X_train.npy / y_train.npy 없음. preprocess_lstm.py 먼저 실행하세요.")
        return

    print(f"   데이터: X_train={X_train.shape}, y_train={y_train.shape}")
    window_size, n_features = X_train.shape[1], X_train.shape[2]

    # 모델 정의 (8개 피처 자동 대응)
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(window_size, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)   # linear — 피크값 상한 제거
    ])

    # 수정한 가중치 로직 적용
    peak_loss = make_peak_weighted_loss(y_train, peak_quantile=0.65, peak_weight=3.0)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=peak_loss)
    model.summary()

    callbacks = [
        # patience 10으로 조정하여 모델이 충분히 수렴할 때까지 대기
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        # ReduceLROnPlateau도 patience를 연동하여 조정 (보통 ES의 절반 수준)
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(MODEL_DIR / 'lstm_model.h5'), monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    print(f"\n🧠 학습 중... (max 150 epochs, batch_size=64, patience=10)")
    print("   → 미세한 트래픽 변화를 포착하기 위해 가중치를 자주 업데이트합니다.\n")

    history = model.fit(
        X_train, y_train,
        epochs=150,           # 사용자 제안 수치
        batch_size=64,        # 사용자 제안 수치 (성능/속도 균형)
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
