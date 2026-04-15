import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 현실적인 패턴 학습 시작...")
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')

    # --- [핵심] 밸런스 잡힌 모델 구조 ---
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2), # 암기 방지 (일반화)
        LSTM(32),
        Dense(1)
    ])

    # --- [핵심] Loss를 Huber로 변경 (바늘 보존) ---
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=100, # 무리한 500회 대신 100회
        batch_size=32,
        shuffle=True, 
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 피드백 반영 모델 저장 완료!")

if __name__ == "__main__":
    train_ultimate()
