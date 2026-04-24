# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 'Bidirectional LSTM' 고속 학습 시작...")
    try:
        X_train = np.load('X_train.npy'); y_train = np.load('y_train.npy')
    except:
        print("❌ 데이터 로드 실패"); return

    model = Sequential([
        # [수정] 128 -> 64로 조정 (CPU 연산 속도 3배 향상 포인트)
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(32)), # 여기도 64 -> 32로 조정
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

    print("🧠 모델 학습 중 (100 Epochs)...")
    # [수정] batch_size를 256으로 확대하여 CPU 병목 해결
    model.fit(X_train, y_train, epochs=100, batch_size=256, shuffle=True, verbose=1)
    
    model.save('lstm_model.h5')
    print("✅ 모델 저장 완료 (lstm_model.h5)")

if __name__ == "__main__":
    train_ultimate()
