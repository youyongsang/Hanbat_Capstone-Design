# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 'Bidirectional LSTM' 학습 시작...")
    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except:
        print("❌ 데이터를 찾을 수 없습니다. 전처리를 먼저 수행하세요."); return

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

    print("🧠 모델 학습 중 (300 Epochs)...")
    model.fit(X_train, y_train, epochs=300, batch_size=64, shuffle=True, verbose=1)
    
    model.save('lstm_model.h5')
    print("✅ 모델 저장 완료 (lstm_model.h5)")

if __name__ == "__main__":
    train_ultimate()
