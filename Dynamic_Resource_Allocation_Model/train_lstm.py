# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 성능 최적화 모드로 학습을 시작합니다 (Batch Size: 64)...")
    
    # 데이터 로드
    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 모델 구조 유지
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # 옵티마이저 및 손실 함수 설정
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')

    print(f"🧠 학습 시작: 총 300 에포크 / 배치 사이즈: 64")
    
    # [수정] TypeError를 일으킨 workers, use_multiprocessing 인자를 제거했습니다.
    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=64,
        shuffle=True, 
        verbose=1
    )
    
    # 모델 저장
    model.save('lstm_model.h5')
    print("✅ 모델 저장 완료 (lstm_model.h5)")

if __name__ == "__main__":
    train_ultimate()
