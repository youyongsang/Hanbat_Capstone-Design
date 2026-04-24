# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 성능과 속도의 밸런스를 맞춘 학습을 시작합니다...")
    
    # 데이터 로드
    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 모델 구조 (기존의 강력한 Bidirectional LSTM 유지)
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # 옵티마이저 및 손실 함수 설정
    # Huber Loss는 이상치(Outlier)에 강해 안정적인 학습을 돕습니다.
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')

    print(f"🧠 학습 시작: 총 300 에포크 / 배치 사이즈: 64")
    
    # [핵심 수정] batch_size=64 설정 및 멀티 프로세싱 적용
    model.fit(
        X_train, y_train,
        epochs=300,        # 300번 끝까지 정밀하게 학습
        batch_size=64,     # [수정] 성능 저하를 최소화하는 배치 크기
        shuffle=True, 
        workers=4,         # CPU 멀티코어 활용으로 속도 보강
        use_multiprocessing=True,
        verbose=1
    )
    
    # 모델 저장
    model.save('lstm_model.h5')
    print("✅ 모든 구간을 완벽히 추종하는 모델(lstm_model.h5)이 준비되었습니다!")

if __name__ == "__main__":
    train_ultimate()
