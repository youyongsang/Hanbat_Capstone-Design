import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 고성능 LSTM 모델 학습 시작 (학습률 0.01 / Shuffle Off)...")
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 학습 데이터가 없습니다. 전처리 코드를 먼저 실행하세요.")

    # --- [1] 데이터 로드 ---
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')

    # --- [2] 모델 아키텍처 (유닛 수 증가) ---
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    # --- [3] 컴파일 (학습률 0.01로 대폭 상향) ---
    optimizer = Adam(learning_rate=0.01) 
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.Huber(delta=1.0), 
        metrics=['mae', 'mse']
    )
    
    print("🧠 100회차 집중 학습을 시작합니다...")
    
    # --- [4] 학습 진행 (Batch 크기 축소 및 Shuffle 끄기) ---
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,      # 100회 학습
        batch_size=16,   # 배치를 16으로 줄여 더 자주 업데이트
        shuffle=False,   # 시계열 순서대로 학습시켜 진동 패턴 보존
        verbose=1
    )
    
    # --- [5] 모델 저장 ---
    model.save('lstm_model.h5')
    print("✅ 100 에포크 학습 완료! 'lstm_model.h5'로 저장되었습니다.")

if __name__ == "__main__":
    train_ultimate()
