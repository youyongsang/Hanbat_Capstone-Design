import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 극한의 과적합(암기) 버전 학습 준비...")
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 전처리 코드를 먼저 실행하세요.")

    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')

    # --- [핵심 수정] 무식할 정도로 큰 모델 구조 ---
    model = Sequential([
        # 엄청나게 무거운 레이어로 데이터 전체를 memorize하게 만듦
        LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.0), # 드롭아웃 완전 제거 (과적합 유도 핵심)
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(1)
    ])

    # --- [핵심 수정] 무식할 정도로 오차에 집착하는 설정 ---
    # Huber 대신 MSE 사용, 매우 낮은 학습률로 오랫동안 학습
    optimizer = Adam(learning_rate=0.0001) 
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=50, # 거의 멈출 때까지 끝까지 학습
        restore_best_weights=True,
        verbose=1
    )
    
    print("🧠 학습 진행 중 (매우 오래 걸릴 수 있습니다)...")
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500, # 학습 횟수 대폭 증가
        batch_size=16, # 배치 사이즈를 줄여서 데이터 하나하나의 굴곡을 다 보도록 함
        shuffle=True, 
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 극한의 암기 모델 저장 완료!")

if __name__ == "__main__":
    train_ultimate()
