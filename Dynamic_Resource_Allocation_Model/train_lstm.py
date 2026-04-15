import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 고성능 LSTM 모델 학습 준비 (고변동성 완벽 추적)...")
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 학습 데이터가 없습니다. 전처리 코드를 먼저 실행하세요.")

    # --- [1] 데이터 로드 ---
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')

    # --- [2] 모델 아키텍처 (변동성 학습 강화) ---
    model = Sequential([
        # 노드를 128로 늘려 고주파 패턴을 더 잘 캐치하도록 수정
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1), # 피팅력을 높이기 위해 드롭아웃 감소 (0.2 -> 0.1)
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1) # 최종 예측
    ])

    # --- [3] 컴파일 (MSE와 조금 더 높은 학습률) ---
    # Huber 대신 MSE를 사용하여 피크(이상치) 오차에 매우 민감하게 반응하게 함
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss='mse', # 핵심 변경 포인트
        metrics=['mae']
    )
    
    # --- [4] 콜백 설정 ---
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    print("🧠 모델 학습 시작...")
    
    # --- [5] 학습 진행 ---
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, 
        batch_size=32, 
        shuffle=True, # 윈도우 샘플 셔플링
        callbacks=[early_stop],
        verbose=1
    )
    
    # --- [6] 모델 저장 ---
    model.save('lstm_model.h5')
    print("✅ 최고 성능의 모델이 'lstm_model.h5'로 저장되었습니다!")

if __name__ == "__main__":
    train_ultimate()
