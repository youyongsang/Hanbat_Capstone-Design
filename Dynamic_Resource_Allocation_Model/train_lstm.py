import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 밸런스 조정 LSTM 모델 학습 시작...")
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 전처리 코드를 먼저 실행하세요.")

    # 1. 데이터 로드
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')

    # --- [수정] 강박적인 가중치(Sample Weight) 제거 ---
    # 모델이 특정 구간에 매몰되지 않고 전체적인 흐름과 변동을 모두 배우게 합니다.

    # --- [수정] 모델 아키텍처 경량화 (과적합 방지) ---
    model = Sequential([
        # 노드 수를 적정 수준(64, 32)으로 줄여 모델이 데이터를 암기하지 않고 패턴을 배우게 함
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2), # 일반화 능력을 위해 드롭아웃 유지
        LSTM(32),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    # --- [수정] 학습 파라미터 최적화 ---
    optimizer = Adam(learning_rate=0.001) # 표준 학습률로 복귀
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    print("🧠 학습 진행 중...")
    
    # --- [수정] 배치 사이즈 확대 (16 -> 64) ---
    # 배치 사이즈를 키우면 학습이 훨씬 안정적이며, 바늘 같은 노이즈 속에서 패턴을 더 잘 찾습니다.
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, 
        batch_size=64, # 핵심 수정 포인트: 학습 안정성 강화
        shuffle=True, 
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 밸런스 최적화 모델이 'lstm_model.h5'로 저장되었습니다.")

if __name__ == "__main__":
    train_ultimate()
