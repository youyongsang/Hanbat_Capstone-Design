import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 하드코어 LSTM 모델 학습 준비...")
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 전처리 코드를 먼저 실행하세요.")

    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')

    # --- [핵심] 트래픽이 높은 구간에 가중치 5배 부여 ---
    # 상위 30%의 높은 RPS 구간을 못 맞추면 엄청난 페널티를 받게 함
    sample_weights = np.ones(len(y_train))
    high_val_idx = np.where(y_train > np.percentile(y_train, 70))[0]
    sample_weights[high_val_idx] = 5.0 

    # --- 모델 아키텍처 (더 깊고 무겁게) ---
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.05), # 데이터에 거의 달라붙게(Overfitting 유도) 만듦
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.0005) # 세밀한 학습을 위해 학습률 낮춤
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=20, # 충분히 오래 학습하도록 늘림
        restore_best_weights=True,
        verbose=1
    )
    
    print("🧠 모델 학습 시작...")
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300, 
        batch_size=16, # 배치 사이즈를 줄여서 데이터 하나하나의 굴곡을 다 보도록 함
        sample_weight=sample_weights, # 가중치 적용!
        shuffle=True, 
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 하드코어 피팅 모델 저장 완료!")

if __name__ == "__main__":
    train_ultimate()
