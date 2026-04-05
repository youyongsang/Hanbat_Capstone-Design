import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def train():
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 학습 데이터가 없습니다. 전처리를 먼저 수행하세요.")

    # Train, Validation 데이터 로드
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')

    # 가이드라인 8.4 모델 구조
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("🚀 학습 시작 (Validation Set 적용 및 시계열 순서 유지)...")
    model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), # 분리된 검증셋 명시적 사용
        epochs=50, 
        batch_size=32, 
        shuffle=False, # 시계열 특성 유지
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 모델 저장 완료: lstm_model.h5")

if __name__ == "__main__":
    train()
