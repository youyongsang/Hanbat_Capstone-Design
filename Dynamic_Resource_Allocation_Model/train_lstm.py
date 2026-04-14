import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 고성능 LSTM 모델 학습 준비...")
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 학습 데이터가 없습니다. 전처리 코드를 먼저 실행하세요.")

    # --- [1] 데이터 로드 ---
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')

    # --- [2] 모델 아키텍처 (과적합 방지 및 시계열 패턴 심층 학습) ---
    model = Sequential([
        # 첫 번째 층: 흐름 전체를 다음 레이어로 넘김
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        # 두 번째 층: 엑기스 특징 추출
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        # 출력층: Linear (활성화 함수 없이 값 자체를 예측)
        Dense(1)
    ])

    # --- [3] 컴파일 (Huber Loss 및 Adam 최적화) ---
    # Huber Loss: 오차가 작을 땐 MSE(정밀도), 오차가 클 땐 MAE(이상치 강건성)로 작동
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.Huber(delta=1.0), 
        metrics=['mae', 'mse']
    )
    
    # --- [4] 콜백 설정 (충분한 인내심 부여) ---
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    print("🧠 모델 학습 시작...")
    
    # --- [5] 학습 진행 (배치 사이즈 32, 배치 셔플링 적용) ---
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, 
        batch_size=32, 
        shuffle=True, # 윈도우 샘플 간의 순서를 섞어 Local Minima 방지
        callbacks=[early_stop],
        verbose=1
    )
    
    # --- [6] 모델 저장 ---
    model.save('lstm_model.h5')
    print("✅ 최고 성능의 모델이 'lstm_model.h5'로 저장되었습니다!")

if __name__ == "__main__":
    train_ultimate()
