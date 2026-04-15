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

evaluate_full.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

def evaluate_model_full():
    print("📊 1초부터 1000초까지 전체 트래픽 흐름 시각화 시작...")
    
    # 1. 모델 및 데이터 로드
    model = load_model('lstm_model.h5')
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')
    X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')
    
    # 데이터를 하나로 합치기
    X_all = np.concatenate((X_train, X_val, X_test))
    y_all = np.concatenate((y_train, y_val, y_test))
    
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    # 2. 전체 데이터 예측
    y_pred_scaled = model.predict(X_all, verbose=0)
    
    # 3. 역정규화 (실제 RPS 수치로 복원)
    y_true = scaler_y.inverse_transform(y_all)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 4. 시각화 (기존 양식 그대로)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual Traffic', color='black', alpha=0.7)
    plt.plot(y_pred, label='LSTM Predicted', color='red', linestyle='--')
    
    # 학습과 테스트 경계에 파란 점선 하나 추가 (참고용)
    plt.axvline(x=len(X_train)+len(X_val), color='blue', linestyle=':', label='Test Start')
    
    plt.title('Full Traffic Prediction (1s - 1000s)')
    plt.xlabel('Time (sec)')
    plt.ylabel('RPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    evaluate_model_full()
