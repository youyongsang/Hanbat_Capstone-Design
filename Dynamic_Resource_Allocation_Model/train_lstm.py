import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 고성능 LSTM 모델 학습 시작 (Full Training 모드)...")
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 학습 데이터가 없습니다. 전처리 코드를 먼저 실행하세요.")

    # --- [1] 데이터 로드 ---
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')

    # --- [2] 모델 아키텍처 ---
    # 조금 더 깊게 학습할 수 있도록 유닛 수를 유지하고 구조를 탄탄히 유지합니다.
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    # --- [3] 컴파일 ---
    # 학습률을 0.001로 설정하여 피크를 더 적극적으로 쫓아가게 합니다.
    optimizer = Adam(learning_rate=0.001) 
    model.compile(
        optimizer=optimizer, 
        loss=tf.keras.losses.Huber(delta=1.0), 
        metrics=['mae', 'mse']
    )
    
    print("🧠 조기 종료 없이 150회차까지 끝까지 학습합니다...")
    
    # --- [4] 학습 진행 (EarlyStopping 제거) ---
    # epochs를 150 정도로 설정하여 모델이 데이터에 충분히 젖어들게 만듭니다.
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,      # 150번 끝까지 학습
        batch_size=32, 
        shuffle=True, 
        verbose=1
    )
    
    # --- [5] 모델 저장 ---
    model.save('lstm_model.h5')
    print("✅ 150 에포크 학습 완료! 'lstm_model.h5'로 저장되었습니다.")

if __name__ == "__main__":
    train_ultimate()
