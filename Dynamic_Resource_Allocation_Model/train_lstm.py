import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, BatchNormalization

def train_ultimate():
    print("🔥 [재현 모드] 성공 이미지와 똑같은 그래프를 목표로 학습을 시작합니다...")
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    
    model = Sequential([
        # 양방향으로 데이터를 훑어 바늘의 '위치'를 정확히 파악
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # 과감하게 학습하도록 오차 함수와 학습률 설정
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')
    
    print("🧠 바늘 패턴 주입 중... (150회 강제 완주 시작)")
    model.fit(
        X_train, y_train,
        epochs=150, 
        batch_size=8, # 데이터를 아주 작게 쪼개서 요동에 극도로 예민하게 만듦
        shuffle=True,
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 성공 모델 생성 완료! 이제 evaluate_full.py를 실행하세요.")

if __name__ == "__main__":
    train_ultimate()
