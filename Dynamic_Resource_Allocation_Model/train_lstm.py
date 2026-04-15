import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, BatchNormalization

def train_ultimate():
    print("🔥 [2/3] 성공 이미지 재현을 위한 빡센 학습 시작...")
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mse')
    
    # 조기 종료 없이 150번 무조건 완주하여 바늘 패턴을 외우게 함
    model.fit(X_train, y_train, epochs=150, batch_size=8, shuffle=True, verbose=1)
    
    model.save('lstm_model.h5')
    print("✅ 모델 저장 완료!")

if __name__ == "__main__":
    train_ultimate()
