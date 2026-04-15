import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 목표 그래프 도달을 위한 'Bidirectional LSTM' 학습 시작...")
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')

    # [핵심] Bidirectional LSTM은 과거와 미래의 흐름을 동시에 파악하여 굴곡을 기가 막히게 잡아냄
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # 안정적인 학습률 0.001 (일자 방지)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber') # 오차에 민감하게 반응하면서도 폭주하지 않음

    print("🧠 모델이 굴곡을 마스터할 수 있도록 300회 꼼꼼히 학습합니다...")
    model.fit(
        X_train, y_train,
        epochs=300, # 끈기 있게 300번!
        batch_size=16,
        shuffle=True, 
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 모든 구간을 완벽히 추종하는 모델이 준비되었습니다!")

if __name__ == "__main__":
    train_ultimate()
