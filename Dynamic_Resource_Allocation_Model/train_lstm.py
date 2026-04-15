import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_ultimate():
    print("🚀 바늘 피크 추적을 위한 딥러닝 학습 시작...")
    
    # 1. 데이터 로드 확인
    if not os.path.exists('X_train.npy'):
        raise FileNotFoundError("❌ 전처리(preprocess_lstm.py)를 먼저 실행하세요.")

    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    # 전처리에서 X_val을 만들지 않았다면 아래와 같이 분리하거나 validation_split 사용
    # 여기서는 안전하게 전체 데이터의 일부를 검증용으로 씁니다.

    # 2. 모델 구조 (피드백 반영: 64-32 레이어 + Dropout 0.2)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    # 3. 설정 (피드백 반영: Huber Loss 사용)
    # Huber Loss는 MSE보다 이상치(Peak)에 유연하면서도 패턴을 잘 잡습니다.
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae', 'mse'])
    
    # --- [핵심 수정] 조기 종료 방지 설정 ---
    early_stop = EarlyStopping(
        monitor='loss',        # val_loss가 아닌 실제 학습 오차를 기준
        patience=50,           # 오차가 안 줄어들어도 50번은 더 기회를 줌 (포기 방지)
        restore_best_weights=True,
        verbose=1
    )
    
    print("🧠 모델이 바늘 패턴을 학습 중입니다. 잠시만 기다려주세요...")
    
    # 4. 학습 실행
    history = model.fit(
        X_train, y_train,
        epochs=150,            # 150번까지 충분히 기회를 줌
        batch_size=16,         # 배치를 줄여 데이터 하나하나의 요동을 세밀하게 학습
        validation_split=0.1,  # 10%는 검증용으로 사용
        shuffle=True, 
        callbacks=[early_stop],
        verbose=1
    )
    
    # 5. 모델 저장
    model.save('lstm_model.h5')
    print("✅ 최고 성능의 모델이 'lstm_model.h5'로 저장되었습니다!")

if __name__ == "__main__":
    train_ultimate()
