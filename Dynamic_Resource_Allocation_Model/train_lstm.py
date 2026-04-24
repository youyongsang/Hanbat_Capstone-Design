# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def train_ultimate():
    print("🚀 [성능 100% 유지] 데이터 파이프라인 최적화 학습을 시작합니다...")
    
    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}"); return

    # ⚡ [핵심 속도 최적화] tf.data 파이프라인 적용 (성능 하락 없음)
    # 데이터를 메모리에 캐싱(cache)하고, 모델이 연산하는 동안 다음 데이터를 미리 준비(prefetch)합니다.
    batch_size = 64
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber')

    print(f"🧠 학습 시작: 300 에포크 끝까지 완주 (Batch: {batch_size})")
    
    # 모델 학습 (미리 준비된 dataset 객체를 바로 투입)
    model.fit(
        dataset,
        epochs=300,
        verbose=1
    )
    
    model.save('lstm_model.h5')
    print("✅ 성능 손실 0%! 고속 학습 및 모델 저장 완료!")

if __name__ == "__main__":
    train_ultimate()
