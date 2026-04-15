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

    # 4. 시각화 
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual Traffic', color='black', alpha=0.7)
    plt.plot(y_pred, label='LSTM Predicted', color='red', linestyle='--')
    
    # 학습과 테스트 경계에 파란 점선 하나 추가
    plt.axvline(x=len(X_train)+len(X_val), color='blue', linestyle=':', label='Test Start')
    
    plt.title('Full Traffic Prediction (1s - 1000s) - Ultimate Model')
    plt.xlabel('Time (sec)')
    plt.ylabel('RPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    evaluate_model_full()
