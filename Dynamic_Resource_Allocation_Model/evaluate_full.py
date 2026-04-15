import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow.keras.models import load_model

def evaluate_model_full():
    print("🛠️ [전체 흐름 시각화] 데이터 로드 및 예측 시작...")
    
    # 1. 파일 존재 여부 확인 및 로드
    files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy', 'lstm_model.h5', 'scaler_y.pkl']
    for f in files:
        if not os.path.exists(f):
            print(f"❌ 파일이 없습니다: {f}")
            return

    model = load_model('lstm_model.h5')
    
    # 모든 데이터셋 로드
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')
    X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')
    
    with open('scaler_y.pkl', 'rb') as f: 
        sy = pickle.load(f)

    # 2. 데이터 병합 (Train -> Val -> Test 순서로 이어붙임)
    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)

    # 3. 전체 데이터 예측
    print("🔮 모델 예측 중...")
    y_pred_scaled = model.predict(X_all, verbose=0)

    # 4. 역정규화 (실제 RPS 수치로 변환)
    y_true = sy.inverse_transform(y_all)
    y_pred = sy.inverse_transform(y_pred_scaled)

    # 5. 그래프 시각화
    plt.figure(figsize=(18, 7))
    
    # 실제 값과 예측 값
    plt.plot(y_true, label='Actual Traffic (Ground Truth)', color='black', alpha=0.5, linewidth=1)
    plt.plot(y_pred, label='LSTM Predicted Traffic', color='red', linestyle='--', alpha=0.8)

    # 구분을 위한 수직선 추가
    train_end = len(X_train)
    val_end = len(X_train) + len(X_val)
    
    plt.axvline(x=train_end, color='blue', linestyle=':', label='Validation Start')
    plt.axvline(x=val_end, color='green', linestyle=':', label='Test Start')

    # 그래프 꾸미기
    plt.title('Full Timeline Traffic Prediction: From Training to Test', fontsize=15)
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('RPS (Requests Per Second)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    print("✅ 그래프 출력이 완료되었습니다.")
    plt.show()

if __name__ == "__main__":
    evaluate_model_full()
