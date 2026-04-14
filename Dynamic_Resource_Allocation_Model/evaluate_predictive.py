import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model():
    print("📊 모델 평가 및 시각화 시작...")
    
    # 1. 모델 및 데이터 로드
    try:
        model = load_model('lstm_model.h5')
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        
        # [수정] 출력용 스케일러 로드
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
    except FileNotFoundError as e:
        print(f"❌ 필요한 파일이 없습니다: {e}")
        return

    # 2. 예측 수행
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # 3. 역정규화 (StandardScaler -> 실제 RPS)
    y_true = scaler_y.inverse_transform(y_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 4. 성능 평가
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n✅ 평가 결과")
    print(f" - MAE  (평균 오차): {mae:.2f} RPS")
    print(f" - RMSE (오차 제곱근): {rmse:.2f} RPS")

    # 5. 시각화 (처음 500개 샘플)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[:500], label='Actual Traffic', color='black', alpha=0.7)
    plt.plot(y_pred[:500], label='LSTM Predicted', color='red', linestyle='--')
    plt.title('RPS Prediction: Actual vs Improved LSTM (Huber Loss)')
    plt.xlabel('Time (sec)')
    plt.ylabel('RPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    evaluate_model()
