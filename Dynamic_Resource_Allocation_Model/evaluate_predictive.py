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

predict_and_allocate.py

import numpy as np
import pandas as pd
import pickle
import time
from tensorflow.keras.models import load_model
from resource_policy import allocate_resource # 사용자님의 기존 정책 파일

def run_realtime_simulation(test_rps_data):
    """
    test_rps_data: 실시간으로 들어오는 RPS 리스트 (최소 60개 이상)
    """
    window_size = 60
    
    # 1. 모델 및 스케일러 로드
    try:
        model = load_model('lstm_model.h5')
        with open('scaler_x.pkl', 'rb') as f: sx = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f: sy = pickle.load(f)
    except Exception as e:
        print(f"❌ 로드 에러: {e}")
        return

    print("🚀 실시간 오토스케일링 시뮬레이션 시작...")
    
    # 최근 데이터 60개가 확보되었을 때부터 예측 시작
    if len(test_rps_data) < window_size:
        print(f"⚠️ 데이터 부족: {len(test_rps_data)}/{window_size}")
        return

    # 2. [중요] 실시간 데이터를 학습 규격(3개 Feature)으로 가공
    df = pd.DataFrame(test_rps_data, columns=['target_rps'])
    df['moving_avg'] = df['target_rps'].rolling(window=10).mean()
    df['diff'] = df['target_rps'].diff().fillna(0).rolling(window=3).mean()
    
    # 동적 클리핑 (학습 때와 동일한 로직 권장이나 여기선 간단히 처리)
    df = df.ffill().bfill()
    
    # 최신 윈도우 추출
    recent_features = df[['target_rps', 'moving_avg', 'diff']].values[-window_size:]

    # 3. 스케일링 및 예측
    input_scaled = sx.transform(recent_features).reshape(1, window_size, 3)
    pred_scaled = model.predict(input_scaled, verbose=0)
    
    # 4. RPS 복원
    predicted_rps = sy.inverse_transform(pred_scaled)[0][0]

    # 5. [상호작용] 기존 자원 할당 정책 호출
    cpu, replicas = allocate_resource(predicted_rps)

    print(f"🔮 [예측] {predicted_rps:.1f} RPS | ⚙️ [할당] CPU: {cpu}, Replicas: {replicas}")
    return predicted_rps, cpu, replicas

if __name__ == "__main__":
    # 시뮬레이션용 가짜 데이터 (90개)
    dummy_history = np.random.randint(80, 120, 90).tolist()
    run_realtime_simulation(dummy_history)
