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
