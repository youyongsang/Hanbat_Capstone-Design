import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from resource_policy import allocate_resource

def run_realtime_simulation(test_rps_data):
    window_size = 10 # 10으로 변경
    
    try:
        model = load_model('lstm_model.h5')
        with open('scaler_x.pkl', 'rb') as f: sx = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f: sy = pickle.load(f)
    except Exception as e:
        print(f"❌ 파일 로드 에러: {e}")
        return

    if len(test_rps_data) < window_size:
        print(f"⚠️ 데이터 부족: {len(test_rps_data)}/{window_size}")
        return

    # [핵심 수정] 바뀐 피처 로직 적용 (원본 데이터만)
    df = pd.DataFrame(test_rps_data, columns=['target_rps'])
    
    feature_cols = ['target_rps']
    recent_features = df[feature_cols].values[-window_size:]

    # 피처가 1개이므로 shape는 (1, 10, 1)가 됨
    input_scaled = sx.transform(recent_features).reshape(1, window_size, 1)
    pred_scaled = model.predict(input_scaled, verbose=0)
    
    predicted_rps = sy.inverse_transform(pred_scaled)[0][0]

    current_rps = df['target_rps'].iloc[-1]
    cpu, replicas = allocate_resource(predicted_rps, current_rps=current_rps)

    print(f"🔮 [예측] {predicted_rps:.1f} RPS (현재: {current_rps:.1f}) | ⚙️ [할당] CPU: {cpu}, Replicas: {replicas}")
    return predicted_rps, cpu, replicas

if __name__ == "__main__":
    dummy_history = np.random.randint(80, 120, 15).tolist()
    run_realtime_simulation(dummy_history)
