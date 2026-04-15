import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from resource_policy import allocate_resource

def run_realtime_simulation(test_rps_data):
    window_size = 20 # 반드시 20으로 변경
    
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

    # [핵심] 바뀐 4가지 피처 로직 적용
    df = pd.DataFrame(test_rps_data, columns=['target_rps'])
    df['instant_diff'] = df['target_rps'].diff().fillna(0)
    df['realtime_std'] = df['target_rps'].rolling(window=2).std().fillna(0)
    df['acceleration'] = df['instant_diff'].diff().fillna(0)
    
    df = df.ffill().bfill()
    
    feature_cols = ['target_rps', 'instant_diff', 'realtime_std', 'acceleration']
    recent_features = df[feature_cols].values[-window_size:]

    # 피처가 4개이므로 shape는 (1, 20, 4)가 됨
    input_scaled = sx.transform(recent_features).reshape(1, window_size, 4)
    pred_scaled = model.predict(input_scaled, verbose=0)
    
    predicted_rps = sy.inverse_transform(pred_scaled)[0][0]

    current_rps = df['target_rps'].iloc[-1]
    cpu, replicas = allocate_resource(predicted_rps, current_rps=current_rps)

    print(f"🔮 [예측] {predicted_rps:.1f} RPS (현재: {current_rps:.1f}) | ⚙️ [할당] CPU: {cpu}, Replicas: {replicas}")
    return predicted_rps, cpu, replicas

if __name__ == "__main__":
    dummy_history = np.random.randint(80, 120, 30).tolist()
    run_realtime_simulation(dummy_history)
