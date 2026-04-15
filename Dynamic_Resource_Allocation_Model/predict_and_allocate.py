import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from resource_policy import allocate_resource

def run_realtime_simulation(test_rps_data):
    window_size = 60 # 60으로 변경
    
    try:
        model = load_model('lstm_model.h5')
        with open('scaler_x.pkl', 'rb') as f: sx = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f: sy = pickle.load(f)
    except Exception as e:
        print(f"❌ 로드 에러: {e}")
        return

    if len(test_rps_data) < window_size:
        return

    # 피처 생성 로직 동기화
    df = pd.DataFrame(test_rps_data, columns=['target_rps'])
    df['diff'] = df['target_rps'].diff().fillna(0)
    df['rolling_mean'] = df['target_rps'].rolling(window=10).mean().bfill()
    
    recent_features = df[['target_rps', 'diff', 'rolling_mean']].values[-window_size:]
    input_scaled = sx.transform(recent_features).reshape(1, window_size, 3)
    
    pred_scaled = model.predict(input_scaled, verbose=0)
    predicted_rps = sy.inverse_transform(pred_scaled)[0][0]

    current_rps = df['target_rps'].iloc[-1]
    cpu, replicas = allocate_resource(predicted_rps, current_rps=current_rps)
    
    print(f"🔮 예측: {predicted_rps:.1f} RPS | 할당: CPU {cpu}, Rep {replicas}")
    return predicted_rps, cpu, replicas

if __name__ == "__main__":
    dummy = np.random.randint(80, 120, 70).tolist()
    run_realtime_simulation(dummy)
