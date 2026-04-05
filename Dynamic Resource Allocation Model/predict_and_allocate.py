import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from resource_policy import allocate_resource

def run_simulation(recent_rps_sequence):
    window_size = 12
    
    # 예외 처리: window_size 이하 데이터 유입 방어
    if len(recent_rps_sequence) < window_size:
        print(f"⚠️ 경고: 예측을 위해 최소 {window_size}개의 트래픽 데이터가 필요합니다. (현재: {len(recent_rps_sequence)}개)")
        return None, None, None

    # 최신 window_size 만큼만 슬라이싱
    input_seq = recent_rps_sequence[-window_size:]

    if not os.path.exists('lstm_model.h5') or not os.path.exists('scaler.pkl'):
        raise FileNotFoundError("❌ 모델이나 스케일러가 없습니다.")

    model = load_model('lstm_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 차원 맞추기 및 정규화
    input_data = np.array(input_seq).reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    model_input = scaled_input.reshape(1, window_size, 1)

    pred_scaled = model.predict(model_input, verbose=0)
    pred_rps = scaler.inverse_transform(pred_scaled)[0][0]

    cpu, replicas = allocate_resource(pred_rps)

    print("-" * 45)
    print(f"📊 입력된 과거 {window_size}초 RPS: {input_seq}")
    print(f"🔮 예측된 다음 1초 RPS: {pred_rps:.2f}")
    print(f"⚙️  선제적 할당 정책: CPU {cpu} / 컨테이너 {replicas}개")
    print("-" * 45)
    
    return pred_rps, cpu, replicas

if __name__ == "__main__":
    test_sequence = [90, 95, 100, 110, 130, 150, 180, 220, 270, 320, 380, 410]
    run_simulation(test_sequence)
