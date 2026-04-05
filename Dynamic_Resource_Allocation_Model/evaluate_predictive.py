import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate():
    # plots 폴더가 없으면 자동 생성 코드 추가
    os.makedirs('plots', exist_ok=True)
    
    if not os.path.exists('X_test.npy'):
        raise FileNotFoundError("❌ 테스트 데이터가 없습니다. 전처리를 먼저 수행하세요.")

    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    model = load_model('lstm_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 모델 예측
    pred_scaled = model.predict(X_test, verbose=0)
    
    # 역정규화 (실제 RPS 값으로 복원)
    y_true = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(pred_scaled)

    # 객관적 평가지표 계산
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n📊 [Test 데이터(미학습 구간) 평가 결과]")
    print(f" - MAE  (평균 절대 오차): {mae:.2f}")
    print(f" - RMSE (루트 평균 제곱 오차): {rmse:.2f}")

    # 실제 vs 예측 시각화
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Actual Traffic (Test Set)', color='black', alpha=0.7)
    plt.plot(y_pred, label='Predicted Traffic (LSTM)', color='blue', linestyle='--')
    plt.title('Test Set Target RPS vs Predicted RPS')
    plt.xlabel('Time Step')
    plt.ylabel('RPS')
    plt.legend()
    plt.grid(True)
    
    plot_path = 'plots/test_evaluation_result.png'
    plt.savefig(plot_path)
    print(f"✅ 평가 그래프 저장 완료: {plot_path}\n")

if __name__ == "__main__":
    evaluate()
