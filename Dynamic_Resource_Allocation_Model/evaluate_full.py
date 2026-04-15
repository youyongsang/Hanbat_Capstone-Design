import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

def evaluate_model_full():
    print("📊 전체 트래픽 예측 시각화 중...")
    
    model = load_model('lstm_model.h5')
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_val, y_val = np.load('X_val.npy'), np.load('y_val.npy')
    X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')
    
    X_all = np.concatenate((X_train, X_val, X_test))
    y_all = np.concatenate((y_train, y_val, y_test))
    
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    y_pred_scaled = model.predict(X_all, verbose=0)
    
    y_true = scaler_y.inverse_transform(y_all)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual Traffic', color='black', alpha=0.7)
    plt.plot(y_pred, label='LSTM Predicted (Hardcore)', color='red', linestyle='--')
    
    plt.axvline(x=len(X_train)+len(X_val), color='blue', linestyle=':', label='Test Start')
    
    plt.title('Full Traffic Prediction - Hardcore Peak Tracking')
    plt.xlabel('Time (sec)')
    plt.ylabel('RPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    evaluate_model_full()
