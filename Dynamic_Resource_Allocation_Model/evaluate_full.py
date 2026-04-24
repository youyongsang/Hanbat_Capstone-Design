# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

def evaluate_model_full():
    print("📊 결과 시각화 시작...")
    try:
        model = load_model('lstm_model.h5')
        X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
        X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')
        with open('scaler_y.pkl', 'rb') as f: scaler_y = pickle.load(f)
    except:
        print("❌ 필요한 파일이 없습니다."); return

    y_tr_pred = model.predict(X_train, verbose=0)
    y_te_pred = model.predict(X_test, verbose=0)
    
    y_tr_true = scaler_y.inverse_transform(y_train)
    y_te_true = scaler_y.inverse_transform(y_test)
    y_tr_pred = scaler_y.inverse_transform(y_tr_pred)
    y_te_pred = scaler_y.inverse_transform(y_te_pred)

    y_true = np.concatenate([y_tr_true, y_te_true])
    y_pred = np.concatenate([y_tr_pred, y_te_pred])

    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual Traffic', color='black', alpha=0.6)
    plt.plot(y_pred, label='LSTM Predicted', color='red', linestyle='--')
    plt.axvline(x=len(y_tr_true), color='blue', linestyle=':', label='Test Boundary')
    plt.title('Traffic Prediction: Full Data Tracking')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('full_evaluation.png')
    print("✅ 그래프 저장 완료 (full_evaluation.png)")
    plt.show()

if __name__ == "__main__":
    evaluate_model_full()
