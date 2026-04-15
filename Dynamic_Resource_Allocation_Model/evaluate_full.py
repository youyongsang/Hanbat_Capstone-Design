import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

def evaluate_model_full():
    model = load_model('lstm_model.h5')
    X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')
    with open('scaler_y.pkl', 'rb') as f:
        sy = pickle.load(f)

    y_pred = sy.inverse_transform(model.predict(X_test))
    y_true = sy.inverse_transform(y_test)

    plt.figure(figsize=(15, 5))
    plt.plot(y_true, label='Actual', color='black', alpha=0.6)
    plt.plot(y_pred, label='Predicted (Feedback Applied)', color='red', linestyle='--')
    plt.title('Final Optimized Traffic Prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_model_full()
