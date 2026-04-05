# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(42)
time = np.linspace(0, 100, 50000)
raw_rps = 500 + 1500 * (np.sin(time) * 0.5 + 0.5) + np.random.normal(0, 50, 50000)
raw_rps = raw_rps.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_rps = scaler.fit_transform(raw_rps)

window_size = 12
X, y = [], []
for i in range(len(scaled_rps) - window_size):
    X.append(scaled_rps[i:i+window_size])
    y.append(scaled_rps[i+window_size])
X, y = np.array(X), np.array(y)

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

model = Sequential([
    LSTM(64, input_shape=(window_size, 1)),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("Starting Production-Level LSTM Training (Epochs=20)...")
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

predicted_scaled = model.predict(X_test)
predicted_rps = scaler.inverse_transform(predicted_scaled).flatten()
actual_rps = scaler.inverse_transform(y_test).flatten()

mae = mean_absolute_error(actual_rps, predicted_rps)
rmse = np.sqrt(mean_squared_error(actual_rps, predicted_rps))

print("\n=== Model Evaluation Metrics ===")
print(f"MAE: {mae:.2f} RPS")
print(f"RMSE: {rmse:.2f} RPS")
print("================================\n")

allocated_resources = []
current_containers = 25
smoothed_capacity = current_containers * 80

for p in predicted_rps:
    safe_pred = p * 1.1
    target_containers = max(5, int(np.ceil(safe_pred / 80)))
    
    if target_containers > current_containers:
        current_containers = target_containers
    elif target_containers < current_containers - 1:
        current_containers = target_containers
        
    current_containers = min(current_containers, 35)
    target_capacity = current_containers * 80
    smoothed_capacity = 0.7 * smoothed_capacity + 0.3 * target_capacity
    allocated_resources.append(smoothed_capacity)

plt.figure(figsize=(15, 6))
plot_range = 1000 
plt.plot(actual_rps[:plot_range], label='Actual Traffic', color='lightgray', alpha=0.7)
plt.plot(predicted_rps[:plot_range], label=f'LSTM Predicted (MAE: {mae:.1f})', color='blue', linestyle='--', linewidth=1.5)
plt.plot(allocated_resources[:plot_range], label='Smart Allocation (Production-Logic)', color='red', linewidth=2)

plt.title('Final Production-Ready Auto-Scaling Model')
plt.legend(loc='upper right')

if not os.path.exists('plots'): os.makedirs('plots')
plt.savefig('plots/real_lstm_production_final.png')
print("--- [SUCCESS] Production-Ready Model Created ---")
