import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load Data
try:
    df = pd.read_csv('sale_event_traffic.csv')
    raw_rps = df['target_rps'].values
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Strong EMA (alpha=0.1)
alpha = 0.1
ema_rps = [raw_rps[0]]
for i in range(1, len(raw_rps)):
    ema_rps.append(alpha * raw_rps[i] + (1 - alpha) * ema_rps[-1])
ema_rps = np.array(ema_rps)

# 3. Scaling Settings
allocated_servers = []
current_servers = 2
cooldown_timer = 0
COOLDOWN_PERIOD = 80
MAX_SERVERS = 10

for i in range(len(ema_rps)):
    # 4. Adaptive Margin based on Increase Rate
    if i > 0:
        increase_rate = (ema_rps[i] - ema_rps[i-1]) / max(1, ema_rps[i-1])
        if increase_rate > 0.1: margin = 1.4
        elif increase_rate > 0.03: margin = 1.3
        else: margin = 1.15
    else:
        margin = 1.15

    safe_rps = ema_rps[i] * margin
    target_servers = 2 + int(max(0, safe_rps) // 80)
    target_servers = min(target_servers, MAX_SERVERS)

    # 5. Scaling Logic with Scale-down relaxation
    if cooldown_timer <= 0:
        if target_servers >= current_servers + 2: # Scale-UP
            current_servers = target_servers
            cooldown_timer = COOLDOWN_PERIOD
        elif target_servers < current_servers - 1: # Scale-DOWN (Relaxed)
            current_servers = target_servers
            cooldown_timer = COOLDOWN_PERIOD
    else:
        cooldown_timer -= 1
    allocated_servers.append(current_servers)

# 6. Visualization & Save
plt.figure(figsize=(15, 6))
plt.plot(raw_rps, label='Raw Traffic', color='lightgray', alpha=0.5)
plt.plot(ema_rps, label='EMA (alpha=0.1)', color='blue', linewidth=2)
plt.step(range(len(allocated_servers)), [s*80 for s in allocated_servers], label='Intelligent Res', color='red', linewidth=2, where='post')
plt.axhline(y=MAX_SERVERS*80, color='orange', linestyle='--', label='Max Capacity')
plt.title('Ultimate Auto-Scaling: Intelligent Feedback Applied')
plt.legend()

if not os.path.exists('plots'):
    os.makedirs('plots')

plt.savefig('plots/ultimate_final_result.png')
print("--- Success: ultimate_final_result.png Created! ---")