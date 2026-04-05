import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
df = pd.read_csv('sale_event_traffic.csv') 
raw_rps = df['target_rps'].values 
alpha = 0.1 
ema_rps = [raw_rps[0]] 
for i in range(1, len(raw_rps)): 
    ema_rps.append(alpha * raw_rps[i] + (1 - alpha) * ema_rps[-1]) 
ema_rps = np.array(ema_rps) 
allocated_servers = [] 
current_servers = 2 
cooldown_timer = 0 
COOLDOWN_PERIOD = 50 
for rps in ema_rps: 
    safe_rps = rps * 1.2 
    target_servers = 2 + int(max(0, safe_rps) // 100) 
    if target_servers > current_servers: 
        current_servers = target_servers 
        cooldown_timer = COOLDOWN_PERIOD 
    elif target_servers < current_servers: 
        if cooldown_timer <= 0: 
            current_servers = target_servers 
            cooldown_timer = COOLDOWN_PERIOD 
        else: 
            cooldown_timer -= 1 
    else: 
        if cooldown_timer > 0: 
            cooldown_timer -= 1 
    allocated_servers.append(current_servers) 
plt.figure(figsize=(15, 6)) 
plt.plot(raw_rps, label='Raw', color='lightgray', alpha=0.5) 
plt.plot(ema_rps, label='EMA', color='blue', alpha=0.8) 
plt.step(range(len(allocated_servers)), [s*50 for s in allocated_servers], label='Res', color='red') 
plt.title('Final Advanced Scaling') 
plt.legend() 
if not os.path.exists('plots'): os.makedirs('plots') 
plt.savefig('plots/final_advanced_result.png') 
print('--- Simulation Success! ---') 
