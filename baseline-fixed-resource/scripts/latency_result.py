import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/output/result_cpu_1_5.csv")

plt.figure(figsize=(10, 5))

plt.plot(df["time_sec"], df["avg_latency_ms"], label="avg_latency")
plt.plot(df["time_sec"], df["p95_latency_ms"], label="p95_latency")

plt.xlabel("Time (sec)")
plt.ylabel("Latency (ms)")
plt.title("Latency Over Time")
plt.legend()

plt.savefig("../results/latency_graph_cpu_1.5.png", dpi=300, bbox_inches="tight")
plt.show()