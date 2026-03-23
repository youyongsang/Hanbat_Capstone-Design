import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loadgen_result.csv")

plt.figure(figsize=(10, 5))

# 평균 latency
plt.plot(df["time_sec"], df["avg_latency_ms"], label="avg_latency")

# p95 latency
plt.plot(df["time_sec"], df["p95_latency_ms"], label="p95_latency")

plt.xlabel("Time (sec)")
plt.ylabel("Latency (ms)")
plt.title("Latency Over Time")
plt.legend()

plt.savefig("latency_graph.png", dpi=300, bbox_inches="tight")
plt.show()
