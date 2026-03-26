import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/output/loadgen_result.csv")

plt.figure(figsize=(10, 6))

plt.plot(df["time_sec"], df["avg_latency_ms"], label="avg_latency_ms")
plt.plot(df["time_sec"], df["p95_latency_ms"], label="p95_latency_ms")

plt.xlabel("Time (sec)")
plt.ylabel("Latency (ms)")
plt.title("Load Generator - Latency")
plt.legend()
plt.grid()

plt.savefig("latency_graph_no_reactive.png", dpi=300, bbox_inches="tight")
plt.show()