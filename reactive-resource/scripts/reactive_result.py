import pandas as pd
import matplotlib.pyplot as plt

# CSV 읽기
df = pd.read_csv("data/output/reactive_result.csv")

plt.figure(figsize=(10, 6))

# CPU 변화
plt.plot(df["elapsed_sec"], df["current_cpu"], label="CPU", linewidth=2)

# latency
plt.plot(df["elapsed_sec"], df["avg_latency_ms"], label="avg_latency_ms")
plt.plot(df["elapsed_sec"], df["p95_latency_ms"], label="p95_latency_ms")

# 요청 수
plt.plot(df["elapsed_sec"], df["request_count"], label="request_count", linestyle="--")

plt.xlabel("Time (sec)")
plt.ylabel("Value")
plt.title("Reactive Controller Result")
plt.legend()

plt.grid()
plt.savefig("reactive_graph.png", dpi=300, bbox_inches="tight")
plt.show()