import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/output/loadgen_result.csv")

plt.figure(figsize=(10, 6))

plt.plot(df["time_sec"], df["success_count"], label="success_count")
plt.plot(df["time_sec"], df["fail_count"], label="fail_count")
plt.plot(df["time_sec"], df["target_rps"], label="target_rps", linestyle="--")

plt.xlabel("Time (sec)")
plt.ylabel("Count")
plt.title("Load Generator - Count")
plt.legend()
plt.grid()

plt.savefig("count_graph_no_reactive.png", dpi=300, bbox_inches="tight")
plt.show()