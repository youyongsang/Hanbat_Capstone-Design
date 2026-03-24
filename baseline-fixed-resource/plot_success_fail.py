import pandas as pd
import matplotlib.pyplot as plt

# CSV 읽기
df = pd.read_csv("result_cpu_1_5.csv")

plt.figure()

# 기존
plt.plot(df["time_sec"], df["success_count"], label="success_count")
plt.plot(df["time_sec"], df["fail_count"], label="fail_count")

# 추가 (RPS)
plt.plot(df["time_sec"], df["target_rps"], label="target_rps")

# 설정
plt.xlabel("Time (sec)")
plt.ylabel("Count")
plt.title("RPS vs Success/Fail Over Time")
plt.legend()

plt.savefig("success_fail_graph_cpu_1.5.png", dpi=300, bbox_inches="tight")
plt.show()
