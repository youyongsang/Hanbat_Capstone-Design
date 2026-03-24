import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 기준 경로 (프로젝트 루트)
BASE_DIR = Path(__file__).resolve().parent.parent

# 파일 경로
CSV_PATH = BASE_DIR / "data/output/result_cpu_1_5.csv"
SAVE_PATH = BASE_DIR / "results/success_fail_graph_cpu_1.5.png"

# CSV 읽기
df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(10, 5))

# 성공/실패
plt.plot(df["time_sec"], df["success_count"], label="success_count")
plt.plot(df["time_sec"], df["fail_count"], label="fail_count")

# RPS
plt.plot(df["time_sec"], df["target_rps"], label="target_rps")

# 설정
plt.xlabel("Time (sec)")
plt.ylabel("Count")
plt.title("RPS vs Success/Fail Over Time")
plt.legend()

# 결과 폴더 없으면 생성
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()