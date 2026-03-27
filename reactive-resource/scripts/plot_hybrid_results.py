import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


LOADGEN_CSV = Path("data/output/loadgen_result.csv")
HYBRID_CSV = Path("data/output/hybrid_reactive_result.csv")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


def plot_loadgen_count(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    plt.plot(df["time_sec"], df["success_count"], label="success_count")
    plt.plot(df["time_sec"], df["fail_count"], label="fail_count", linestyle="--")
    plt.plot(df["time_sec"], df["target_rps"], label="target_rps", linestyle="--")

    if "sla_violation_count" in df.columns:
        plt.plot(df["time_sec"], df["sla_violation_count"], label="sla_violation_count")

    plt.xlabel("Time (sec)")
    plt.ylabel("Count")
    plt.title("Load Generator - Count")
    plt.legend()
    plt.grid()

    plt.savefig(RESULT_DIR / "hybrid_loadgen_count.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_loadgen_latency(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    plt.plot(df["time_sec"], df["avg_latency_ms"], label="avg_latency_ms")
    plt.plot(df["time_sec"], df["p95_latency_ms"], label="p95_latency_ms")

    plt.xlabel("Time (sec)")
    plt.ylabel("Latency (ms)")
    plt.title("Load Generator - Latency")
    plt.legend()
    plt.grid()

    plt.savefig(RESULT_DIR / "hybrid_loadgen_latency.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_hybrid_resource(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 왼쪽 축: CPU / replicas
    ax1.plot(df["elapsed_sec"], df["current_cpu"], label="current_cpu")
    ax1.plot(df["elapsed_sec"], df["current_replicas"], label="current_replicas")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("CPU / Replicas")

    # 오른쪽 축: cpu usage
    ax2 = ax1.twinx()
    ax2.plot(df["elapsed_sec"], df["cpu_usage_percent"], label="cpu_usage_percent", linestyle="--")
    ax2.set_ylabel("CPU Usage (%)")

    # 액션 시점 표시
    for _, row in df.iterrows():
        action = str(row["action"])
        x = row["elapsed_sec"]

        if action == "CPU_SCALE_OUT":
            ax1.axvline(x=x, linestyle=":", alpha=0.7)
        elif action == "CPU_SCALE_IN":
            ax1.axvline(x=x, linestyle=":", alpha=0.7)
        elif action == "CONTAINER_SCALE_OUT":
            ax1.axvline(x=x, linestyle="--", alpha=0.7)
        elif action == "CONTAINER_SCALE_IN":
            ax1.axvline(x=x, linestyle="--", alpha=0.7)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.title("Hybrid Reactive - Resource Scaling")
    plt.grid()

    plt.savefig(RESULT_DIR / "hybrid_resource_scaling.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_hybrid_performance(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 왼쪽 축: latency
    ax1.plot(df["elapsed_sec"], df["avg_latency_ms"], label="avg_latency_ms")
    ax1.plot(df["elapsed_sec"], df["p95_latency_ms"], label="p95_latency_ms")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Latency (ms)")

    # 오른쪽 축: request_count / fail_count
    ax2 = ax1.twinx()
    ax2.plot(df["elapsed_sec"], df["request_count"], label="request_count", linestyle="--")
    ax2.plot(df["elapsed_sec"], df["fail_count"], label="fail_count", linestyle="--")
    ax2.set_ylabel("Request / Fail Count")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.title("Hybrid Reactive - Performance")
    plt.grid()

    plt.savefig(RESULT_DIR / "hybrid_performance.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    if not LOADGEN_CSV.exists():
        print(f"Loadgen CSV not found: {LOADGEN_CSV}")
        return

    if not HYBRID_CSV.exists():
        print(f"Hybrid CSV not found: {HYBRID_CSV}")
        return

    loadgen_df = pd.read_csv(LOADGEN_CSV)
    hybrid_df = pd.read_csv(HYBRID_CSV)

    plot_loadgen_count(loadgen_df)
    plot_loadgen_latency(loadgen_df)
    plot_hybrid_resource(hybrid_df)
    plot_hybrid_performance(hybrid_df)


if __name__ == "__main__":
    main()