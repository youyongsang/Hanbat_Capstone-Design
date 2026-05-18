from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from config import (
    RESULTS_DIR,
    RL_PLAN_PATH,
    RUNTIME_ALLOC_LOG_CSV,
    RUNTIME_CORRECTION_LOG_CSV,
    RUNTIME_LOADGEN_CSV,
    RUNTIME_PREDICTED_CSV,
)


def save_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def count_active_replicas(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len([part for part in text.split(",") if part.strip()])


def plot_predicted_vs_actual(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(df["time_sec"], df["actual_rps"], label="actual_rps", linewidth=2)
    plt.plot(df["time_sec"], df["predicted_rps"], label="predicted_rps", linewidth=2, linestyle="--")
    plt.xlabel("Time (sec)")
    plt.ylabel("RPS")
    plt.title("RL Predicted Traffic vs Actual Traffic")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(RESULTS_DIR / "rl_predicted_vs_actual_rps.png")


def plot_resource_plan(plan_df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.step(plan_df["time_sec"], plan_df["planned_cpu"], where="post", label="planned_cpu")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("CPU")

    ax2 = ax1.twinx()
    ax2.step(plan_df["time_sec"], plan_df["planned_replicas"], where="post", label="planned_replicas", color="tab:orange")
    ax2.set_ylabel("Replicas")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    plt.title("RL Planned Resource Allocation")
    ax1.grid(alpha=0.3)
    save_figure(RESULTS_DIR / "rl_resource_plan.png")


def plot_plan_vs_actual_resource(plan_df: pd.DataFrame, loadgen_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    actual_replica_df = loadgen_df.copy()
    actual_replica_df["actual_replicas"] = actual_replica_df["active_ports"].apply(count_active_replicas)
    plt.step(plan_df["time_sec"], plan_df["planned_replicas"], where="post", label="planned_replicas")
    plt.step(actual_replica_df["time_sec"], actual_replica_df["actual_replicas"], where="post", label="actual_replicas", linestyle="--")
    plt.xlabel("Time (sec)")
    plt.ylabel("Replicas")
    plt.title("RL Planned vs Actual Replicas")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(RESULTS_DIR / "rl_plan_vs_actual_replicas.png")


def plot_loadgen_performance(loadgen_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(loadgen_df["time_sec"], loadgen_df["target_rps"], label="target_rps")
    axes[0].plot(loadgen_df["time_sec"], loadgen_df["success_count"], label="success_count")
    axes[0].plot(loadgen_df["time_sec"], loadgen_df["fail_count"], label="fail_count", linestyle="--")
    if "sla_violation_count" in loadgen_df.columns:
        axes[0].plot(loadgen_df["time_sec"], loadgen_df["sla_violation_count"], label="sla_violation_count", linestyle=":")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RL Load Generator Throughput")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(loadgen_df["time_sec"], loadgen_df["avg_latency_ms"], label="avg_latency_ms")
    axes[1].plot(loadgen_df["time_sec"], loadgen_df["p95_latency_ms"], label="p95_latency_ms")
    axes[1].set_xlabel("Time (sec)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("RL Load Generator Latency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    save_figure(RESULTS_DIR / "rl_loadgen_performance.png")


def plot_correction_actions(correction_df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(correction_df["elapsed_sec"], correction_df["avg_latency_ms"], label="avg_latency_ms")
    ax1.plot(correction_df["elapsed_sec"], correction_df["p95_latency_ms"], label="p95_latency_ms")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Latency (ms)")

    ax2 = ax1.twinx()
    ax2.plot(correction_df["elapsed_sec"], correction_df["cpu_usage_pct"], label="cpu_usage_pct", linestyle="--")
    ax2.plot(correction_df["elapsed_sec"], correction_df["fail_count"], label="fail_count", linestyle=":")
    ax2.set_ylabel("CPU Usage / Fail Count")

    for _, row in correction_df.iterrows():
        action = str(row["action"])
        if action != "HOLD":
            ax1.axvline(x=row["elapsed_sec"], color="gray", linestyle=":", alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    plt.title("RL Reactive Corrections and Performance")
    ax1.grid(alpha=0.3)
    save_figure(RESULTS_DIR / "rl_correction_actions.png")


def main():
    required = [RUNTIME_PREDICTED_CSV, RL_PLAN_PATH, RUNTIME_CORRECTION_LOG_CSV, RUNTIME_LOADGEN_CSV]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("다음 입력 파일이 없어 RL 결과 그래프를 생성할 수 없습니다.")
        for path in missing:
            print(f"  - {path}")
        return

    predicted_df = pd.read_csv(RUNTIME_PREDICTED_CSV)
    plan_df = pd.read_csv(RL_PLAN_PATH)
    correction_df = pd.read_csv(RUNTIME_CORRECTION_LOG_CSV)
    loadgen_df = pd.read_csv(RUNTIME_LOADGEN_CSV)

    plot_predicted_vs_actual(predicted_df)
    plot_resource_plan(plan_df)
    plot_plan_vs_actual_resource(plan_df, loadgen_df)
    plot_loadgen_performance(loadgen_df)
    plot_correction_actions(correction_df)
    print(f"그래프 저장 완료: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
