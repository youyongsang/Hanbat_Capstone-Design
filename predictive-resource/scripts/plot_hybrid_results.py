from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "output"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

PREDICTED_CSV = OUTPUT_DIR / "predicted_traffic.csv"
PLAN_CSV = OUTPUT_DIR / "resource_allocation_plan.csv"
ALLOC_LOG_CSV = OUTPUT_DIR / "predictive_allocation_log.csv"
CORRECTION_CSV = OUTPUT_DIR / "hybrid_correction_log.csv"
LOADGEN_CSV = OUTPUT_DIR / "loadgen_result.csv"


def save_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(df["time_sec"], df["actual_rps"], label="actual_rps", linewidth=2)
    plt.plot(df["time_sec"], df["predicted_rps"], label="predicted_rps", linewidth=2, linestyle="--")
    plt.xlabel("Time (sec)")
    plt.ylabel("RPS")
    plt.title("Predicted Traffic vs Actual Traffic")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(RESULT_DIR / "predicted_vs_actual_rps.png")


def plot_prediction_error(df: pd.DataFrame):
    error = df["predicted_rps"] - df["actual_rps"]
    abs_error = error.abs()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["time_sec"], error, label="prediction_error", color="tab:red")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Prediction Error")

    ax2 = ax1.twinx()
    ax2.plot(df["time_sec"], abs_error, label="absolute_error", color="tab:blue", linestyle=":")
    ax2.set_ylabel("Absolute Error")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    plt.title("Prediction Error Over Time")
    ax1.grid(alpha=0.3)
    save_figure(RESULT_DIR / "prediction_error.png")


def plot_resource_plan(plan_df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.step(plan_df["time_sec"], plan_df["planned_cpu"], where="post", label="planned_cpu")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("CPU")

    ax2 = ax1.twinx()
    ax2.step(
        plan_df["time_sec"],
        plan_df["planned_replicas"],
        where="post",
        label="planned_replicas",
        color="tab:orange",
    )
    ax2.set_ylabel("Replicas")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    plt.title("Planned Resource Allocation")
    ax1.grid(alpha=0.3)
    save_figure(RESULT_DIR / "resource_plan.png")


def plot_plan_vs_actual_resource(correction_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].step(correction_df["elapsed_sec"], correction_df["pred_cpu"], where="post", label="planned_cpu")
    axes[0].step(correction_df["elapsed_sec"], correction_df["curr_cpu"], where="post", label="actual_cpu")
    axes[0].set_ylabel("CPU")
    axes[0].set_title("Planned vs Actual CPU")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].step(
        correction_df["elapsed_sec"],
        correction_df["pred_replicas"],
        where="post",
        label="planned_replicas",
    )
    axes[1].step(
        correction_df["elapsed_sec"],
        correction_df["curr_replicas"],
        where="post",
        label="actual_replicas",
    )
    axes[1].set_xlabel("Time (sec)")
    axes[1].set_ylabel("Replicas")
    axes[1].set_title("Planned vs Actual Replicas")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    save_figure(RESULT_DIR / "plan_vs_actual_resource.png")


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
    plt.title("Reactive Corrections and Performance")
    ax1.grid(alpha=0.3)
    save_figure(RESULT_DIR / "correction_actions.png")


def plot_loadgen_performance(loadgen_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(loadgen_df["time_sec"], loadgen_df["target_rps"], label="target_rps")
    axes[0].plot(loadgen_df["time_sec"], loadgen_df["success_count"], label="success_count")
    axes[0].plot(loadgen_df["time_sec"], loadgen_df["fail_count"], label="fail_count", linestyle="--")
    if "sla_violation_count" in loadgen_df.columns:
        axes[0].plot(
            loadgen_df["time_sec"],
            loadgen_df["sla_violation_count"],
            label="sla_violation_count",
            linestyle=":",
        )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Load Generator Throughput")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(loadgen_df["time_sec"], loadgen_df["avg_latency_ms"], label="avg_latency_ms")
    axes[1].plot(loadgen_df["time_sec"], loadgen_df["p95_latency_ms"], label="p95_latency_ms")
    axes[1].set_xlabel("Time (sec)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Load Generator Latency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    save_figure(RESULT_DIR / "loadgen_performance.png")


def print_prediction_metrics(df: pd.DataFrame):
    model_df = df[df["source"] == "model_forecast"].copy()
    if model_df.empty:
        print("예측 구간이 없어 예측 정확도 요약을 건너뜁니다.")
        return

    abs_error = (model_df["predicted_rps"] - model_df["actual_rps"]).abs()
    mae = abs_error.mean()
    rmse = ((model_df["predicted_rps"] - model_df["actual_rps"]) ** 2).mean() ** 0.5

    non_zero = model_df["actual_rps"] != 0
    if non_zero.any():
        mape = (
            ((abs_error[non_zero] / model_df.loc[non_zero, "actual_rps"]).mean()) * 100.0
        )
    else:
        mape = float("nan")

    print("예측 정확도 요약")
    print(f"  forecast rows : {len(model_df)}")
    print(f"  MAE           : {mae:.3f}")
    print(f"  RMSE          : {rmse:.3f}")
    if pd.notna(mape):
        print(f"  MAPE          : {mape:.3f}%")
    else:
        print("  MAPE          : actual_rps가 0뿐이라 계산 불가")


def main():
    missing = [
        path for path in [PREDICTED_CSV, PLAN_CSV, CORRECTION_CSV, LOADGEN_CSV]
        if not path.exists()
    ]
    if missing:
        print("다음 입력 파일이 없어 그래프를 생성할 수 없습니다.")
        for path in missing:
            print(f"  - {path}")
        return

    predicted_df = pd.read_csv(PREDICTED_CSV)
    plan_df = pd.read_csv(PLAN_CSV)
    correction_df = pd.read_csv(CORRECTION_CSV)
    loadgen_df = pd.read_csv(LOADGEN_CSV)

    plot_predicted_vs_actual(predicted_df)
    plot_prediction_error(predicted_df)
    plot_resource_plan(plan_df)
    plot_plan_vs_actual_resource(correction_df)
    plot_correction_actions(correction_df)
    plot_loadgen_performance(loadgen_df)
    print_prediction_metrics(predicted_df)

    if ALLOC_LOG_CSV.exists():
        print(f"계획 실행 로그 확인 가능: {ALLOC_LOG_CSV}")

    print(f"그래프 저장 완료: {RESULT_DIR}")


if __name__ == "__main__":
    main()
