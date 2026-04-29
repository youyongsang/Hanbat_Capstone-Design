import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from schema_adapter import SmartCSVLoader, STANDARD_FEATURES  # noqa: E402


MODEL_PATH = MODEL_DIR / "lstm_model.h5"
SCALER_X_PATH = MODEL_DIR / "scaler_x.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"
METADATA_PATH = MODEL_DIR / "metadata.pkl"

DEFAULT_INPUT_CSV = BASE_DIR / "data" / "input" / "sale_event_traffic.csv"
DEFAULT_FORECAST_CSV = BASE_DIR / "data" / "output" / "predicted_traffic.csv"
DEFAULT_PLAN_CSV = BASE_DIR / "data" / "output" / "resource_allocation_plan.csv"
DEFAULT_PLAN_PLOT = BASE_DIR / "results" / "resource_allocation_plan_overview.png"

SAFETY_MARGIN = 1.60
MIN_REPLICAS = 3
MAX_REPLICAS = 8
MIN_CPU = 2.0
MAX_CPU = 6.0


def load_model_artifacts():
    try:
        from tensorflow.keras.models import load_model  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "tensorflow.keras를 불러오지 못했습니다. 먼저 TensorFlow를 설치하세요."
        ) from exc

    model = load_model(MODEL_PATH, compile=False)
    with open(SCALER_X_PATH, "rb") as f:
        scaler_x = pickle.load(f)
    with open(SCALER_Y_PATH, "rb") as f:
        scaler_y = pickle.load(f)

    feature_cols = STANDARD_FEATURES
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH, "rb") as f:
            feature_cols = pickle.load(f)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

    window_size = int(metadata.get("window_size", 60))
    return model, scaler_x, scaler_y, feature_cols, window_size


def compute_allocation(pred_rps: float) -> tuple[float, int]:
    safe_rps = pred_rps * SAFETY_MARGIN
    cpu_capacity_policy = [
        (2.0, 70),
        (3.0, 105),
        (4.0, 140),
        (5.0, 175),
        (6.0, 210),
    ]

    # CPU를 먼저 충분히 올리고, 그래도 감당이 안 될 때만 replica를 늘린다.
    # 즉 "최소 replica 수"를 우선으로 찾고, 그 replica 수 안에서 가장 낮은 CPU를 고른다.
    for replicas in range(MIN_REPLICAS, MAX_REPLICAS + 1):
        for cpu, per_replica_capacity in cpu_capacity_policy:
            total_capacity = replicas * per_replica_capacity
            if safe_rps <= total_capacity:
                return min(MAX_CPU, max(MIN_CPU, cpu)), replicas

    return MAX_CPU, MAX_REPLICAS


def load_schedule(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    required = {"time_sec", "target_rps"}
    if not required.issubset(raw_df.columns):
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {required}")

    raw_df = raw_df.reset_index(drop=True).copy()
    feature_df = SmartCSVLoader(str(csv_path), verbose=False).load().reset_index(drop=True)

    schedule_df = feature_df.copy()
    schedule_df["time_sec"] = raw_df["time_sec"].astype(int).values
    schedule_df["actual_rps"] = raw_df["target_rps"].astype(float).values

    for col in ["phase", "scenario"]:
        if col in raw_df.columns:
            schedule_df[col] = raw_df[col].values
        else:
            schedule_df[col] = ""

    return schedule_df


def one_step_forecast_from_actual_history(df: pd.DataFrame, observed_points: int) -> pd.DataFrame:
    model, scaler_x, scaler_y, feature_cols, window_size = load_model_artifacts()

    if observed_points < window_size:
        raise ValueError(
            f"observed_points는 최소 {window_size} 이상이어야 합니다. 현재 값: {observed_points}"
        )
    if observed_points >= len(df):
        raise ValueError(
            "observed_points가 전체 길이 이상입니다. 미래 구간이 남도록 더 작은 값을 사용하세요."
        )

    feature_frame = df[feature_cols].copy().reset_index(drop=True)
    rows = []

    for _, row in df.iloc[:observed_points].iterrows():
        rows.append({
            "time_sec": int(row["time_sec"]),
            "actual_rps": float(row["actual_rps"]),
            "predicted_rps": float(row["actual_rps"]),
            "phase": row["phase"] if "phase" in df.columns else "",
            "scenario": row["scenario"] if "scenario" in df.columns else "",
            "source": "observed_history",
        })

    for idx in range(observed_points, len(df)):
        window_df = feature_frame.iloc[idx - window_size : idx]
        model_input = scaler_x.transform(window_df[feature_cols].values, check_ood=False)
        model_input = model_input.reshape(1, window_size, len(feature_cols))

        pred_scaled = model.predict(model_input, verbose=0)
        pred_rps = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0])
        pred_rps = max(0.0, pred_rps)

        row = df.iloc[idx]
        rows.append({
            "time_sec": int(row["time_sec"]),
            "actual_rps": float(row["actual_rps"]),
            "predicted_rps": pred_rps,
            "phase": row["phase"] if "phase" in df.columns else "",
            "scenario": row["scenario"] if "scenario" in df.columns else "",
            "source": "model_forecast",
        })

    return pd.DataFrame(rows)


def build_plan(forecast_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in forecast_df.iterrows():
        cpu, replicas = compute_allocation(float(row["predicted_rps"]))
        rows.append({
            "time_sec": int(row["time_sec"]),
            "predicted_rps": round(float(row["predicted_rps"]), 3),
            "planned_cpu": cpu,
            "planned_replicas": replicas,
            "phase": row.get("phase", ""),
            "scenario": row.get("scenario", ""),
            "source": row.get("source", ""),
        })
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def plot_plan(forecast_df: pd.DataFrame, plan_df: pd.DataFrame, output_path: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(
        forecast_df["time_sec"],
        forecast_df["actual_rps"],
        label="actual_rps",
        color="#B0BEC5",
        linewidth=1.5,
    )
    axes[0].plot(
        forecast_df["time_sec"],
        forecast_df["predicted_rps"],
        label="predicted_rps",
        color="#1E88E5",
        linestyle="--",
        linewidth=1.8,
    )
    axes[0].set_ylabel("RPS")
    axes[0].set_title("Traffic Forecast")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].step(
        plan_df["time_sec"],
        plan_df["planned_cpu"],
        where="post",
        color="#E53935",
        linewidth=1.8,
        label="planned_cpu",
    )
    axes[1].set_ylabel("CPU")
    axes[1].set_title("Planned CPU Allocation")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].step(
        plan_df["time_sec"],
        plan_df["planned_replicas"],
        where="post",
        color="#43A047",
        linewidth=1.8,
        label="planned_replicas",
    )
    axes[2].set_xlabel("Time (sec)")
    axes[2].set_ylabel("Replicas")
    axes[2].set_title("Planned Replica Allocation")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="학습된 LSTM으로 미래 트래픽을 예측하고 자원 계획 CSV를 생성합니다."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--forecast-csv", type=Path, default=DEFAULT_FORECAST_CSV)
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--plan-plot", type=Path, default=DEFAULT_PLAN_PLOT)
    parser.add_argument(
        "--observed-points",
        type=int,
        default=180,
        help="실제로 관측한 과거 구간 길이. 이후 구간은 모델이 순차 예측합니다.",
    )
    args = parser.parse_args()

    schedule_df = load_schedule(args.input_csv)
    forecast_df = one_step_forecast_from_actual_history(schedule_df, args.observed_points)
    plan_df = build_plan(forecast_df)

    save_dataframe(forecast_df, args.forecast_csv)
    save_dataframe(plan_df, args.plan_csv)
    plot_plan(forecast_df, plan_df, args.plan_plot)

    print("예측 및 자원 계획 생성 완료")
    print(f"  입력 CSV          : {args.input_csv}")
    print(f"  관측 구간 길이    : {args.observed_points}")
    print(f"  예측 결과 CSV     : {args.forecast_csv}")
    print(f"  자원 계획 CSV     : {args.plan_csv}")
    print(f"  자원 계획 그래프  : {args.plan_plot}")
    print(f"  미래 예측 포인트  : {len(forecast_df)}")


if __name__ == "__main__":
    main()
