import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "lstm_model.h5"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

DEFAULT_INPUT_CSV = BASE_DIR / "data" / "input" / "sale_event_traffic.csv"
DEFAULT_FORECAST_CSV = BASE_DIR / "data" / "output" / "predicted_traffic.csv"
DEFAULT_PLAN_CSV = BASE_DIR / "data" / "output" / "resource_allocation_plan.csv"

WINDOW_SIZE = 12
CONTAINER_CAPACITY = 80
SAFETY_MARGIN = 1.2
MIN_REPLICAS = 1
MAX_REPLICAS = 5
MIN_CPU = 0.5
MAX_CPU = 3.0


def load_lstm_predictor():
    try:
        from tensorflow.keras.models import load_model  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "tensorflow.keras를 불러오지 못했습니다. 먼저 TensorFlow를 설치하세요."
        ) from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"LSTM 모델 파일이 없습니다: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler 파일이 없습니다: {SCALER_PATH}")

    # 예측 전용 로드이므로 학습 당시의 loss/metric 복원 없이 읽는다.
    model = load_model(str(MODEL_PATH), compile=False)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def compute_allocation(pred_rps: float) -> tuple[float, int]:
    safe_rps = pred_rps * SAFETY_MARGIN

    replicas = int(np.ceil(safe_rps / CONTAINER_CAPACITY))
    replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, replicas))

    rps_per_container = safe_rps / replicas if replicas > 0 else 0.0
    if rps_per_container <= 40:
        cpu = 0.5
    elif rps_per_container <= 80:
        cpu = 1.0
    elif rps_per_container <= 120:
        cpu = 2.0
    else:
        cpu = 3.0

    cpu = min(MAX_CPU, max(MIN_CPU, cpu))
    return cpu, replicas


def load_schedule(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"time_sec", "target_rps"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {required}")

    df = df.copy()
    df["target_rps"] = df["target_rps"].astype(float)
    return df


def autoregressive_forecast(df: pd.DataFrame, observed_points: int) -> pd.DataFrame:
    model, scaler = load_lstm_predictor()

    if observed_points < WINDOW_SIZE:
        raise ValueError(
            f"observed_points는 최소 {WINDOW_SIZE} 이상이어야 합니다. 현재 값: {observed_points}"
        )
    if observed_points >= len(df):
        raise ValueError(
            "observed_points가 전체 길이 이상입니다. 미래 구간이 남도록 더 작은 값을 사용하세요."
        )

    history = df["target_rps"].iloc[:observed_points].tolist()
    rows = []

    for _, row in df.iloc[:observed_points].iterrows():
        rows.append({
            "time_sec": int(row["time_sec"]),
            "actual_rps": float(row["target_rps"]),
            "predicted_rps": float(row["target_rps"]),
            "phase": row["phase"] if "phase" in df.columns else "",
            "scenario": row["scenario"] if "scenario" in df.columns else "",
            "source": "observed_history",
        })

    for _, row in df.iloc[observed_points:].iterrows():
        recent = np.array(history[-WINDOW_SIZE:]).reshape(-1, 1)
        scaled = scaler.transform(recent)
        model_input = scaled.reshape(1, WINDOW_SIZE, 1)
        pred_scaled = model.predict(model_input, verbose=0)
        pred_rps = float(scaler.inverse_transform(pred_scaled)[0][0])
        pred_rps = max(0.0, pred_rps)

        rows.append({
            "time_sec": int(row["time_sec"]),
            "actual_rps": float(row["target_rps"]),
            "predicted_rps": pred_rps,
            "phase": row["phase"] if "phase" in df.columns else "",
            "scenario": row["scenario"] if "scenario" in df.columns else "",
            "source": "model_forecast",
        })
        history.append(pred_rps)

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


def main():
    parser = argparse.ArgumentParser(
        description="학습된 LSTM으로 미래 트래픽과 자원 할당 계획 CSV를 생성합니다."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--forecast-csv", type=Path, default=DEFAULT_FORECAST_CSV)
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument(
        "--observed-points",
        type=int,
        default=60,
        help="실제로 본 과거 구간 길이. 이 이후 구간은 모델이 순차 예측합니다.",
    )
    args = parser.parse_args()

    schedule_df = load_schedule(args.input_csv)
    forecast_df = autoregressive_forecast(schedule_df, args.observed_points)
    plan_df = build_plan(forecast_df)

    save_dataframe(forecast_df, args.forecast_csv)
    save_dataframe(plan_df, args.plan_csv)

    print("예측 및 자원 계획 생성 완료")
    print(f"  입력 CSV          : {args.input_csv}")
    print(f"  관측 구간 길이    : {args.observed_points}")
    print(f"  예측 결과 CSV     : {args.forecast_csv}")
    print(f"  자원 계획 CSV     : {args.plan_csv}")
    print(f"  미래 예측 포인트  : {len(forecast_df)}")


if __name__ == "__main__":
    main()
