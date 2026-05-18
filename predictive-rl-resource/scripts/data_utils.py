from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    ACTUAL_TRAFFIC_CSV,
    FORECAST_TRAFFIC_CSV,
    LOOKAHEAD_WINDOW,
    REFERENCE_CORRECTION_CSV,
    REFERENCE_LOADGEN_CSV,
    REFERENCE_PLAN_CSV,
)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    return path


def load_actual_schedule(path: Path | None = None) -> pd.DataFrame:
    target = require_file(path or ACTUAL_TRAFFIC_CSV)
    df = pd.read_csv(target).copy()
    required = {"time_sec", "target_rps"}
    if not required.issubset(df.columns):
        raise ValueError(f"입력 CSV에 필요한 컬럼이 없습니다: {required}")
    df["time_sec"] = df["time_sec"].astype(int)
    df["actual_rps"] = df["target_rps"].astype(float)
    return df


def load_forecast(path: Path | None = None) -> pd.DataFrame:
    target = require_file(path or FORECAST_TRAFFIC_CSV)
    df = pd.read_csv(target).copy()
    required = {"time_sec", "predicted_rps"}
    if not required.issubset(df.columns):
        raise ValueError(f"예측 CSV에 필요한 컬럼이 없습니다: {required}")
    df["time_sec"] = df["time_sec"].astype(int)
    df["predicted_rps"] = df["predicted_rps"].astype(float)
    return df


def load_merged_schedule(
    actual_path: Path | None = None,
    forecast_path: Path | None = None,
) -> pd.DataFrame:
    actual_df = load_actual_schedule(actual_path)
    forecast_df = load_forecast(forecast_path)
    merged = actual_df.merge(
        forecast_df[["time_sec", "predicted_rps"]],
        on="time_sec",
        how="inner",
    )
    if REFERENCE_PLAN_CSV.exists():
        reference_plan = pd.read_csv(REFERENCE_PLAN_CSV).copy()
        required = {"time_sec", "planned_cpu", "planned_replicas"}
        if required.issubset(reference_plan.columns):
            reference_plan = reference_plan.rename(
                columns={
                    "planned_cpu": "teacher_cpu",
                    "planned_replicas": "teacher_replicas",
                }
            )
            reference_plan["time_sec"] = reference_plan["time_sec"].astype(int)
            merged = merged.merge(
                reference_plan[["time_sec", "teacher_cpu", "teacher_replicas"]],
                on="time_sec",
                how="left",
            )
    merged = merged.sort_values("time_sec").reset_index(drop=True)
    return enrich_schedule_features(merged)


def enrich_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["rps_gap"] = enriched["predicted_rps"] - enriched["actual_rps"]
    enriched["delta_actual_rps"] = enriched["actual_rps"].diff().fillna(0.0)
    enriched["delta_predicted_rps"] = enriched["predicted_rps"].diff().fillna(0.0)

    lookahead_values = []
    predicted = enriched["predicted_rps"].tolist()
    for idx in range(len(predicted)):
        end = min(len(predicted), idx + LOOKAHEAD_WINDOW)
        lookahead_values.append(max(predicted[idx:end]))
    enriched["lookahead_peak_rps"] = lookahead_values

    phase = []
    for _, row in enriched.iterrows():
        if row["lookahead_peak_rps"] >= 500 or row["predicted_rps"] >= 450:
            phase.append("peak")
        elif row["predicted_rps"] >= 200:
            phase.append("ramp")
        else:
            phase.append("stable")
    enriched["phase"] = phase
    if "teacher_cpu" in enriched.columns:
        enriched["teacher_cpu"] = enriched["teacher_cpu"].ffill().bfill().fillna(2.0).astype(float)
    if "teacher_replicas" in enriched.columns:
        enriched["teacher_replicas"] = enriched["teacher_replicas"].ffill().bfill().fillna(3).astype(float)
    return enriched


def load_predictive_reference_outputs() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for name, path in {
        "plan": REFERENCE_PLAN_CSV,
        "loadgen": REFERENCE_LOADGEN_CSV,
        "correction": REFERENCE_CORRECTION_CSV,
    }.items():
        if path.exists():
            outputs[name] = pd.read_csv(path)
    return outputs
