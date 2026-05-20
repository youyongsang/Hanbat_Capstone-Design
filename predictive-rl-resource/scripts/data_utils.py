from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ACTUAL_TRAFFIC_CSV,
    ACTION_CPU_AND_REP_UP,
    ACTION_CPU_DOWN,
    ACTION_CPU_UP,
    ACTION_HOLD,
    ACTION_REP_DOWN,
    ACTION_REP_UP,
    CPU_CAPACITY_POLICY,
    CPU_STEP,
    FORECAST_TRAFFIC_CSV,
    LOOKAHEAD_WINDOW,
    MAX_CPU,
    MAX_REPLICAS,
    MIN_CPU,
    MIN_REPLICAS,
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


def _load_correction_action_map() -> dict[int, str]:
    if not REFERENCE_CORRECTION_CSV.exists():
        return {}
    correction_df = pd.read_csv(REFERENCE_CORRECTION_CSV).copy()
    required = {"elapsed_sec", "action"}
    if not required.issubset(correction_df.columns):
        return {}
    correction_df["teacher_time_sec"] = correction_df["elapsed_sec"].round().astype(int)
    priority = {
        "CPU_AND_REP_UP": 5,
        "REP_UP": 4,
        "CPU_UP": 3,
        "REP_DOWN": 2,
        "CPU_DOWN": 1,
        "HOLD": 0,
    }
    action_map: dict[int, str] = {}
    for _, row in correction_df.iterrows():
        key = int(row["teacher_time_sec"])
        action = str(row["action"])
        if key not in action_map or priority.get(action, -1) > priority.get(action_map[key], -1):
            action_map[key] = action
    return action_map


def _capacity_per_replica(cpu_value: float) -> float:
    rounded_cpu = round(float(cpu_value) * 2) / 2
    rounded_cpu = min(MAX_CPU, max(MIN_CPU, rounded_cpu))
    return float(CPU_CAPACITY_POLICY[rounded_cpu])


def infer_teacher_action(
    current_cpu: float,
    current_replicas: float,
    next_cpu: float,
    next_replicas: float,
) -> int:
    cpu_delta = round(float(next_cpu) - float(current_cpu), 4)
    replica_delta = round(float(next_replicas) - float(current_replicas), 4)

    cpu_up = cpu_delta >= CPU_STEP / 2
    cpu_down = cpu_delta <= -(CPU_STEP / 2)
    rep_up = replica_delta >= 0.5
    rep_down = replica_delta <= -0.5

    if cpu_up and rep_up:
        return ACTION_CPU_AND_REP_UP
    if cpu_up:
        return ACTION_CPU_UP
    if rep_up:
        return ACTION_REP_UP
    if rep_down:
        return ACTION_REP_DOWN
    if cpu_down:
        return ACTION_CPU_DOWN
    return ACTION_HOLD


def build_teacher_dataset(
    actual_path: Path | None = None,
    forecast_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    schedule_df = load_merged_schedule(actual_path=actual_path, forecast_path=forecast_path)
    if "teacher_cpu" not in schedule_df.columns or "teacher_replicas" not in schedule_df.columns:
        raise ValueError("teacher dataset 생성을 위해 reference plan이 필요합니다.")
    if not REFERENCE_LOADGEN_CSV.exists():
        raise FileNotFoundError(f"teacher dataset 생성을 위해 loadgen 결과가 필요합니다: {REFERENCE_LOADGEN_CSV}")

    loadgen_df = pd.read_csv(REFERENCE_LOADGEN_CSV).copy()
    required = {
        "time_sec",
        "avg_latency_ms",
        "p95_latency_ms",
        "fail_count",
        "sla_violation_count",
    }
    if not required.issubset(loadgen_df.columns):
        raise ValueError(f"loadgen reference에 필요한 컬럼이 없습니다: {required}")

    loadgen_df["time_sec"] = loadgen_df["time_sec"].astype(int)
    merged = schedule_df.merge(
        loadgen_df[["time_sec", "avg_latency_ms", "p95_latency_ms", "fail_count", "sla_violation_count"]],
        on="time_sec",
        how="left",
    )
    merged["avg_latency_ms"] = merged["avg_latency_ms"].ffill().bfill().fillna(0.0)
    merged["p95_latency_ms"] = merged["p95_latency_ms"].ffill().bfill().fillna(0.0)
    merged["fail_count"] = merged["fail_count"].ffill().bfill().fillna(0.0)
    merged["sla_violation_count"] = merged["sla_violation_count"].ffill().bfill().fillna(0.0)
    correction_action_map = _load_correction_action_map()

    teacher_obs: list[list[float]] = []
    teacher_actions: list[int] = []

    prev_avg_latency = 0.0
    prev_p95_latency = 0.0
    for idx, row in merged.iterrows():
        current_cpu = float(row["teacher_cpu"])
        current_replicas = float(row["teacher_replicas"])
        next_cpu = current_cpu if idx == len(merged) - 1 else float(merged.iloc[idx + 1]["teacher_cpu"])
        next_replicas = current_replicas if idx == len(merged) - 1 else float(merged.iloc[idx + 1]["teacher_replicas"])

        capacity = _capacity_per_replica(current_cpu) * max(current_replicas, MIN_REPLICAS)
        cpu_usage_pct = min(100.0, (float(row["actual_rps"]) / max(capacity, 1.0)) * 100.0)
        avg_latency = float(row["avg_latency_ms"])
        p95_latency = float(row["p95_latency_ms"])
        fail_count = float(row["fail_count"])
        sla_violation_count = float(row["sla_violation_count"])

        teacher_obs.append(
            [
                float(row["actual_rps"]),
                float(row["predicted_rps"]),
                float(row.get("lookahead_peak_rps", row["predicted_rps"])),
                cpu_usage_pct,
                avg_latency,
                p95_latency,
                fail_count,
                sla_violation_count,
                current_cpu,
                current_replicas,
                current_cpu,
                current_replicas,
                float(row.get("rps_gap", row["predicted_rps"] - row["actual_rps"])),
                float(row.get("delta_actual_rps", 0.0)),
                avg_latency - prev_avg_latency,
                p95_latency - prev_p95_latency,
            ]
        )
        teacher_action = ACTION_HOLD
        correction_action = correction_action_map.get(int(row["time_sec"]))
        if correction_action == "CPU_AND_REP_UP":
            teacher_action = ACTION_CPU_AND_REP_UP
        elif correction_action == "REP_UP":
            teacher_action = ACTION_REP_UP
        elif correction_action == "CPU_UP" and teacher_action == ACTION_HOLD:
            teacher_action = ACTION_CPU_UP
        elif correction_action == "REP_DOWN" and teacher_action == ACTION_HOLD:
            teacher_action = ACTION_REP_DOWN
        elif correction_action == "CPU_DOWN" and teacher_action == ACTION_HOLD:
            teacher_action = ACTION_CPU_DOWN
        teacher_actions.append(teacher_action)
        prev_avg_latency = avg_latency
        prev_p95_latency = p95_latency

    return np.asarray(teacher_obs, dtype=np.float32), np.asarray(teacher_actions, dtype=np.int64)
