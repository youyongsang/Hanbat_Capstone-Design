from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import ACTION_CPU_DOWN, ACTION_NAMES, ACTION_REP_DOWN, MIN_PLAN_HOLD_STEPS, RL_PLAN_PATH, TRAINED_MODEL_PATH
from data_utils import load_merged_schedule
from resource_env import ResourceAllocationEnv


def main():
    try:
        from stable_baselines3 import PPO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3가 필요합니다. requirements.txt를 먼저 설치하세요.") from exc

    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"학습 모델이 없습니다: {TRAINED_MODEL_PATH}")

    schedule_df = load_merged_schedule()
    env = ResourceAllocationEnv(schedule_df)
    model = PPO.load(TRAINED_MODEL_PATH)

    obs, _ = env.reset()
    done = False
    rows = []
    hold_steps_since_change = 0

    while not done:
        current_idx = env.step_idx
        row = schedule_df.iloc[current_idx]
        action, _ = model.predict(obs, deterministic=True)
        if rows and hold_steps_since_change < MIN_PLAN_HOLD_STEPS and int(action) in (ACTION_CPU_DOWN, ACTION_REP_DOWN):
            action = 0
        obs, reward, done, _, info = env.step(action)
        if rows and (
            info["current_cpu"] != rows[-1]["planned_cpu"]
            or info["current_replicas"] != rows[-1]["planned_replicas"]
        ):
            hold_steps_since_change = 0
        else:
            hold_steps_since_change += 1
        rows.append({
            "time_sec": int(row["time_sec"]),
            "actual_rps": float(row["actual_rps"]),
            "predicted_rps": float(row["predicted_rps"]),
            "planned_cpu": info["current_cpu"],
            "planned_replicas": info["current_replicas"],
            "action": ACTION_NAMES[int(action)],
            "sim_avg_latency_ms": info["avg_latency_ms"],
            "sim_p95_latency_ms": info["p95_latency_ms"],
            "sim_fail_count": info["fail_count"],
            "sim_sla_violation_count": info["sla_violation_count"],
            "reward": reward,
        })

    RL_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RL_PLAN_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] RL 계획 CSV 저장: {RL_PLAN_PATH}")


if __name__ == "__main__":
    main()
