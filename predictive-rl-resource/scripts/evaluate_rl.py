from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import EVAL_SUMMARY_PATH, TRAINED_MODEL_PATH
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
    total_reward = 0.0
    rows = []
    actions = Counter()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions[int(action)] += 1
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        rows.append(info)

    if not rows:
        raise RuntimeError("평가 결과가 비어 있습니다.")

    peak_rows = [r for r in rows if 400 <= r["time_sec"] <= 459]
    summary = {
        "steps": len(rows),
        "total_reward": round(total_reward, 3),
        "avg_latency_ms": round(sum(r["avg_latency_ms"] for r in rows) / len(rows), 3),
        "avg_p95_latency_ms": round(sum(r["p95_latency_ms"] for r in rows) / len(rows), 3),
        "max_p95_latency_ms": round(max(r["p95_latency_ms"] for r in rows), 3),
        "total_fail_count": int(sum(r["fail_count"] for r in rows)),
        "total_sla_violation_count": int(sum(r["sla_violation_count"] for r in rows)),
        "avg_replicas": round(sum(r["current_replicas"] for r in rows) / len(rows), 3),
        "replica_time": int(sum(r["current_replicas"] for r in rows)),
        "avg_cpu": round(sum(r["current_cpu"] for r in rows) / len(rows), 3),
        "cpu_time": round(sum(r["current_cpu"] for r in rows), 3),
        "peak_avg_latency_ms": round(sum(r["avg_latency_ms"] for r in peak_rows) / max(1, len(peak_rows)), 3),
        "peak_avg_p95_latency_ms": round(sum(r["p95_latency_ms"] for r in peak_rows) / max(1, len(peak_rows)), 3),
        "peak_total_fail_count": int(sum(r["fail_count"] for r in peak_rows)),
        "peak_total_sla_violation_count": int(sum(r["sla_violation_count"] for r in peak_rows)),
        "peak_avg_replicas": round(sum(r["current_replicas"] for r in peak_rows) / max(1, len(peak_rows)), 3),
        "action_counts": dict(actions),
    }

    EVAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] 평가 요약 저장: {EVAL_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
