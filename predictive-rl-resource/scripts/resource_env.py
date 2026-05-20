from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gymnasium이 필요합니다. requirements.txt를 먼저 설치하세요.") from exc

from config import (
    ACTION_CPU_AND_REP_UP,
    ACTION_CPU_DOWN,
    ACTION_CPU_UP,
    ACTION_HOLD,
    ACTION_REP_DOWN,
    ACTION_REP_UP,
    AVG_LATENCY_PENALTY,
    CPU_COST_PENALTY,
    CPU_STEP,
    FAIL_PENALTY,
    IDLE_HIGH_CPU_THRESHOLD,
    IDLE_HIGH_REPLICA_THRESHOLD,
    IDLE_PREDICTED_RPS_THRESHOLD,
    IDLE_RESOURCE_PENALTY,
    IDLE_RPS_THRESHOLD,
    LOW_TAIL_BONUS,
    MAX_CPU,
    MAX_REPLICAS,
    MIN_CPU,
    MIN_REPLICAS,
    NON_PEAK_CPU_COST_MULTIPLIER,
    NON_PEAK_REPLICA_COST_MULTIPLIER,
    NON_PEAK_UPSCALE_PENALTY,
    P95_LATENCY_PENALTY,
    PEAK_CPU_COST_MULTIPLIER,
    PEAK_BONUS_THRESHOLD_RPS,
    PEAK_PROTECTION_BONUS,
    PEAK_UNDER_REPLICATION_CPU_PENALTY,
    PEAK_UNDER_REPLICATION_PENALTY,
    PEAK_REPLICA_COST_MULTIPLIER,
    REPLICA_COST_PENALTY,
    SCALE_DOWN_CPU_BONUS,
    SCALE_DOWN_MAX_AVG_LATENCY_MS,
    SCALE_DOWN_MAX_P95_MS,
    SCALE_DOWN_REPLICA_BONUS,
    SCALE_DOWN_STABLE_BONUS,
    SLA_PENALTY,
    STEP_CHANGE_PENALTY,
    SUCCESS_REWARD,
    TEACHER_ALIGNMENT_BONUS,
    TEACHER_CPU_DISTANCE_PENALTY,
    TEACHER_PEAK_ALIGNMENT_BONUS,
    TEACHER_REPLICA_DISTANCE_PENALTY,
    UNDER_SLA_BONUS,
    WARMUP_PENALTY_STEPS,
)
from simulator import SimStepResult, simulate_service_step


class ResourceAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, schedule_df, mode: str = "full"):
        super().__init__()
        self.mode = mode
        self.schedule_df = self._select_schedule(schedule_df.reset_index(drop=True), mode)
        self.max_steps = len(self.schedule_df)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, MIN_CPU, MIN_REPLICAS, MIN_CPU, MIN_REPLICAS, -1000, -1000, -8000, -8000], dtype=np.float32),
            high=np.array([1000, 1000, 1000, 100, 12000, 20000, 4000, 4000, MAX_CPU, MAX_REPLICAS, MAX_CPU, MAX_REPLICAS, 1000, 1000, 12000, 12000], dtype=np.float32),
            dtype=np.float32,
        )

        self.step_idx = 0
        self.current_cpu = MIN_CPU
        self.current_replicas = MIN_REPLICAS
        self.last_result = SimStepResult(0, 0, 0, 0, 0, 0)
        self.prev_avg_latency = 0.0
        self.prev_p95_latency = 0.0
        self.warmup_steps_left = 0
        self.last_action = ACTION_HOLD
        self.prev_cpu_alloc = MIN_CPU
        self.prev_replica_alloc = MIN_REPLICAS

    def _select_schedule(self, schedule_df, mode: str):
        if mode == "easy":
            filtered = schedule_df[schedule_df["time_sec"] < 350].copy()
        elif mode == "peak":
            filtered = schedule_df[(schedule_df["time_sec"] >= 300) & (schedule_df["time_sec"] <= 700)].copy()
        else:
            filtered = schedule_df.copy()
        if filtered.empty:
            return schedule_df.copy()
        return filtered.reset_index(drop=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.current_cpu = MIN_CPU
        self.current_replicas = MIN_REPLICAS
        self.prev_avg_latency = 0.0
        self.prev_p95_latency = 0.0
        self.warmup_steps_left = 0
        self.last_action = ACTION_HOLD
        first = self.schedule_df.iloc[self.step_idx]
        self.current_cpu = float(first.get("teacher_cpu", MIN_CPU))
        self.current_replicas = int(round(float(first.get("teacher_replicas", MIN_REPLICAS))))
        self.prev_cpu_alloc = self.current_cpu
        self.prev_replica_alloc = self.current_replicas
        self.last_result = simulate_service_step(
            actual_rps=float(first["actual_rps"]),
            predicted_rps=float(first["predicted_rps"]),
            cpu=self.current_cpu,
            replicas=self.current_replicas,
            lookahead_peak_rps=float(first.get("lookahead_peak_rps", first["predicted_rps"])),
            delta_rps=float(first.get("delta_actual_rps", 0.0)),
            warmup_steps_left=self.warmup_steps_left,
        )
        return self._get_obs(first), {}

    def step(self, action):
        current = self.schedule_df.iloc[self.step_idx]

        self.prev_avg_latency = self.last_result.avg_latency_ms
        self.prev_p95_latency = self.last_result.p95_latency_ms
        self._apply_action(int(action), current)
        self.last_action = int(action)
        result = simulate_service_step(
            actual_rps=float(current["actual_rps"]),
            predicted_rps=float(current["predicted_rps"]),
            cpu=self.current_cpu,
            replicas=self.current_replicas,
            lookahead_peak_rps=float(current.get("lookahead_peak_rps", current["predicted_rps"])),
            delta_rps=float(current.get("delta_actual_rps", 0.0)),
            warmup_steps_left=self.warmup_steps_left,
        )
        self.last_result = result
        if self.warmup_steps_left > 0:
            self.warmup_steps_left -= 1

        reward = self._compute_reward(result, current)

        info = {
            "time_sec": int(current["time_sec"]),
            "actual_rps": float(current["actual_rps"]),
            "predicted_rps": float(current["predicted_rps"]),
            "lookahead_peak_rps": float(current.get("lookahead_peak_rps", current["predicted_rps"])),
            "current_cpu": self.current_cpu,
            "current_replicas": self.current_replicas,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "fail_count": result.fail_count,
            "sla_violation_count": result.sla_violation_count,
            "cpu_usage_pct": result.cpu_usage_pct,
            "success_count": result.success_count,
        }

        self.step_idx += 1
        terminated = self.step_idx >= self.max_steps
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._get_obs(self.schedule_df.iloc[self.step_idx])

        return next_obs, reward, terminated, False, info

    def _apply_action(self, action: int, row):
        prev_cpu = self.current_cpu
        prev_replicas = self.current_replicas
        self.prev_cpu_alloc = prev_cpu
        self.prev_replica_alloc = prev_replicas

        teacher_cpu = float(row.get("teacher_cpu", MIN_CPU))
        teacher_replicas = int(round(float(row.get("teacher_replicas", MIN_REPLICAS))))
        target_cpu = teacher_cpu
        target_replicas = teacher_replicas

        if action == ACTION_CPU_UP:
            target_cpu = min(MAX_CPU, teacher_cpu + CPU_STEP)
        elif action == ACTION_CPU_DOWN:
            target_cpu = max(MIN_CPU, teacher_cpu - CPU_STEP)
        elif action == ACTION_REP_UP:
            target_replicas = min(MAX_REPLICAS, teacher_replicas + 1)
        elif action == ACTION_REP_DOWN:
            target_replicas = max(MIN_REPLICAS, teacher_replicas - 1)
        elif action == ACTION_CPU_AND_REP_UP:
            target_cpu = min(MAX_CPU, teacher_cpu + CPU_STEP)
            target_replicas = min(MAX_REPLICAS, teacher_replicas + 1)
        elif action == ACTION_HOLD:
            pass

        self.current_cpu = target_cpu
        self.current_replicas = target_replicas
        if self.current_cpu > prev_cpu or self.current_replicas > prev_replicas:
            self.warmup_steps_left = WARMUP_PENALTY_STEPS

    def _get_obs(self, row) -> np.ndarray:
        current_rps = float(row["actual_rps"])
        predicted_rps = float(row["predicted_rps"])
        lookahead_peak_rps = float(row.get("lookahead_peak_rps", predicted_rps))
        teacher_cpu = float(row.get("teacher_cpu", self.current_cpu))
        teacher_replicas = float(row.get("teacher_replicas", self.current_replicas))
        gap = float(row.get("rps_gap", predicted_rps - current_rps))
        delta_rps = float(row.get("delta_actual_rps", 0.0))
        delta_avg_latency = self.last_result.avg_latency_ms - self.prev_avg_latency
        delta_p95_latency = self.last_result.p95_latency_ms - self.prev_p95_latency
        obs = np.array(
            [
                current_rps,
                predicted_rps,
                lookahead_peak_rps,
                self.last_result.cpu_usage_pct,
                self.last_result.avg_latency_ms,
                self.last_result.p95_latency_ms,
                self.last_result.fail_count,
                self.last_result.sla_violation_count,
                self.current_cpu,
                self.current_replicas,
                teacher_cpu,
                teacher_replicas,
                gap,
                delta_rps,
                delta_avg_latency,
                delta_p95_latency,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self, result: SimStepResult, row) -> float:
        reward = 0.0
        phase = str(row.get("phase", "stable"))
        lookahead_peak = float(row.get("lookahead_peak_rps", row["predicted_rps"]))
        predicted_rps = float(row["predicted_rps"])
        actual_rps = float(row["actual_rps"])
        teacher_cpu = float(row.get("teacher_cpu", self.current_cpu))
        teacher_replicas = float(row.get("teacher_replicas", self.current_replicas))
        peak_mode = phase == "peak" or lookahead_peak >= PEAK_BONUS_THRESHOLD_RPS
        cpu_cost_multiplier = PEAK_CPU_COST_MULTIPLIER if peak_mode else NON_PEAK_CPU_COST_MULTIPLIER
        replica_cost_multiplier = PEAK_REPLICA_COST_MULTIPLIER if peak_mode else NON_PEAK_REPLICA_COST_MULTIPLIER

        reward += SUCCESS_REWARD * result.success_count
        reward -= FAIL_PENALTY * result.fail_count
        reward -= SLA_PENALTY * result.sla_violation_count
        reward -= AVG_LATENCY_PENALTY * result.avg_latency_ms
        reward -= P95_LATENCY_PENALTY * result.p95_latency_ms
        reward -= CPU_COST_PENALTY * cpu_cost_multiplier * self.current_cpu
        reward -= REPLICA_COST_PENALTY * replica_cost_multiplier * self.current_replicas
        reward -= TEACHER_CPU_DISTANCE_PENALTY * abs(self.current_cpu - teacher_cpu)
        reward -= TEACHER_REPLICA_DISTANCE_PENALTY * abs(self.current_replicas - teacher_replicas)
        if self.last_action != ACTION_HOLD:
            reward -= STEP_CHANGE_PENALTY
        if result.avg_latency_ms <= 700 and result.sla_violation_count == 0:
            reward += UNDER_SLA_BONUS
        if result.p95_latency_ms <= 900 and result.fail_count == 0:
            reward += LOW_TAIL_BONUS
        if peak_mode and self.current_replicas >= 5:
            reward += PEAK_PROTECTION_BONUS
        if peak_mode and self.current_replicas + 0.5 < teacher_replicas:
            reward -= PEAK_UNDER_REPLICATION_PENALTY * (teacher_replicas - self.current_replicas)
        if peak_mode and self.current_cpu + (CPU_STEP / 2) < teacher_cpu:
            reward -= PEAK_UNDER_REPLICATION_CPU_PENALTY * max(0.0, teacher_cpu - self.current_cpu)
        if abs(self.current_cpu - teacher_cpu) <= 0.5 and abs(self.current_replicas - teacher_replicas) <= 1.0:
            reward += TEACHER_ALIGNMENT_BONUS
            if peak_mode:
                reward += TEACHER_PEAK_ALIGNMENT_BONUS
        if (
            not peak_mode
            and self.last_action in (ACTION_CPU_UP, ACTION_REP_UP, ACTION_CPU_AND_REP_UP)
            and actual_rps < IDLE_RPS_THRESHOLD
            and predicted_rps < IDLE_PREDICTED_RPS_THRESHOLD
        ):
            reward -= NON_PEAK_UPSCALE_PENALTY
        if (
            actual_rps < IDLE_RPS_THRESHOLD
            and predicted_rps < IDLE_PREDICTED_RPS_THRESHOLD
            and (self.current_replicas >= IDLE_HIGH_REPLICA_THRESHOLD or self.current_cpu >= IDLE_HIGH_CPU_THRESHOLD)
        ):
            reward -= IDLE_RESOURCE_PENALTY
        if (
            self.last_action in (ACTION_CPU_DOWN, ACTION_REP_DOWN)
            and result.fail_count == 0
            and result.sla_violation_count == 0
            and result.avg_latency_ms <= SCALE_DOWN_MAX_AVG_LATENCY_MS
            and result.p95_latency_ms <= SCALE_DOWN_MAX_P95_MS
        ):
            reward += SCALE_DOWN_STABLE_BONUS
            if self.current_cpu < self.prev_cpu_alloc:
                reward += SCALE_DOWN_CPU_BONUS
            if self.current_replicas < self.prev_replica_alloc:
                reward += SCALE_DOWN_REPLICA_BONUS
        return float(reward)
