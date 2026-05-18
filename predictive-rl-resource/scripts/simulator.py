from __future__ import annotations

from dataclasses import dataclass

from config import (
    CPU_CAPACITY_POLICY,
    SLA_LATENCY_MS,
    WARMUP_LATENCY_MS,
    WARMUP_P95_MS,
)


@dataclass
class SimStepResult:
    avg_latency_ms: float
    p95_latency_ms: float
    fail_count: int
    sla_violation_count: int
    cpu_usage_pct: float
    success_count: int


def _capacity_for_cpu(cpu: float) -> float:
    rounded = round(cpu * 2) / 2
    return CPU_CAPACITY_POLICY.get(rounded, CPU_CAPACITY_POLICY[max(CPU_CAPACITY_POLICY.keys())])


def simulate_service_step(
    actual_rps: float,
    predicted_rps: float,
    cpu: float,
    replicas: int,
    lookahead_peak_rps: float = 0.0,
    delta_rps: float = 0.0,
    warmup_steps_left: int = 0,
) -> SimStepResult:
    per_replica_capacity = _capacity_for_cpu(cpu)
    total_capacity = per_replica_capacity * replicas
    load_ratio = actual_rps / max(total_capacity, 1.0)
    prediction_gap = max(0.0, actual_rps - predicted_rps)
    peak_pressure = max(0.0, lookahead_peak_rps - actual_rps)
    rising_pressure = max(0.0, delta_rps)

    # CPU 사용률은 부하 비율과 예측 오차를 함께 반영한다.
    cpu_usage = min(
        100.0,
        load_ratio * 92.0
        + (prediction_gap / max(actual_rps, 1.0)) * 20.0
        + (rising_pressure / max(actual_rps, 1.0)) * 10.0,
    )

    # 평균 latency는 임계 이전에도 더 민감하게 증가하고,
    # overload 이후에는 p95/fail이 더 빠르게 나빠지도록 보수적으로 근사한다.
    base_latency = 150.0 + load_ratio * 190.0 + min(peak_pressure * 0.08, 120.0)
    if load_ratio <= 0.75:
        avg_latency = base_latency + load_ratio * 85.0
    elif load_ratio <= 0.95:
        avg_latency = 310.0 + (load_ratio - 0.75) * 1450.0 + rising_pressure * 0.35
    elif load_ratio <= 1.10:
        avg_latency = 600.0 + (load_ratio - 0.95) * 2600.0 + prediction_gap * 0.45 + rising_pressure * 0.50
    else:
        avg_latency = (
            990.0
            + (load_ratio - 1.10) * 4800.0
            + prediction_gap * 0.65
            + rising_pressure * 0.75
            + min(peak_pressure * 0.12, 250.0)
        )

    # p95는 평균보다 더 큰 tail penalty를 반영한다.
    tail_multiplier = 1.22 + min(load_ratio, 1.25) * 0.42
    p95_latency = avg_latency * tail_multiplier
    if load_ratio > 0.95:
        p95_latency += (
            max(0.0, load_ratio - 0.95) * 3600.0
            + prediction_gap * 0.55
            + rising_pressure * 0.80
        )
    if load_ratio > 1.10:
        p95_latency += max(0.0, load_ratio - 1.10) * 6200.0

    if warmup_steps_left > 0:
        warmup_factor = min(1.0, warmup_steps_left / 5.0)
        avg_latency += WARMUP_LATENCY_MS * warmup_factor
        p95_latency += WARMUP_P95_MS * warmup_factor

    overload = max(0.0, actual_rps - total_capacity)
    if overload <= 0:
        fail_count = 0
    elif load_ratio <= 1.05:
        fail_count = int(overload * 0.22)
    elif load_ratio <= 1.20:
        fail_count = int(overload * 0.45 + rising_pressure * 0.08)
    else:
        fail_count = int(overload * 0.70 + rising_pressure * 0.15)
    fail_count = min(fail_count, int(actual_rps))
    success_count = max(0, int(actual_rps) - fail_count)

    # SLA 위반은 p95와 평균 latency가 모두 나빠질수록 증가하도록 근사한다.
    if avg_latency <= SLA_LATENCY_MS and p95_latency <= SLA_LATENCY_MS:
        sla_ratio = max(0.0, (p95_latency - 650.0) / 2400.0)
    else:
        avg_over = max(0.0, avg_latency - SLA_LATENCY_MS)
        p95_over = max(0.0, p95_latency - SLA_LATENCY_MS)
        sla_ratio = min(1.0, 0.16 + avg_over / 1800.0 + p95_over / 2200.0)

    sla_violation_count = min(success_count, int(success_count * sla_ratio))

    return SimStepResult(
        avg_latency_ms=round(avg_latency, 3),
        p95_latency_ms=round(p95_latency, 3),
        fail_count=fail_count,
        sla_violation_count=sla_violation_count,
        cpu_usage_pct=round(cpu_usage, 3),
        success_count=success_count,
    )
