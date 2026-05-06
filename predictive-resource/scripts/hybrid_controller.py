import csv
import time
import subprocess
from pathlib import Path
from collections import deque

import requests

# =========================
# 경로 설정
# =========================

BASE_DIR   = Path(__file__).resolve().parent.parent   # predictive-resource/
OUTPUT_CSV = BASE_DIR / "data" / "output" / "hybrid_correction_log.csv"
LOADGEN_CSV = BASE_DIR / "data" / "output" / "loadgen_result.csv"

# =========================
# 보정 정책 파라미터
# =========================

BASE_PORT               = 8001

def get_metrics_url(idx: int) -> str:
    return f"http://localhost:{BASE_PORT + (idx - 1)}/metrics?window_sec=5"


LATENCY_WARN_MS  = 500.0
P95_WARN_MS      = 750.0
CPU_USAGE_WARN   = 60.0
FAIL_THRESHOLD   = 1

LATENCY_EASY_MS  = 250.0
P95_EASY_MS      = 500.0
CPU_USAGE_EASY   = 30.0
MIN_RPS_SCALE_IN = 5.0
CPU_CONTAINER_OUT_THRESHOLD = 70.0
P95_REPLICA_UP_MS = 850.0
LAT_REPLICA_UP_MS = 600.0
MAX_CPU_NEAR_THRESHOLD = 5.0

CPU_STEP  = 0.5
MIN_CPU   = 2.0
MAX_CPU   = 6.0
MIN_REPS  = 3
MAX_REPS  = 7

CHECK_INTERVAL = 2
CPU_SCALE_OUT_COOLDOWN = 5
REPLICA_SCALE_OUT_COOLDOWN = 10
SCALE_IN_COOLDOWN  = 30
OVERRIDE_HOLD_SEC  = 30
SCALE_DOWN_HOLD_SEC = 60
CPU_SCALE_IN_CONFIRMATIONS = 3
REPLICA_SCALE_IN_CONFIRMATIONS = 5

IMAGE_NAME       = "reactive-server"
CONTAINER_PREFIX = "app_server_"


# =========================
# Docker 헬퍼
# =========================

def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_success(result: subprocess.CompletedProcess, context: str):
    if result.returncode != 0:
        raise RuntimeError(
            f"{context} failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
        )


def container_name(index: int) -> str:
    return f"{CONTAINER_PREFIX}{index}"


def list_all_names() -> list:
    r = run_cmd(["docker", "ps", "-a", "--format", "{{.Names}}"])
    require_success(r, "docker ps -a")
    return r.stdout.splitlines()


def container_exists(name: str) -> bool:
    return name in list_all_names()


def update_cpu(name: str, cpu: float):
    result = run_cmd(["docker", "update", f"--cpus={cpu}", name])
    require_success(result, f"docker update {name}")


def update_all_cpu(replicas: int, cpu: float):
    for i in range(1, replicas + 1):
        n = container_name(i)
        if container_exists(n):
            update_cpu(n, cpu)


def start_container(index: int, cpu: float):
    name = container_name(index)
    port = BASE_PORT + (index - 1)
    if container_exists(name):
        update_cpu(name, cpu)
        return
    result = run_cmd([
        "docker", "run", "-d",
        "--name", name,
        f"--cpus={cpu}",
        "-p", f"{port}:8000",
        IMAGE_NAME,
    ])
    require_success(result, f"docker run {name}")
    print(f"  [DOCKER] {name} 보정 시작 (port={port}, cpu={cpu})")


def remove_container(index: int):
    name = container_name(index)
    if not container_exists(name):
        return
    result = run_cmd(["docker", "rm", "-f", name])
    require_success(result, f"docker rm -f {name}")
    print(f"  [DOCKER] {name} 보정 제거")


def ensure_replicas(target: int, cpu: float):
    for i in range(1, target + 1):
        start_container(i, cpu)
    for i in range(MAX_REPS, target, -1):
        remove_container(i)


def get_cpu_usage(name: str) -> float:
    r = run_cmd([
        "docker", "stats", "--no-stream",
        "--format", "{{.CPUPerc}}", name,
    ])
    require_success(r, f"docker stats {name}")
    val = r.stdout.strip().replace("%", "").strip()
    try:
        return float(val) if val else 0.0
    except ValueError:
        return 0.0


def get_cluster_cpu_usage(replicas: int) -> tuple[float, float]:
    usages = []
    for idx in range(1, replicas + 1):
        name = container_name(idx)
        if not container_exists(name):
            continue
        try:
            usages.append(get_cpu_usage(name))
        except Exception:
            continue

    if not usages:
        return 0.0, 0.0

    return sum(usages) / len(usages), max(usages)


# =========================
# Metrics 수집
# =========================

def fetch_metrics_for_replicas(replicas: int) -> dict:
    samples = []
    for idx in range(1, replicas + 1):
        try:
            resp = requests.get(get_metrics_url(idx), timeout=2)
            resp.raise_for_status()
            samples.append(resp.json())
        except Exception:
            continue

    if not samples:
        return {}

    return {
        "avg_latency_ms": max(s.get("avg_latency_ms", 0.0) for s in samples),
        "p95_latency_ms": max(s.get("p95_latency_ms", 0.0) for s in samples),
        "fail_count": sum(int(s.get("fail_count", 0)) for s in samples),
        "request_count": sum(int(s.get("request_count", 0)) for s in samples),
        "current_rps": sum(float(s.get("current_rps", 0.0)) for s in samples),
        "container_sample_count": len(samples),
    }


def read_recent_loadgen_metrics(max_rows: int = 3) -> dict:
    if not LOADGEN_CSV.exists():
        return {}

    try:
        with open(LOADGEN_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            recent = deque(reader, maxlen=max_rows)
    except Exception:
        return {}

    if not recent:
        return {}

    def to_float(row: dict, key: str) -> float:
        try:
            return float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def to_int(row: dict, key: str) -> int:
        try:
            return int(float(row.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    avg_lat = max(to_float(r, "avg_latency_ms") for r in recent)
    p95_lat = max(to_float(r, "p95_latency_ms") for r in recent)
    fail_cnt = sum(to_int(r, "fail_count") for r in recent)
    sla_cnt = sum(to_int(r, "sla_violation_count") for r in recent)
    latest = recent[-1]

    return {
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "fail_count": fail_cnt,
        "sla_violation_count": sla_cnt,
        "current_rps": to_float(latest, "target_rps"),
        "time_sec": to_int(latest, "time_sec"),
    }


# =========================
# 보정 의사결정
# =========================

def decide_correction(
    metrics: dict,
    cpu_usage_avg: float,
    cpu_usage_max: float,
    curr_cpu: float,
    curr_replicas: int,
    pred_cpu: float,
    pred_replicas: int,
    easy_count: int,
    now: float,
    last_scale_out_time: float,
    last_scale_in_time: float,
) -> dict:
    """
    예측 레이어가 확보한 (pred_cpu, pred_replicas) 위에서
    실제 성능 지표의 편차를 보고 보정 방향을 결정한다.

    보정 원칙
    ---------
    1. 과부하  → CPU 먼저 증가, CPU 최대 + 심각 과부하면 컨테이너 추가
    2. 여유    → 예측 수준(pred_*)으로 복귀 (최솟값까지 내리지 않음)
    3. 정상    → HOLD
    """
    avg_lat  = metrics.get("avg_latency_ms", 0.0)
    p95_lat  = metrics.get("p95_latency_ms", 0.0)
    fail_cnt = metrics.get("fail_count", 0)
    sla_cnt  = metrics.get("sla_violation_count", 0)
    cur_rps  = metrics.get("current_rps", 0.0)
    overloaded = (
        avg_lat  > LATENCY_WARN_MS
        or p95_lat  > P95_WARN_MS
        or fail_cnt >= FAIL_THRESHOLD
        or sla_cnt > 0
        or cpu_usage_avg > CPU_USAGE_WARN
        or cpu_usage_max > CPU_USAGE_WARN
    )
    underloaded = (
        avg_lat  < LATENCY_EASY_MS
        and p95_lat < P95_EASY_MS
        and cpu_usage_avg < CPU_USAGE_EASY
        and fail_cnt == 0
        and sla_cnt == 0
        and cur_rps  >= MIN_RPS_SCALE_IN
    )

    can_scale_in = (now - last_scale_in_time) >= SCALE_IN_COOLDOWN
    can_cpu_scale_out = (now - last_scale_out_time.get("cpu", 0.0)) >= CPU_SCALE_OUT_COOLDOWN
    can_replica_scale_out = (now - last_scale_out_time.get("replica", 0.0)) >= REPLICA_SCALE_OUT_COOLDOWN

    if overloaded:
        severe_latency = (
            avg_lat > LAT_REPLICA_UP_MS
            or p95_lat > P95_REPLICA_UP_MS
            or fail_cnt > 0
            or sla_cnt > 0
        )
        near_cpu_limit = curr_cpu >= MAX_CPU_NEAR_THRESHOLD

        if (
            curr_replicas < MAX_REPS
            and severe_latency
            and can_replica_scale_out
            and (
                near_cpu_limit
                and (
                    cpu_usage_max > CPU_CONTAINER_OUT_THRESHOLD
                    or cpu_usage_avg > (CPU_CONTAINER_OUT_THRESHOLD - 5.0)
                )
            )
        ):
            return {
                "action"        : "REP_UP",
                "next_cpu"      : curr_cpu,
                "next_replicas" : curr_replicas + 1,
                "reason"        : "응급 보정: 높은 p95/latency + 최대 컨테이너 CPU 압박 → 컨테이너 추가",
            }
        if curr_cpu < MAX_CPU and can_cpu_scale_out:
            return {
                "action"        : "CPU_UP",
                "next_cpu"      : min(MAX_CPU, curr_cpu + CPU_STEP),
                "next_replicas" : curr_replicas,
                "reason"        : f"응급 보정: lat/p95/cpu 기준 초과 → CPU 증가",
            }
        if (
            curr_replicas < MAX_REPS
            and severe_latency
            and can_replica_scale_out
            and (
                cpu_usage_max > CPU_CONTAINER_OUT_THRESHOLD
                or cpu_usage_avg > (CPU_CONTAINER_OUT_THRESHOLD - 5.0)
                or p95_lat > (P95_REPLICA_UP_MS + 200.0)
            )
        ):
            return {
                "action"        : "REP_UP",
                "next_cpu"      : curr_cpu,
                "next_replicas" : curr_replicas + 1,
                "reason"        : "응급 보정: CPU 증설 후에도 높은 p95/latency 지속 → 컨테이너 추가",
            }

    if underloaded and can_scale_in:
        if curr_replicas > pred_replicas and easy_count >= REPLICA_SCALE_IN_CONFIRMATIONS:
            return {
                "action"        : "REP_DOWN",
                "next_cpu"      : curr_cpu,
                "next_replicas" : max(curr_replicas - 1, pred_replicas, MIN_REPS),
                "reason"        : f"여유 상태 지속 → 예측 replica({pred_replicas})로 복귀",
            }
        if curr_cpu > pred_cpu and easy_count >= CPU_SCALE_IN_CONFIRMATIONS:
            next_cpu = max(curr_cpu - CPU_STEP, pred_cpu, MIN_CPU)
            return {
                "action"        : "CPU_DOWN",
                "next_cpu"      : next_cpu,
                "next_replicas" : curr_replicas,
                "reason"        : f"여유 상태 지속 → 예측 CPU({pred_cpu})로 복귀",
            }

    return {
        "action"        : "HOLD",
        "next_cpu"      : curr_cpu,
        "next_replicas" : curr_replicas,
        "reason"        : "정상 범위 유지",
    }


# =========================
# CSV 로깅
# =========================

def init_csv():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "elapsed_sec",
            "pred_cpu", "pred_replicas",
            "curr_cpu", "curr_replicas",
            "action",
            "cpu_usage_pct",
            "avg_latency_ms", "p95_latency_ms",
            "fail_count", "sla_violation_count", "current_rps",
            "loadgen_avg_latency_ms", "loadgen_p95_latency_ms",
            "loadgen_fail_count", "loadgen_sla_violation_count",
            "reason",
        ])


def append_csv(row: list):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# =========================
# 보정 레이어 메인 루프
# =========================

class HybridController:
    """
    PredictiveAllocator의 AllocationState를 공유 객체로 받아
    실시간 metrics를 보고 편차가 있으면 CPU/컨테이너를 보정한다.
    """

    NO_TRAFFIC_LIMIT = 5

    def __init__(self, state):
        self.state            = state
        self.curr_cpu         = state.current_cpu
        self.curr_replicas    = state.current_replicas
        self.last_scale_out_time = {"cpu": 0.0, "replica": 0.0}
        self.last_scale_in_time = 0.0
        self.start_time       = time.time()
        self.no_traffic_count = 0
        self.easy_count       = 0

    def run(self):
        init_csv()
        print("[HybridController] 보정 루프 시작")

        while True:
            now     = time.time()
            elapsed = now - self.start_time

            pred_cpu      = self.state.planned_cpu
            pred_replicas = self.state.planned_replicas

            # PredictiveAllocator가 시간축에 맞춰 자원을 갱신했다면
            # 보정 레이어도 그 기준 상태를 즉시 따라가도록 동기화한다.
            if (self.state.current_cpu != self.curr_cpu) or (self.state.current_replicas != self.curr_replicas):
                self.curr_cpu = self.state.current_cpu
                self.curr_replicas = self.state.current_replicas

            metrics = fetch_metrics_for_replicas(self.curr_replicas)
            if not metrics:
                time.sleep(CHECK_INTERVAL)
                continue

            loadgen_metrics = read_recent_loadgen_metrics()
            if loadgen_metrics:
                metrics["avg_latency_ms"] = max(metrics.get("avg_latency_ms", 0.0), loadgen_metrics.get("avg_latency_ms", 0.0))
                metrics["p95_latency_ms"] = max(metrics.get("p95_latency_ms", 0.0), loadgen_metrics.get("p95_latency_ms", 0.0))
                metrics["fail_count"] = max(metrics.get("fail_count", 0), loadgen_metrics.get("fail_count", 0))
                metrics["sla_violation_count"] = loadgen_metrics.get("sla_violation_count", 0)
                metrics["current_rps"] = max(metrics.get("current_rps", 0.0), loadgen_metrics.get("current_rps", 0.0))
            else:
                metrics["sla_violation_count"] = metrics.get("sla_violation_count", 0)

            if metrics.get("request_count", 0) == 0:
                self.no_traffic_count += 1
            else:
                self.no_traffic_count = 0

            if self.no_traffic_count >= self.NO_TRAFFIC_LIMIT:
                print("[HybridController] 트래픽 없음 → 루프 종료")
                break

            try:
                cpu_usage_avg, cpu_usage_max = get_cluster_cpu_usage(self.curr_replicas)
            except Exception:
                cpu_usage_avg, cpu_usage_max = 0.0, 0.0

            avg_lat = metrics.get("avg_latency_ms", 0.0)
            p95_lat = metrics.get("p95_latency_ms", 0.0)
            fail_cnt = metrics.get("fail_count", 0)
            sla_cnt = metrics.get("sla_violation_count", 0)
            cur_rps = metrics.get("current_rps", 0.0)

            easy = (
                avg_lat < LATENCY_EASY_MS
                and p95_lat < P95_EASY_MS
                and fail_cnt == 0
                and sla_cnt == 0
                and cpu_usage_avg < CPU_USAGE_EASY
                and cur_rps >= MIN_RPS_SCALE_IN
            )
            if easy:
                self.easy_count += 1
            else:
                self.easy_count = 0

            decision = decide_correction(
                metrics=metrics,
                cpu_usage_avg=cpu_usage_avg,
                cpu_usage_max=cpu_usage_max,
                curr_cpu=self.curr_cpu,
                curr_replicas=self.curr_replicas,
                pred_cpu=pred_cpu,
                pred_replicas=pred_replicas,
                easy_count=self.easy_count,
                now=now,
                last_scale_out_time=self.last_scale_out_time,
                last_scale_in_time=self.last_scale_in_time,
            )

            action    = decision["action"]
            next_cpu  = decision["next_cpu"]
            next_reps = decision["next_replicas"]
            reason    = decision["reason"]

            if action != "HOLD":
                try:
                    with self.state.op_lock:
                        if action in ("CPU_UP", "CPU_DOWN"):
                            update_all_cpu(self.curr_replicas, next_cpu)
                            self.curr_cpu = next_cpu
                        elif action in ("REP_UP", "REP_DOWN"):
                            ensure_replicas(next_reps, self.curr_cpu)
                            self.curr_replicas = next_reps
                    self.state.update_current(self.curr_cpu, self.curr_replicas)
                    self.state.override_until = now + OVERRIDE_HOLD_SEC
                    if action == "CPU_UP":
                        self.last_scale_out_time["cpu"] = now
                        self.easy_count = 0
                        self.state.scale_down_hold_until = now + SCALE_DOWN_HOLD_SEC
                    elif action == "REP_UP":
                        self.last_scale_out_time["replica"] = now
                        self.easy_count = 0
                        self.state.scale_down_hold_until = now + SCALE_DOWN_HOLD_SEC
                    elif action in ("CPU_DOWN", "REP_DOWN"):
                        self.last_scale_in_time = now
                    print(
                        f"  [CORRECTION] {action} | "
                        f"cpu={self.curr_cpu} | replicas={self.curr_replicas} | {reason}"
                    )
                except Exception as e:
                    print(f"  [ERROR] 보정 실패 ({action}, cpu={next_cpu}, replicas={next_reps}): {e}")
                    action = "ERROR"

            print(
                f"[CTRL] t={elapsed:.1f}s | "
                f"pred=({pred_cpu}/{pred_replicas}) | "
                f"curr=({self.curr_cpu}/{self.curr_replicas}) | "
                f"action={action} | "
                f"lat={metrics.get('avg_latency_ms', 0):.0f}ms | "
                f"cpu(avg/max)={cpu_usage_avg:.0f}/{cpu_usage_max:.0f}% | "
                f"fail={metrics.get('fail_count', 0)} | "
                f"sla={metrics.get('sla_violation_count', 0)}"
            )

            append_csv([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(elapsed, 2),
                pred_cpu, pred_replicas,
                self.curr_cpu, self.curr_replicas,
                action,
                round(cpu_usage_avg, 2),
                metrics.get("avg_latency_ms", 0.0),
                metrics.get("p95_latency_ms", 0.0),
                metrics.get("fail_count", 0),
                metrics.get("sla_violation_count", 0),
                metrics.get("current_rps", 0.0),
                loadgen_metrics.get("avg_latency_ms", 0.0) if loadgen_metrics else 0.0,
                loadgen_metrics.get("p95_latency_ms", 0.0) if loadgen_metrics else 0.0,
                loadgen_metrics.get("fail_count", 0) if loadgen_metrics else 0,
                loadgen_metrics.get("sla_violation_count", 0) if loadgen_metrics else 0,
                reason,
            ])

            time.sleep(CHECK_INTERVAL)

        print("[HybridController] 종료")
