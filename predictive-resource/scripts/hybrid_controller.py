import csv
import time
import subprocess
from pathlib import Path

import requests

# =========================
# 경로 설정
# =========================

BASE_DIR   = Path(__file__).resolve().parent.parent   # predictive-resource/
OUTPUT_CSV = BASE_DIR / "data" / "output" / "hybrid_correction_log.csv"

# =========================
# 보정 정책 파라미터
# =========================

METRICS_CONTAINER_INDEX = 1
BASE_PORT               = 8001

def get_metrics_url(idx: int = METRICS_CONTAINER_INDEX) -> str:
    return f"http://localhost:{BASE_PORT + (idx - 1)}/metrics?window_sec=5"


LATENCY_WARN_MS  = 150.0
LATENCY_CRIT_MS  = 300.0
P95_WARN_MS      = 400.0
CPU_USAGE_WARN   = 80.0
FAIL_THRESHOLD   = 1

LATENCY_EASY_MS  = 50.0
CPU_USAGE_EASY   = 30.0
MIN_RPS_SCALE_IN = 5.0

CPU_STEP  = 0.5
MIN_CPU   = 2.0
MAX_CPU   = 6.0
MIN_REPS  = 3
MAX_REPS  = 8

CHECK_INTERVAL = 2
COOLDOWN_SEC   = 5

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


# =========================
# Metrics 수집
# =========================

def fetch_metrics() -> dict:
    try:
        resp = requests.get(get_metrics_url(), timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


# =========================
# 보정 의사결정
# =========================

def decide_correction(
    metrics: dict,
    cpu_usage: float,
    curr_cpu: float,
    curr_replicas: int,
    pred_cpu: float,
    pred_replicas: int,
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
    cur_rps  = metrics.get("current_rps", 0.0)

    overloaded = (
        avg_lat  > LATENCY_WARN_MS
        or p95_lat  > P95_WARN_MS
        or fail_cnt >= FAIL_THRESHOLD
        or cpu_usage > CPU_USAGE_WARN
    )
    critical = avg_lat > LATENCY_CRIT_MS or fail_cnt >= FAIL_THRESHOLD
    underloaded = (
        avg_lat  < LATENCY_EASY_MS
        and cpu_usage < CPU_USAGE_EASY
        and fail_cnt == 0
        and cur_rps  >= MIN_RPS_SCALE_IN
    )

    if overloaded:
        if curr_cpu < MAX_CPU:
            return {
                "action"        : "CPU_UP",
                "next_cpu"      : min(MAX_CPU, curr_cpu + CPU_STEP),
                "next_replicas" : curr_replicas,
                "reason"        : f"성능 저하 (lat={avg_lat:.0f}ms, cpu={cpu_usage:.0f}%) → CPU 증가",
            }
        if critical and curr_replicas < MAX_REPS:
            return {
                "action"        : "REP_UP",
                "next_cpu"      : curr_cpu,
                "next_replicas" : curr_replicas + 1,
                "reason"        : "CPU 한계 + 심각 과부하 → 컨테이너 추가",
            }

    if underloaded:
        if curr_replicas > pred_replicas:
            return {
                "action"        : "REP_DOWN",
                "next_cpu"      : curr_cpu,
                "next_replicas" : max(pred_replicas, MIN_REPS),
                "reason"        : f"여유 상태 → 예측 replica({pred_replicas})로 복귀",
            }
        if curr_cpu > pred_cpu:
            return {
                "action"        : "CPU_DOWN",
                "next_cpu"      : max(pred_cpu, MIN_CPU),
                "next_replicas" : curr_replicas,
                "reason"        : f"여유 상태 → 예측 CPU({pred_cpu})로 복귀",
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
            "fail_count", "current_rps",
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
        self.last_action_time = 0.0
        self.start_time       = time.time()
        self.no_traffic_count = 0

    def run(self):
        init_csv()
        print("[HybridController] 보정 루프 시작")

        while True:
            now     = time.time()
            elapsed = now - self.start_time

            pred_cpu      = self.state.current_cpu
            pred_replicas = self.state.current_replicas

            # PredictiveAllocator가 시간축에 맞춰 자원을 갱신했다면
            # 보정 레이어도 그 기준 상태를 즉시 따라가도록 동기화한다.
            if (pred_cpu != self.curr_cpu) or (pred_replicas != self.curr_replicas):
                self.curr_cpu = pred_cpu
                self.curr_replicas = pred_replicas

            metrics = fetch_metrics()
            if not metrics:
                time.sleep(CHECK_INTERVAL)
                continue

            if metrics.get("request_count", 0) == 0:
                self.no_traffic_count += 1
            else:
                self.no_traffic_count = 0

            if self.no_traffic_count >= self.NO_TRAFFIC_LIMIT:
                print("[HybridController] 트래픽 없음 → 루프 종료")
                break

            try:
                cpu_usage = get_cpu_usage(container_name(METRICS_CONTAINER_INDEX))
            except Exception:
                cpu_usage = 0.0

            in_cooldown = (now - self.last_action_time) < COOLDOWN_SEC

            if in_cooldown:
                decision = {
                    "action"        : "HOLD",
                    "next_cpu"      : self.curr_cpu,
                    "next_replicas" : self.curr_replicas,
                    "reason"        : "쿨다운 대기",
                }
            else:
                decision = decide_correction(
                    metrics       = metrics,
                    cpu_usage     = cpu_usage,
                    curr_cpu      = self.curr_cpu,
                    curr_replicas = self.curr_replicas,
                    pred_cpu      = pred_cpu,
                    pred_replicas = pred_replicas,
                )

            action    = decision["action"]
            next_cpu  = decision["next_cpu"]
            next_reps = decision["next_replicas"]
            reason    = decision["reason"]

            if action != "HOLD":
                try:
                    if action in ("CPU_UP", "CPU_DOWN"):
                        update_all_cpu(self.curr_replicas, next_cpu)
                        self.curr_cpu = next_cpu
                    elif action in ("REP_UP", "REP_DOWN"):
                        ensure_replicas(next_reps, self.curr_cpu)
                        self.curr_replicas = next_reps
                    self.last_action_time = now
                    print(
                        f"  [CORRECTION] {action} | "
                        f"cpu={self.curr_cpu} | replicas={self.curr_replicas} | {reason}"
                    )
                except Exception as e:
                    print(f"  [ERROR] 보정 실패: {e}")
                    action = "ERROR"

            print(
                f"[CTRL] t={elapsed:.1f}s | "
                f"pred=({pred_cpu}/{pred_replicas}) | "
                f"curr=({self.curr_cpu}/{self.curr_replicas}) | "
                f"action={action} | "
                f"lat={metrics.get('avg_latency_ms', 0):.0f}ms | "
                f"cpu={cpu_usage:.0f}% | "
                f"fail={metrics.get('fail_count', 0)}"
            )

            append_csv([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(elapsed, 2),
                pred_cpu, pred_replicas,
                self.curr_cpu, self.curr_replicas,
                action,
                round(cpu_usage, 2),
                metrics.get("avg_latency_ms", 0.0),
                metrics.get("p95_latency_ms", 0.0),
                metrics.get("fail_count", 0),
                metrics.get("current_rps", 0.0),
                reason,
            ])

            time.sleep(CHECK_INTERVAL)

        print("[HybridController] 종료")
