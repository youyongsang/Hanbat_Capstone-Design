import csv
import time
import subprocess
from pathlib import Path

import requests

# =========================
# 기본 설정
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_NAME = "reactive-server"

# 컨테이너 이름 / 포트 규칙
CONTAINER_PREFIX = "app_server_"
BASE_PORT = 8001

# 대표 metrics는 1번 컨테이너 기준으로 확인
# (단순 실험용)
METRICS_CONTAINER_INDEX = 1

# metrics API 주소
def get_metrics_url(container_index: int) -> str:
    port = BASE_PORT + (container_index - 1)
    return f"http://localhost:{port}/metrics?window_sec=5"


# =========================
# CPU / 컨테이너 스케일링 설정
# =========================

INITIAL_CPU = 0.5
MIN_CPU = 0.5
MAX_CPU = 6.0
CPU_STEP = 0.5

MIN_REPLICAS = 1
MAX_REPLICAS = 8

# -------------------------
# Scale Out 기준
# -------------------------
# Load Generator의 SLA 기준은 1000ms이다.
# Reactive가 80/200ms처럼 너무 이른 구간에서 반응하면 SLA 위반이 거의
# 드러나지 않는다. 반대로 SLA 직전에서만 반응하면 피크 구간에서도
# replica가 충분히 늘지 못해 Reactive를 과도하게 불리하게 만든다.
# 따라서 SLA보다 낮지만 충분히 성능 저하가 관측된 500/750ms 지점에서
# scale-out을 시작하도록 조정한다.
LATENCY_UP_THRESHOLD = 500.0
P95_UP_THRESHOLD = 750.0
FAIL_UP_THRESHOLD = 1

# CPU를 먼저 올리는 기준
CPU_UP_USAGE_THRESHOLD = 60.0

# CPU가 이미 최대고, 실제 CPU 사용률도 높으면 컨테이너 증가 고려
CPU_CONTAINER_OUT_THRESHOLD = 70.0

# -------------------------
# Scale In 기준
# -------------------------
LATENCY_DOWN_THRESHOLD = 250.0
P95_DOWN_THRESHOLD = 500.0
CPU_DOWN_USAGE_THRESHOLD = 30.0
MIN_RPS_FOR_SCALE_IN = 5.0

# -------------------------
# 제어 주기
# -------------------------
CHECK_INTERVAL = 2
COOLDOWN_SEC = 5
NO_TRAFFIC_LIMIT = 5

OUTPUT_CSV = BASE_DIR / "data/output/hybrid_reactive_result.csv"
LOADGEN_CSV = BASE_DIR / "data/output/loadgen_result.csv"

# Load Generator는 클라이언트 end-to-end latency를 기록한다.
# Controller 시작 이전의 이전 실험 CSV는 무시하고, 최신 행만 보정 판단에 사용한다.
LOADGEN_METRIC_STALE_SEC = 10
SLA_VIOLATION_UP_THRESHOLD = 1


# =========================
# Docker 헬퍼
# =========================

# 터미널 명령 실행 헬퍼
def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess: 
    return subprocess.run(cmd, capture_output=True, text=True)

# 요청 실패 시 예외 발생
def require_success(result: subprocess.CompletedProcess, context: str):
    if result.returncode != 0:
        raise RuntimeError(
            f"{context} failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
        )

# 현재 컨테이너 네임 반환
def container_name(index: int) -> str:
    return f"{CONTAINER_PREFIX}{index}"

# 현재 컨테이너 포트 반환
def host_port(index: int) -> int:
    return BASE_PORT + (index - 1)

# 모든 컨테이너 이름 리스트 반환
def list_all_container_names() -> list[str]:
    result = run_cmd(["docker", "ps", "-a", "--format", "{{.Names}}"])
    # docker ps -a --format "{{.Names}}" 명령이 실패하면 예외 발생
    # .Names는 docker ps의 Go 템플릿으로, 컨테이너 이름만 출력하도록 함.
    require_success(result, "docker ps -a")
    return result.stdout.splitlines()

# 컨테이너 존재 여부 확인
def container_exists(name: str) -> bool:
    return name in list_all_container_names()

# 현재 실행 중인 컨테이너 이름 리스트 반환
def running_container_names() -> list[str]:
    result = run_cmd(["docker", "ps", "--format", "{{.Names}}"])
    require_success(result, "docker ps")
    return result.stdout.splitlines()


# =========================
# 컨테이너 시작 / 제거
# =========================

def start_container(index: int, cpu_value: float):
    """
    지정 인덱스의 컨테이너를 시작한다.
    이미 있으면 시작하지 않음.
    """
    name = container_name(index)
    port = host_port(index)

    if container_exists(name):
        # 이미 있으면 그냥 CPU만 맞춤
        set_container_cpu(name, cpu_value)
        return

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        f"--cpus={cpu_value}",
        "-p", f"{port}:8000",
        IMAGE_NAME
    ]
    result = run_cmd(cmd)
    require_success(result, f"docker run {name}")
    print(f"[START] {name} on port {port} with cpu={cpu_value}")


def remove_container(index: int):
    """
    지정 인덱스 컨테이너 제거
    """
    name = container_name(index)
    if not container_exists(name):
        return

    result = run_cmd(["docker", "rm", "-f", name])
    require_success(result, f"docker rm -f {name}")
    print(f"[REMOVE] {name}")


def ensure_replicas(target_replicas: int, cpu_value: float):
    """
    target_replicas 수만큼 컨테이너를 맞춘다.
    새로 띄우는 컨테이너는 cpu_value로 시작한다.
    """
    for i in range(1, target_replicas + 1):
        start_container(i, cpu_value)

    for i in range(MAX_REPLICAS, target_replicas, -1):
        remove_container(i)


# =========================
# CPU 제어
# =========================

def set_container_cpu(container_name_str: str, cpu_value: float):
    cmd = ["docker", "update", f"--cpus={cpu_value}", container_name_str]
    result = run_cmd(cmd)
    require_success(result, f"docker update {container_name_str}")


def set_all_running_containers_cpu(replicas: int, cpu_value: float):
    """
    현재 활성 replica 수만큼 CPU를 동일하게 맞춘다.
    """
    for i in range(1, replicas + 1):
        name = container_name(i)
        if container_exists(name):
            set_container_cpu(name, cpu_value)


# =========================
# Metrics 수집
# =========================

def fetch_metrics(url: str) -> dict:
    response = requests.get(url, timeout=3)
    response.raise_for_status()
    return response.json()


def fetch_latest_loadgen_metrics(csv_path: Path, min_mtime: float) -> dict:
    """
    Load Generator 결과 CSV의 최신 행을 읽어 클라이언트 기준 지표를 반환한다.
    서버 /metrics는 FastAPI handler 내부 처리시간만 보므로, 대기열/네트워크를 포함한
    end-to-end SLA 위반 판단에는 loadgen_result.csv의 값이 더 직접적이다.
    """
    if not csv_path.exists():
        return {}

    stat = csv_path.stat()
    now = time.time()
    if stat.st_mtime < min_mtime or now - stat.st_mtime > LOADGEN_METRIC_STALE_SEC:
        return {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    if not rows:
        return {}

    row = rows[-1]

    def to_float(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except ValueError:
            return default

    return {
        "client_time_sec": to_float("time_sec"),
        "client_target_rps": to_float("target_rps"),
        "client_actual_sent": to_float("actual_sent"),
        "client_success_count": to_float("success_count"),
        "client_fail_count": to_float("fail_count"),
        "client_sla_violation_count": to_float("sla_violation_count"),
        "client_avg_latency_ms": to_float("avg_latency_ms"),
        "client_p95_latency_ms": to_float("p95_latency_ms"),
    }


def merge_server_and_client_metrics(server_metrics: dict, client_metrics: dict) -> dict:
    """
    Controller 판단에는 서버 내부 처리시간과 클라이언트 end-to-end 시간을 함께 사용한다.
    latency는 더 큰 값을 사용해 사용자가 실제로 겪는 지연을 놓치지 않도록 한다.
    """
    merged = dict(server_metrics)

    if not client_metrics:
        merged["sla_violation_count"] = 0
        return merged

    server_avg = float(server_metrics.get("avg_latency_ms", 0.0))
    server_p95 = float(server_metrics.get("p95_latency_ms", 0.0))
    server_fail = float(server_metrics.get("fail_count", 0.0))
    server_rps = float(server_metrics.get("current_rps", 0.0))

    client_avg = client_metrics["client_avg_latency_ms"]
    client_p95 = client_metrics["client_p95_latency_ms"]
    client_fail = client_metrics["client_fail_count"]
    client_rps = client_metrics["client_actual_sent"]

    merged.update(client_metrics)
    merged["server_avg_latency_ms"] = server_avg
    merged["server_p95_latency_ms"] = server_p95
    merged["avg_latency_ms"] = max(server_avg, client_avg)
    merged["p95_latency_ms"] = max(server_p95, client_p95)
    merged["fail_count"] = max(server_fail, client_fail)
    merged["current_rps"] = max(server_rps, client_rps)
    merged["sla_violation_count"] = client_metrics["client_sla_violation_count"]
    return merged


# =========================
# Docker stats 기반 CPU 사용률 조회
# =========================

def get_container_cpu_usage_percent(name: str) -> float:
    """
    docker stats --no-stream 으로 현재 컨테이너 CPU 사용률 조회

    반환 예: 83.52
    """
    result = run_cmd([
        "docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", name
    ])
    require_success(result, f"docker stats {name}")

    value = result.stdout.strip().replace("%", "").strip()
    if not value:
        return 0.0

    try:
        return float(value)
    except ValueError:
        return 0.0


# =========================
# 의사결정 로직
# =========================

def decide_hybrid_action(metrics: dict, current_cpu: float, current_replicas: int, cpu_usage: float):
    """
    2단계 Reactive 정책

    1) 성능 저하 시 먼저 CPU scale out
    2) CPU가 최대인데도 성능이 나쁘고 CPU 사용률도 높으면 컨테이너 scale out
    3) 여유가 많으면 먼저 컨테이너 scale in
    4) 컨테이너가 최소일 때만 CPU scale in
    """
    avg_latency = metrics.get("avg_latency_ms", 0.0)
    p95_latency = metrics.get("p95_latency_ms", 0.0)
    fail_count = metrics.get("fail_count", 0)
    sla_violation_count = metrics.get("sla_violation_count", 0)
    current_rps = metrics.get("current_rps", 0.0)

    overloaded = (
        avg_latency > LATENCY_UP_THRESHOLD
        or p95_latency > P95_UP_THRESHOLD
        or fail_count >= FAIL_UP_THRESHOLD
        or sla_violation_count >= SLA_VIOLATION_UP_THRESHOLD
    )

    underloaded = (
        avg_latency < LATENCY_DOWN_THRESHOLD
        and p95_latency < P95_DOWN_THRESHOLD
        and fail_count == 0
        and sla_violation_count == 0
        and cpu_usage < CPU_DOWN_USAGE_THRESHOLD
        and current_rps >= MIN_RPS_FOR_SCALE_IN
    )

    # -------------------------
    # 과부하: CPU 먼저 확장
    # -------------------------
    if overloaded:
        if (
            current_cpu < MAX_CPU
            and (
                cpu_usage >= CPU_UP_USAGE_THRESHOLD
                or sla_violation_count >= SLA_VIOLATION_UP_THRESHOLD
                or p95_latency > P95_UP_THRESHOLD
            )
        ):
            return {
                "action": "CPU_SCALE_OUT",
                "next_cpu": min(MAX_CPU, current_cpu + CPU_STEP),
                "next_replicas": current_replicas,
                "reason": "end-to-end 성능 저하 감지 → CPU 먼저 증설"
            }

        # CPU가 이미 최대인데도 여전히 과부하고 CPU usage 높으면 컨테이너 증가
        if (
            current_cpu >= MAX_CPU
            and (
                cpu_usage >= CPU_CONTAINER_OUT_THRESHOLD
                or sla_violation_count >= SLA_VIOLATION_UP_THRESHOLD
            )
            and current_replicas < MAX_REPLICAS
        ):
            return {
                "action": "CONTAINER_SCALE_OUT",
                "next_cpu": current_cpu,
                "next_replicas": current_replicas + 1,
                "reason": "CPU 최대 + end-to-end 성능 저하 → 컨테이너 증설"
            }

    # -------------------------
    # 여유 상태: 컨테이너 먼저 축소
    # -------------------------
    if underloaded:
        if current_replicas > MIN_REPLICAS:
            return {
                "action": "CONTAINER_SCALE_IN",
                "next_cpu": current_cpu,
                "next_replicas": current_replicas - 1,
                "reason": "자원 여유 많음 → 먼저 컨테이너 축소"
            }

        if current_cpu > MIN_CPU:
            return {
                "action": "CPU_SCALE_IN",
                "next_cpu": max(MIN_CPU, current_cpu - CPU_STEP),
                "next_replicas": current_replicas,
                "reason": "컨테이너 최소 상태에서 CPU 축소"
            }

    return {
        "action": "HOLD",
        "next_cpu": current_cpu,
        "next_replicas": current_replicas,
        "reason": "현재 상태 유지"
    }


# =========================
# CSV 로깅
# =========================

def init_csv(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "elapsed_sec",
            "current_cpu",
            "current_replicas",
            "action",
            "cpu_usage_percent",
            "request_count",
            "success_count",
            "fail_count",
            "avg_latency_ms",
            "p95_latency_ms",
            "sla_violation_count",
            "client_avg_latency_ms",
            "client_p95_latency_ms",
            "client_sla_violation_count",
            "current_rps",
            "reason"
        ])


def append_csv(output_path: Path, row: list):
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =========================
# 메인 루프
# =========================

def main():
    init_csv(OUTPUT_CSV)

    current_cpu = INITIAL_CPU
    current_replicas = MIN_REPLICAS

    last_action_time = 0.0
    start_time = time.time()
    no_traffic_count = 0

    # 시작 시 최소 1개 컨테이너 준비
    print(f"[START] replicas={current_replicas}, cpu={current_cpu}")
    ensure_replicas(current_replicas, current_cpu)

    while True:
        now = time.time()
        elapsed = now - start_time

        metrics_url = get_metrics_url(METRICS_CONTAINER_INDEX)

        # metrics 수집
        try:
            server_metrics = fetch_metrics(metrics_url)
        except Exception as e:
            print(f"[WARN] Failed to fetch metrics: {e}")
            time.sleep(CHECK_INTERVAL)
            continue

        client_metrics = fetch_latest_loadgen_metrics(LOADGEN_CSV, min_mtime=start_time)
        metrics = merge_server_and_client_metrics(server_metrics, client_metrics)

        # 대표 컨테이너 CPU usage 수집
        try:
            cpu_usage = get_container_cpu_usage_percent(container_name(METRICS_CONTAINER_INDEX))
        except Exception as e:
            print(f"[WARN] Failed to fetch cpu usage: {e}")
            cpu_usage = 0.0

        # 트래픽 종료 감지
        if metrics.get("request_count", 0) == 0:
            no_traffic_count += 1
        else:
            no_traffic_count = 0

        if no_traffic_count >= NO_TRAFFIC_LIMIT:
            print("[END] No traffic detected repeatedly, stopping controller")
            break

        cooldown_active = (now - last_action_time) < COOLDOWN_SEC

        decision = {
            "action": "HOLD",
            "next_cpu": current_cpu,
            "next_replicas": current_replicas,
            "reason": "cooldown 또는 유지"
        }

        if not cooldown_active:
            decision = decide_hybrid_action(
                metrics=metrics,
                current_cpu=current_cpu,
                current_replicas=current_replicas,
                cpu_usage=cpu_usage
            )

        action = decision["action"]
        next_cpu = decision["next_cpu"]
        next_replicas = decision["next_replicas"]
        reason = decision["reason"]

        # 액션 수행
        try:
            if action == "CPU_SCALE_OUT" or action == "CPU_SCALE_IN":
                set_all_running_containers_cpu(current_replicas, next_cpu)
                print(
                    f"[ACTION] {action} | cpu {current_cpu} -> {next_cpu} | "
                    f"replicas={current_replicas} | reason={reason}"
                )
                current_cpu = next_cpu
                last_action_time = time.time()

            elif action == "CONTAINER_SCALE_OUT":
                ensure_replicas(next_replicas, current_cpu)
                print(
                    f"[ACTION] {action} | replicas {current_replicas} -> {next_replicas} | "
                    f"cpu={current_cpu} | reason={reason}"
                )
                current_replicas = next_replicas
                last_action_time = time.time()

            elif action == "CONTAINER_SCALE_IN":
                ensure_replicas(next_replicas, current_cpu)
                print(
                    f"[ACTION] {action} | replicas {current_replicas} -> {next_replicas} | "
                    f"cpu={current_cpu} | reason={reason}"
                )
                current_replicas = next_replicas
                last_action_time = time.time()

        except Exception as e:
            print(f"[ERROR] Failed to apply action {action}: {e}")
            action = "HOLD"
            reason = f"실행 실패: {e}"

        # 로그 저장
        append_csv(OUTPUT_CSV, [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round(elapsed, 2),
            current_cpu,
            current_replicas,
            action,
            round(cpu_usage, 2),
            metrics.get("request_count", 0),
            metrics.get("success_count", 0),
            metrics.get("fail_count", 0),
            metrics.get("avg_latency_ms", 0.0),
            metrics.get("p95_latency_ms", 0.0),
            metrics.get("sla_violation_count", 0),
            metrics.get("client_avg_latency_ms", 0.0),
            metrics.get("client_p95_latency_ms", 0.0),
            metrics.get("client_sla_violation_count", 0),
            metrics.get("current_rps", 0.0),
            reason
        ])

        print(
            f"[STATUS] t={elapsed:.1f}s | cpu={current_cpu} | replicas={current_replicas} | "
            f"action={action} | cpu_usage={cpu_usage:.2f}% | "
            f"req={metrics.get('request_count', 0)} | "
            f"succ={metrics.get('success_count', 0)} | "
            f"fail={metrics.get('fail_count', 0)} | "
            f"avg={metrics.get('avg_latency_ms', 0.0)}ms | "
            f"p95={metrics.get('p95_latency_ms', 0.0)}ms | "
            f"sla={metrics.get('sla_violation_count', 0)} | "
            f"client_avg={metrics.get('client_avg_latency_ms', 0.0)}ms | "
            f"client_p95={metrics.get('client_p95_latency_ms', 0.0)}ms | "
            f"rps={metrics.get('current_rps', 0.0)} | "
            f"reason={reason}"
        )

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
