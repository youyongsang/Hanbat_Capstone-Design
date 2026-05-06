import csv
import time
import subprocess
from pathlib import Path

import requests

# =========================
# 설정값
# =========================

# FastAPI metrics API
SERVER_METRICS_URL = "http://localhost:8000/metrics?window_sec=5"

# Docker 컨테이너 이름
CONTAINER_NAME = "busy_antonelli"

# -------------------------
# CPU 스케일링 설정
# -------------------------
# 시작 CPU
INITIAL_CPU = 0.5

# 최소 / 최대 CPU
MIN_CPU = 0.5
MAX_CPU = 6.0

# 한 번 scale 시 증가/감소 폭
CPU_STEP = 0.5

# -------------------------
# Reactive 임계값 설정
# -------------------------
# Load Generator의 SLA 기준은 1000ms이다.
# 너무 낮은 threshold(예: 80/120ms)를 쓰면 SLA 위반이 발생하기 전에
# Reactive가 과도하게 빠르게 반응하여 사후 대응 한계가 드러나지 않는다.
# 반대로 SLA 직전에서만 반응하면 Reactive를 과도하게 불리하게 만든다.
# 따라서 단일 CPU Reactive 실험도 1000ms SLA보다 낮은 500/750ms 지점에서
# scale-out하도록 맞춘다.

# 평균 latency가 이 값보다 크면 scale out 후보
LATENCY_UP_THRESHOLD = 500.0

# 평균 latency가 이 값보다 작고 안정적이면 scale in 후보
LATENCY_DOWN_THRESHOLD = 250.0

# p95 latency가 이 값보다 크면 scale out 후보
P95_UP_THRESHOLD = 750.0

# 실패가 1건 이상이면 즉시 scale out 후보
FAIL_UP_THRESHOLD = 1

# 현재 RPS가 너무 낮은 상태에서는 scale in 판단을 보수적으로 하도록 설정
MIN_RPS_FOR_SCALE_IN = 5.0

# -------------------------
# 제어 주기 설정
# -------------------------
# 상태 체크 주기
CHECK_INTERVAL = 2

# CPU 변경 후 다시 변경하기 전 대기 시간
COOLDOWN_SEC = 4

# 트래픽이 없는 상태가 몇 번 연속이면 종료할지
NO_TRAFFIC_LIMIT = 5

# 결과 저장 파일
OUTPUT_CSV = Path("data/output/reactive_result.csv")


# =========================
# Docker CPU 제어
# =========================
def set_container_cpu(container_name: str, cpu_value: float):
    """
    docker update 명령으로 컨테이너 CPU 제한값 변경
    """
    cmd = ["docker", "update", f"--cpus={cpu_value}", container_name]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Docker update failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
        )


# =========================
# Metrics 읽기
# =========================
def fetch_metrics(url: str):
    """
    FastAPI /metrics 호출
    """
    response = requests.get(url, timeout=3)
    response.raise_for_status()
    return response.json()


# =========================
# Reactive 판단 로직
# =========================
def decide_action(metrics: dict, current_cpu: float):
    """
    현재 metrics 기준으로 SCALE_OUT / SCALE_IN / HOLD 결정
    """
    avg_latency = metrics.get("avg_latency_ms", 0.0)
    p95_latency = metrics.get("p95_latency_ms", 0.0)
    fail_count = metrics.get("fail_count", 0)
    current_rps = metrics.get("current_rps", 0.0)

    # -------------------------
    # Scale Out
    # -------------------------
    # 평균 지연 / p95 지연 / 실패 발생 중 하나라도 조건 만족하면 scale out
    if (
        avg_latency > LATENCY_UP_THRESHOLD
        or p95_latency > P95_UP_THRESHOLD
        or fail_count >= FAIL_UP_THRESHOLD
    ):
        if current_cpu < MAX_CPU:
            return "SCALE_OUT", min(MAX_CPU, current_cpu + CPU_STEP)

    # -------------------------
    # Scale In
    # -------------------------
    # 충분히 안정적이고, 요청도 어느 정도 들어오고 있을 때만 scale in
    elif (
        avg_latency < LATENCY_DOWN_THRESHOLD
        and fail_count == 0
        and current_rps >= MIN_RPS_FOR_SCALE_IN
    ):
        if current_cpu > MIN_CPU:
            return "SCALE_IN", max(MIN_CPU, current_cpu - CPU_STEP)

    return "HOLD", current_cpu


# =========================
# CSV 초기화
# =========================
def init_csv(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "elapsed_sec",
            "current_cpu",
            "action",
            "request_count",
            "success_count",
            "fail_count",
            "avg_latency_ms",
            "p95_latency_ms",
            "current_rps"
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
    last_scale_time = 0.0
    start_time = time.time()
    no_traffic_count = 0

    print(f"[START] Set initial CPU = {current_cpu}")
    set_container_cpu(CONTAINER_NAME, current_cpu)

    while True:
        now = time.time()
        elapsed = now - start_time

        # 1) metrics 수집
        try:
            metrics = fetch_metrics(SERVER_METRICS_URL)
        except Exception as e:
            print(f"[WARN] Failed to fetch metrics: {e}")
            time.sleep(CHECK_INTERVAL)
            continue

        # 2) 트래픽 종료 감지
        if metrics.get("request_count", 0) == 0:
            no_traffic_count += 1
        else:
            no_traffic_count = 0

        if no_traffic_count >= NO_TRAFFIC_LIMIT:
            print("[END] No traffic detected repeatedly, stopping controller")
            break

        action = "HOLD"
        next_cpu = current_cpu

        # 3) cooldown 체크
        cooldown_active = (now - last_scale_time) < COOLDOWN_SEC

        if not cooldown_active:
            action, next_cpu = decide_action(metrics, current_cpu)

            # 4) 자원 변경
            if action in ("SCALE_OUT", "SCALE_IN") and next_cpu != current_cpu:
                try:
                    set_container_cpu(CONTAINER_NAME, next_cpu)
                    print(
                        f"[ACTION] {action} | CPU {current_cpu} -> {next_cpu} | "
                        f"avg={metrics.get('avg_latency_ms', 0.0)}ms | "
                        f"p95={metrics.get('p95_latency_ms', 0.0)}ms | "
                        f"fail={metrics.get('fail_count', 0)} | "
                        f"rps={metrics.get('current_rps', 0.0)}"
                    )
                    current_cpu = next_cpu
                    last_scale_time = time.time()
                except Exception as e:
                    print(f"[ERROR] Failed to apply CPU update: {e}")
                    action = "HOLD"

        # 5) 로그 저장
        append_csv(OUTPUT_CSV, [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round(elapsed, 2),
            current_cpu,
            action,
            metrics.get("request_count", 0),
            metrics.get("success_count", 0),
            metrics.get("fail_count", 0),
            metrics.get("avg_latency_ms", 0.0),
            metrics.get("p95_latency_ms", 0.0),
            metrics.get("current_rps", 0.0),
        ])

        # 6) 콘솔 출력
        print(
            f"[STATUS] t={elapsed:.1f}s | cpu={current_cpu} | action={action} | "
            f"req={metrics.get('request_count', 0)} | "
            f"succ={metrics.get('success_count', 0)} | "
            f"fail={metrics.get('fail_count', 0)} | "
            f"avg={metrics.get('avg_latency_ms', 0.0)}ms | "
            f"p95={metrics.get('p95_latency_ms', 0.0)}ms | "
            f"rps={metrics.get('current_rps', 0.0)}"
        )

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
