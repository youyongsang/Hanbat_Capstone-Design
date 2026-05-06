import asyncio
import csv
import statistics
import subprocess
import sys
import time
from itertools import cycle
from pathlib import Path

import aiohttp


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data/input/sale_event_traffic.csv"
OUTPUT_LOG = BASE_DIR / "data/output/loadgen_result.csv"

# -------------------------
# 실험 파라미터
# -------------------------
REQUEST_TIMEOUT = 5
SLA_LATENCY_MS = 1000
MAX_CONCURRENCY = 1000

# 컨테이너/포트 규칙
CONTAINER_PREFIX = "app_server_"
BASE_PORT = 8001
MAX_REPLICAS = 8


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


# 현재 살아있는 포트 목록 조회 함수
def get_active_ports() -> list[int]:
    """
    현재 실행 중인 app_server_i 컨테이너를 확인하고,
    대응되는 포트(8001, 8002, 8003...) 목록 반환
    """
    result = run_cmd(["docker", "ps", "--format", "{{.Names}}"])

    if result.returncode != 0:
        print("[WARN] docker ps 실패, 포트 조회 불가")
        return []

    ports = []

    for name in result.stdout.splitlines():
        if name.startswith(CONTAINER_PREFIX):
            try:
                idx = int(name.split("_")[-1])
                if 1 <= idx <= MAX_REPLICAS:
                    ports.append(BASE_PORT + (idx - 1))
            except ValueError:
                continue

    ports.sort()
    return ports


def build_round_robin_urls(active_ports: list[int], request_count: int) -> list[str]:
    """
    활성 포트 목록을 기반으로 round-robin 순서의 URL 리스트 생성
    예:
    active_ports = [8001, 8002, 8003], request_count = 8
    -> 8001,8002,8003,8001,8002,8003,8001,8002
    """
    if not active_ports or request_count <= 0:
        return []

    port_cycle = cycle(active_ports)
    return [f"http://localhost:{next(port_cycle)}/work" for _ in range(request_count)]


async def send_one_request(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore): # 비동기 함수
    """
    단일 요청 전송

    구분 기준:
    - ok: HTTP 요청 자체가 성공했는가 (2xx~3xx)
    - sla_violation: 성공은 했지만 SLA 시간 초과인가
    - fail: 이 함수 안에서는 ok=False 인 경우가 나중에 fail_count로 집계됨
    """
    async with sem: # 비동기 작업을 시작하고 종료하면 자동 close되는 컨텍스트 매니저
        start = time.perf_counter() # 요청 보내기 직전 시간 기록
        try:
            async with session.get(url) as resp:
                await resp.read() # http 응답 전체 읽기 (실제 네트워크 왕복 시간 측정 위해)
                latency_ms = (time.perf_counter() - start) * 1000.0 # 읽는데 걸린 시간 계산

                http_ok = 200 <= resp.status < 400

                # 진짜 성공 여부: HTTP 정상 응답이면 성공
                ok = http_ok

                # SLA 위반 여부: 응답은 성공했지만 너무 느린 경우, true false전달
                sla_violation = http_ok and (latency_ms > SLA_LATENCY_MS) 

                return {
                    "ok": ok,
                    "status": resp.status, # 200, 500 등 HTTP 상태 코드
                    "latency_ms": latency_ms,
                    "sla_violation": sla_violation,
                    "url": url,
                }

        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "status": "EXCEPTION",
                "latency_ms": latency_ms,
                "sla_violation": False,
                "url": url,
            }

# 단순히 p95 계산을 위해 백분위수 함수 구현
def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    k = (len(values) - 1) * p # 배열에서 p 퍼센트 위치에 해당하는 인덱스 계산
    f = int(k) # k의 정수 부분 (아래값)
    c = min(f + 1, len(values) - 1) # 보간을 위한 위값 인덱스, 배열 범위 넘어가지 않도록 조정

    if f == c:
        return values[f]

    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def load_schedule(csv_path: Path):
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"time_sec", "target_rps", "scenario", "phase"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV 컬럼이 부족합니다. 필요 컬럼: {required}")

        for row in reader:
            rows.append({
                "time_sec": int(row["time_sec"]),
                "target_rps": int(float(row["target_rps"])),
                "scenario": row["scenario"],
                "phase": row["phase"],
            })

    return rows


async def run_schedule(csv_path: Path, output_log: Path):
    schedule = load_schedule(csv_path)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    headers = {"User-Agent": "capstone-loadgen/1.0"}

    output_log.parent.mkdir(parents=True, exist_ok=True)

    with open(output_log, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow([
            "time_sec",
            "target_rps",
            "actual_sent",
            "success_count",
            "fail_count",
            "sla_violation_count",
            "avg_latency_ms",
            "p95_latency_ms",
            "active_ports",
            "scenario",
            "phase",
        ])

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            exp_start = time.perf_counter() # 실험 시작 시간 기록

            for row in schedule:
                target_sec = row["time_sec"]
                target_rps = row["target_rps"]

                while True:
                    elapsed = time.perf_counter() - exp_start
                    if elapsed >= target_sec:
                        break
                    await asyncio.sleep(0.001)

                second_start = time.perf_counter() # 요청 보내기 시작 시간 기록

                active_ports = get_active_ports() # 현재 활성화된 포트 목록 조회

                if not active_ports:
                    print(
                        f"[t={row['time_sec']:>4}] "
                        f"활성 컨테이너 없음 - 요청 전송 불가"
                    )
                    writer.writerow([
                        row["time_sec"],
                        row["target_rps"],
                        0,
                        0,
                        0,
                        0,
                        0.0,
                        0.0,
                        "",
                        row["scenario"],
                        row["phase"],
                    ])
                    out.flush()
                    continue

                # round-robin 방식으로 URL 목록 생성
                urls = build_round_robin_urls(active_ports, target_rps) # 요청 수 만큼 URL 리스트 생성 (포트 순환)

                tasks = [
                    asyncio.create_task(send_one_request(session, url, sem))
                    for url in urls
                ]

                results = await asyncio.gather(*tasks) # 모든 요청이 완료될 때까지 대기

                latencies = [r["latency_ms"] for r in results] # 모든 요청의 지연 시간 리스트

                # 진짜 성공/실패 분리
                success_count = sum(1 for r in results if r["ok"])
                fail_count = len(results) - success_count

                # 느린 성공만 따로 카운트
                sla_violation_count = sum(1 for r in results if r["sla_violation"])

                avg_latency = statistics.mean(latencies) if latencies else 0.0
                p95_latency = percentile(latencies, 0.95) if latencies else 0.0

                ports_str = ",".join(str(p) for p in active_ports)

                writer.writerow([
                    row["time_sec"],
                    row["target_rps"],
                    len(results),
                    success_count,
                    fail_count,
                    sla_violation_count,
                    round(avg_latency, 2),
                    round(p95_latency, 2),
                    ports_str,
                    row["scenario"],
                    row["phase"],
                ])
                out.flush()

                spent = time.perf_counter() - second_start

                print(
                    f"[t={row['time_sec']:>4}] " # 빈칸4자리 확보해서 시간 표시
                    f"target={row['target_rps']:>4} "
                    f"sent={len(results):>4} "
                    f"ok={success_count:>4} "
                    f"fail={fail_count:>4} "
                    f"sla_fail={sla_violation_count:>4} "
                    f"avg={avg_latency:>7.2f}ms "
                    f"p95={p95_latency:>7.2f}ms "
                    f"ports={ports_str} "
                    f"phase={row['phase']} "
                    f"(spent {spent:.2f}s)"
                )


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    output_log = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_LOG

    if not csv_path.exists():
        print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)

    try:
        asyncio.run(run_schedule(csv_path, output_log))
    except KeyboardInterrupt:
        print("\n중단됨")


if __name__ == "__main__":
    main()
