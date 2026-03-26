import asyncio
import csv
import statistics
import sys
import time
from pathlib import Path

import aiohttp


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data/input/sale_event_traffic.csv"
TARGET_URL = "http://localhost:8000/work"
OUTPUT_LOG = BASE_DIR / "data/output/loadgen_result.csv"

# -------------------------
# 실험 파라미터
# -------------------------

# 요청 전체 타임아웃(초)
# 이 시간을 넘기면 aiohttp exception으로 처리되어 fail_count에 반영됨
REQUEST_TIMEOUT = 5

# 응답은 왔더라도, 이 시간(ms)을 넘으면 "품질 실패"로 간주
# 즉, HTTP 200이어도 너무 느리면 fail 처리
SLA_LATENCY_MS = 2000

# 동시에 너무 많은 요청이 몰려 프로그램 자체가 불안정해지는 걸 방지
MAX_CONCURRENCY = 1000


async def send_one_request(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore):
    """
    단일 요청 전송 함수

    성공 기준:
    1) HTTP 상태 코드가 2xx~3xx
    2) latency_ms <= SLA_LATENCY_MS

    위 조건 둘 다 만족해야 ok=True
    """
    async with sem:
        start = time.perf_counter()
        try:
            async with session.get(url) as resp:
                await resp.read()
                latency_ms = (time.perf_counter() - start) * 1000.0

                http_ok = 200 <= resp.status < 400
                sla_ok = latency_ms <= SLA_LATENCY_MS
                ok = http_ok and sla_ok

                return {
                    "ok": ok,
                    "status": resp.status,
                    "latency_ms": latency_ms,
                    "sla_violation": http_ok and not sla_ok,
                }

        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "ok": False,
                "status": "EXCEPTION",
                "latency_ms": latency_ms,
                "sla_violation": False,
            }


def percentile(values, p):
    """
    간단한 percentile 계산 함수
    p=0.95 이면 p95 latency 계산
    """
    if not values:
        return 0.0

    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def load_schedule(csv_path: Path):
    """
    트래픽 시나리오 CSV 로드
    필수 컬럼:
    - time_sec
    - target_rps
    - scenario
    - phase
    """
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


async def run_schedule(csv_path: Path, target_url: str, output_log: Path):
    """
    CSV 스케줄에 따라 초 단위로 target_rps만큼 요청 전송
    """
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
            "scenario",
            "phase",
        ])

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            exp_start = time.perf_counter()

            for row in schedule:
                target_sec = row["time_sec"]
                target_rps = row["target_rps"]

                # 해당 초까지 대기
                while True:
                    elapsed = time.perf_counter() - exp_start
                    if elapsed >= target_sec:
                        break
                    await asyncio.sleep(0.001)

                second_start = time.perf_counter()

                # target_rps만큼 요청 생성
                tasks = [
                    asyncio.create_task(send_one_request(session, target_url, sem))
                    for _ in range(target_rps)
                ]

                results = await asyncio.gather(*tasks)

                latencies = [r["latency_ms"] for r in results]
                success_count = sum(1 for r in results if r["ok"])
                fail_count = len(results) - success_count
                sla_violation_count = sum(1 for r in results if r["sla_violation"])

                avg_latency = statistics.mean(latencies) if latencies else 0.0
                p95_latency = percentile(latencies, 0.95) if latencies else 0.0

                writer.writerow([
                    row["time_sec"],
                    row["target_rps"],
                    len(results),
                    success_count,
                    fail_count,
                    sla_violation_count,
                    round(avg_latency, 2),
                    round(p95_latency, 2),
                    row["scenario"],
                    row["phase"],
                ])
                out.flush()

                spent = time.perf_counter() - second_start

                print(
                    f"[t={row['time_sec']:>4}] "
                    f"target={row['target_rps']:>4} "
                    f"sent={len(results):>4} "
                    f"ok={success_count:>4} "
                    f"fail={fail_count:>4} "
                    f"sla_fail={sla_violation_count:>4} "
                    f"avg={avg_latency:>7.2f}ms "
                    f"p95={p95_latency:>7.2f}ms "
                    f"phase={row['phase']} "
                    f"(spent {spent:.2f}s)"
                )


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    target_url = sys.argv[2] if len(sys.argv) > 2 else TARGET_URL
    output_log = Path(sys.argv[3]) if len(sys.argv) > 3 else OUTPUT_LOG

    if not csv_path.exists():
        print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)

    try:
        asyncio.run(run_schedule(csv_path, target_url, output_log))
    except KeyboardInterrupt:
        print("\n중단됨")


if __name__ == "__main__":
    main()