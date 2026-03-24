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

# 요청 타임아웃(초)
REQUEST_TIMEOUT = 5

# 한 번에 너무 많은 태스크가 쌓이지 않도록 제한
MAX_CONCURRENCY = 1000


async def send_one_request(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore):
    async with sem:
        start = time.perf_counter()
        try:
            async with session.get(url) as resp:
                await resp.read()
                latency_ms = (time.perf_counter() - start) * 1000
                return {
                    "ok": 200 <= resp.status < 400,
                    "status": resp.status,
                    "latency_ms": latency_ms,
                }
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "ok": False,
                "status": "EXCEPTION",
                "latency_ms": latency_ms,
            }


def percentile(values, p):
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

                # CSV time_sec 기준으로 실제 실행 시간 동기화
                while True:
                    elapsed = time.perf_counter() - exp_start
                    if elapsed >= target_sec:
                        break
                    await asyncio.sleep(0.001)

                second_start = time.perf_counter()

                tasks = [
                    asyncio.create_task(send_one_request(session, target_url, sem))
                    for _ in range(target_rps)
                ]
                results = await asyncio.gather(*tasks)

                latencies = [r["latency_ms"] for r in results]
                success_count = sum(1 for r in results if r["ok"])
                fail_count = len(results) - success_count
                avg_latency = statistics.mean(latencies) if latencies else 0.0
                p95_latency = percentile(latencies, 0.95) if latencies else 0.0

                writer.writerow([
                    row["time_sec"],
                    row["target_rps"],
                    len(results),
                    success_count,
                    fail_count,
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