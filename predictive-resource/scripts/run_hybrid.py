"""
run_hybrid.py
=============
하이브리드 동적 자원 할당 시스템 진입점.

스레드 구성
-----------
- Thread A : PredictiveAllocator  — CSV 스케줄 기반 LSTM 예측 + Docker 선제 할당
- Thread B : HybridController     — 실시간 /metrics 폴링 + 편차 발생 시 보정
- Thread C : LoadGenerator (선택) — hybrid_load_generator.py 서브프로세스 실행

사용법
------
  # 기본 실행 (로드 생성기는 별도 터미널)
  python scripts/run_hybrid.py

  # 로드 생성기 함께 실행
  python scripts/run_hybrid.py --with-loadgen

  # CSV 경로 직접 지정
  python scripts/run_hybrid.py --csv data/input/sale_event_traffic.csv
"""

import sys
import time
import threading
import argparse
import subprocess
from pathlib import Path

# scripts/ 디렉토리 기준 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predictive_allocator import PredictiveAllocator, AllocationState
from hybrid_controller import HybridController

# =========================
# 경로 설정
# =========================

BASE_DIR     = Path(__file__).resolve().parent.parent   # predictive-resource/
DEFAULT_CSV  = BASE_DIR / "data" / "input" / "sale_event_traffic.csv"
LOAD_GEN     = BASE_DIR / "scripts" / "hybrid_load_generator.py"


# =========================
# 스레드 래퍼
# =========================

def run_predictive(csv_path: Path, state: AllocationState):
    try:
        PredictiveAllocator(csv_path, state).run()
    except Exception as e:
        print(f"[ERROR] PredictiveAllocator: {e}")


def run_controller(state: AllocationState):
    print("[Main] HybridController: 5초 후 시작 (예측 레이어 준비 대기)")
    time.sleep(5)
    try:
        HybridController(state).run()
    except Exception as e:
        print(f"[ERROR] HybridController: {e}")


def run_loadgen(csv_path: Path):
    if not LOAD_GEN.exists():
        print(f"[WARN] 로드 생성기 없음: {LOAD_GEN}")
        return
    print(f"[Main] 로드 생성기 시작: {LOAD_GEN}")
    subprocess.run([sys.executable, str(LOAD_GEN), str(csv_path)])


# =========================
# 결과 요약
# =========================

def print_summary():
    import csv as _csv

    pred_csv = BASE_DIR / "data" / "output" / "predictive_allocation_plan.csv"
    corr_csv = BASE_DIR / "data" / "output" / "hybrid_correction_log.csv"

    print("\n" + "=" * 55)
    print("  실험 결과 요약")
    print("=" * 55)

    if pred_csv.exists():
        with open(pred_csv, "r", encoding="utf-8") as f:
            rows = list(f)
        print(f"  예측 할당 로그  : {pred_csv.name}  ({len(rows)-1} rows)")
    else:
        print("  예측 할당 로그  : 파일 없음")

    if corr_csv.exists():
        actions = {}
        with open(corr_csv, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                a = row.get("action", "HOLD")
                actions[a] = actions.get(a, 0) + 1
        print(f"  보정 로그       : {corr_csv.name}")
        for k, v in sorted(actions.items()):
            if v > 0:
                print(f"    {k:<12}: {v}회")
    else:
        print("  보정 로그       : 파일 없음")

    print("=" * 55)


# =========================
# 메인
# =========================

def main():
    parser = argparse.ArgumentParser(description="하이브리드 동적 자원 할당 시스템")
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"트래픽 스케줄 CSV (기본: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--with-loadgen", action="store_true",
        help="로드 생성기를 함께 실행",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"[ERROR] CSV 없음: {args.csv}")
        sys.exit(1)

    print("=" * 55)
    print("  predictive-resource: 하이브리드 동적 자원 할당")
    print(f"  트래픽 CSV : {args.csv}")
    print("=" * 55)

    state   = AllocationState()
    threads = []

    t_pred = threading.Thread(
        target=run_predictive, args=(args.csv, state),
        name="PredictiveAllocator", daemon=True,
    )
    t_ctrl = threading.Thread(
        target=run_controller, args=(state,),
        name="HybridController", daemon=True,
    )
    threads += [t_pred, t_ctrl]

    if args.with_loadgen:
        threads.append(threading.Thread(
            target=run_loadgen, args=(args.csv,),
            name="LoadGenerator", daemon=True,
        ))

    for t in threads:
        t.start()
        print(f"[Main] {t.name} 시작")

    t_pred.join()
    print("[Main] 예측 레이어 완료 — 보정 레이어 종료 대기 (최대 30초)")
    t_ctrl.join(timeout=30)

    print_summary()
    print("[Main] 종료")


if __name__ == "__main__":
    main()
