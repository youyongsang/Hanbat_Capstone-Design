import sys
import time
import threading
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predictive_allocator import PredictiveAllocator, AllocationState
from hybrid_controller import HybridController


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ACTUAL_CSV = BASE_DIR / "data" / "input" / "sale_event_traffic.csv"
DEFAULT_PLAN_CSV = BASE_DIR / "data" / "output" / "resource_allocation_plan.csv"
LOAD_GEN = BASE_DIR / "scripts" / "hybrid_load_generator.py"


def run_predictive(plan_csv: Path, state: AllocationState):
    try:
        PredictiveAllocator(plan_csv, state).run()
    except Exception as e:
        print(f"[ERROR] PredictiveAllocator: {e}")


def run_controller(state: AllocationState):
    print("[Main] HybridController: 5초 후 시작 (계획 적용 안정화 대기)")
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


def print_summary():
    import csv as _csv

    pred_csv = BASE_DIR / "data" / "output" / "predictive_allocation_log.csv"
    corr_csv = BASE_DIR / "data" / "output" / "hybrid_correction_log.csv"

    print("\n" + "=" * 55)
    print("  실험 결과 요약")
    print("=" * 55)

    if pred_csv.exists():
        with open(pred_csv, "r", encoding="utf-8") as f:
            rows = list(f)
        print(f"  계획 실행 로그  : {pred_csv.name}  ({len(rows) - 1} rows)")
    else:
        print("  계획 실행 로그  : 파일 없음")

    if corr_csv.exists():
        actions = {}
        with open(corr_csv, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                action = row.get("action", "HOLD")
                actions[action] = actions.get(action, 0) + 1
        print(f"  보정 로그       : {corr_csv.name}")
        for key, value in sorted(actions.items()):
            if value > 0:
                print(f"    {key:<12}: {value}회")
    else:
        print("  보정 로그       : 파일 없음")

    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="계획 기반 선제 할당 + 실시간 보정 하이브리드 시스템"
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=DEFAULT_PLAN_CSV,
        help=f"미리 생성된 자원 할당 계획 CSV (기본: {DEFAULT_PLAN_CSV})",
    )
    parser.add_argument(
        "--actual-csv",
        type=Path,
        default=DEFAULT_ACTUAL_CSV,
        help=f"실제 로드 생성용 트래픽 CSV (기본: {DEFAULT_ACTUAL_CSV})",
    )
    parser.add_argument(
        "--with-loadgen",
        action="store_true",
        help="로드 생성기를 함께 실행",
    )
    args = parser.parse_args()

    if not args.plan_csv.exists():
        print(f"[ERROR] 계획 CSV 없음: {args.plan_csv}")
        sys.exit(1)
    if args.with_loadgen and not args.actual_csv.exists():
        print(f"[ERROR] 실제 트래픽 CSV 없음: {args.actual_csv}")
        sys.exit(1)

    print("=" * 55)
    print("  predictive-resource: 계획 기반 하이브리드 동적 자원 할당")
    print(f"  계획 CSV   : {args.plan_csv}")
    if args.with_loadgen:
        print(f"  실제 CSV   : {args.actual_csv}")
    print("=" * 55)

    state = AllocationState()
    threads = []

    t_pred = threading.Thread(
        target=run_predictive,
        args=(args.plan_csv, state),
        name="PredictiveAllocator",
        daemon=False,
    )
    t_ctrl = threading.Thread(
        target=run_controller,
        args=(state,),
        name="HybridController",
        daemon=False,
    )
    threads += [t_pred, t_ctrl]
    t_loadgen = None

    if args.with_loadgen:
        t_loadgen = threading.Thread(
            target=run_loadgen,
            args=(args.actual_csv,),
            name="LoadGenerator",
            daemon=False,
        )
        threads.append(t_loadgen)

    for thread in threads:
        thread.start()
        print(f"[Main] {thread.name} 시작")

    t_pred.join()
    print("[Main] 계획 실행 완료 — 보정 레이어 종료 대기 (최대 30초)")

    if t_loadgen is not None:
        print("[Main] 로드 생성기 종료 대기")
        t_loadgen.join()

    print("[Main] 보정 레이어 종료 대기")
    t_ctrl.join(timeout=30)

    print_summary()
    print("[Main] 종료")


if __name__ == "__main__":
    main()
