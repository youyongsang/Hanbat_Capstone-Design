from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    ACTUAL_TRAFFIC_CSV,
    FORECAST_TRAFFIC_CSV,
    PREDICTIVE_BASE_DIR,
    PREDICTIVE_RUN_HYBRID,
    PREDICTIVE_RESULTS_DIR,
    RL_PLAN_PATH,
    RESULTS_DIR,
    RUNTIME_ALLOC_LOG_CSV,
    RUNTIME_CORRECTION_LOG_CSV,
    RUNTIME_LOADGEN_CSV,
    RUNTIME_PREDICTED_CSV,
)


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} 파일이 없습니다: {path}")


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def collect_runtime_outputs():
    output_dir = PREDICTIVE_BASE_DIR / "data" / "output"
    copy_if_exists(output_dir / "predictive_allocation_log.csv", RUNTIME_ALLOC_LOG_CSV)
    copy_if_exists(output_dir / "hybrid_correction_log.csv", RUNTIME_CORRECTION_LOG_CSV)
    copy_if_exists(output_dir / "loadgen_result.csv", RUNTIME_LOADGEN_CSV)
    copy_if_exists(output_dir / "predicted_traffic.csv", RUNTIME_PREDICTED_CSV)

    for name in [
        "loadgen_performance.png",
        "correction_actions.png",
        "plan_vs_actual_resource.png",
        "resource_plan.png",
        "prediction_error.png",
        "predicted_vs_actual_rps.png",
    ]:
        copy_if_exists(PREDICTIVE_RESULTS_DIR / name, RESULTS_DIR / name)


def main():
    parser = argparse.ArgumentParser(description="RL 계획 CSV를 실제 predictive Docker 환경에 연결해 검증")
    parser.add_argument("--plan-csv", type=Path, default=RL_PLAN_PATH, help=f"RL 계획 CSV (기본: {RL_PLAN_PATH})")
    parser.add_argument("--actual-csv", type=Path, default=ACTUAL_TRAFFIC_CSV, help=f"실제 트래픽 CSV (기본: {ACTUAL_TRAFFIC_CSV})")
    parser.add_argument("--forecast-csv", type=Path, default=FORECAST_TRAFFIC_CSV, help=f"예측 CSV 확인용 입력 (기본: {FORECAST_TRAFFIC_CSV})")
    parser.add_argument("--skip-loadgen", action="store_true", help="로드 생성기 없이 RL 계획만 적용")
    args = parser.parse_args()

    require_file(PREDICTIVE_RUN_HYBRID, "predictive run_hybrid")
    require_file(args.plan_csv, "RL 계획 CSV")
    require_file(args.actual_csv, "실제 트래픽 CSV")
    require_file(args.forecast_csv, "예측 CSV")

    cmd = [
        sys.executable,
        str(PREDICTIVE_RUN_HYBRID),
        "--plan-csv",
        str(args.plan_csv),
        "--actual-csv",
        str(args.actual_csv),
    ]
    if not args.skip_loadgen:
        cmd.append("--with-loadgen")

    print("=" * 60)
    print(" predictive-rl-resource: RL 계획 기반 Docker 검증")
    print(f" RL plan     : {args.plan_csv}")
    print(f" Actual CSV  : {args.actual_csv}")
    print(f" Forecast CSV: {args.forecast_csv}")
    print(f" LoadGen     : {'OFF' if args.skip_loadgen else 'ON'}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(PREDICTIVE_BASE_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"run_hybrid.py 실행 실패 (exit={result.returncode})")

    collect_runtime_outputs()
    print(f"[DONE] RL 런타임 로그 복사 완료: {RUNTIME_ALLOC_LOG_CSV.parent}")
    print(f"[DONE] RL 결과 그래프 복사 완료: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
