# main.py (프로젝트 루트에 저장)
import argparse
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run(cmd: list[str], title: str):
    print(f"\n==================== {title} ====================")
    print("CMD:", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True)
    if p.returncode != 0:
        raise SystemExit(f"\n❌ 실패: {title} (exit={p.returncode})")
    print(f"✅ 완료: {title}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_gen", action="store_true", help="데이터 생성(generator) 스킵")
    ap.add_argument("--skip_prep", action="store_true", help="전처리(preprocessor) 스킵")
    ap.add_argument("--skip_train", action="store_true", help="학습(train) 스킵")
    ap.add_argument("--check_npz", action="store_true", help="전처리 후 NPZ 구조 출력")

    ap.add_argument("--label_mode", choices=["next_step", "last_step"], default="next_step")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    py = sys.executable  # 현재 가상환경의 python을 그대로 사용 (중요!)

    # 1) 데이터 생성
    if not args.skip_gen:
        run([py, "preprocessing/generator.py"], "1) 데이터 생성 (generator)")

    # 2) 전처리
    if not args.skip_prep:
        run([py, "preprocessing/preprocessor.py"], "2) 전처리 (preprocessor)")

    # 3) npz 확인(선택)
    if args.check_npz:
        run([py, "preprocessing/check_npz.py"], "3) NPZ 구조 확인 (check_npz)")

    # 4) 학습
    if not args.skip_train:
        train_cmd = [
            py, "train/train_ys_alloc.py",
            "--label_mode", args.label_mode,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--hidden", str(args.hidden),
            "--dropout", str(args.dropout),
        ]
        run(train_cmd, "4) 모델 학습 (train_ys_alloc)")

    # 5) 최종 결과 안내
    print("\n==================== 5) 최종 결과 위치 ====================")
    artifacts = PROJECT_ROOT / "artifacts"
    model_path = artifacts / "ys_alloc_net.pt"

    print("📁 데이터(원본):", PROJECT_ROOT / "data" / "raw" / "traffic_log.csv")
    print("📁 데이터(가공):", PROJECT_ROOT / "data" / "processed")
    print("📁 산출물(모델):", model_path)

    if model_path.exists():
        print("✅ 모델 파일이 존재합니다. (학습 성공)")
    else:
        print("⚠️ 모델 파일이 없습니다. 학습을 스킵했거나 오류로 저장이 안 됐을 수 있어요.")

    print("\n끝! 이제 artifacts/ys_alloc_net.pt 가 최종 결과물이다.")

if __name__ == "__main__":
    main()
