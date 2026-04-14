# main.py
import subprocess
import os
import sys

def run_project():
    print("🚀 [지능형 Auto-Scaling 시스템] 파이프라인 가동...")

    # 실행할 순서와 파일명 정의 (사용자님 깃허브 파일명 기준)
    pipeline = [
        ("전처리", "preprocess_lstm.py"),
        ("모델 학습", "train_lstm.py"),  # train_ultimate.py를 train_lstm.py로 수정
        ("결과 평가", "evaluate_predictive.py")
    ]

    for step_name, file_name in pipeline:
        if not os.path.exists(file_name):
            print(f"❌ 에러: '{file_name}' 파일이 현재 폴더에 없습니다.")
            print("파일명을 확인하거나 깃허브에서 경로를 체크해주세요.")
            sys.exit(1)

        print(f"\n[작업 시작] {step_name} 중... ({file_name})")
        
        try:
            # check=True: 실행 중 에러 발생 시 즉시 중단
            subprocess.run(["python", file_name], check=True)
            print(f"✅ {step_name} 완료!")
        except subprocess.CalledProcessError as e:
            print(f"💥 {step_name} 단계에서 치명적 오류 발생: {e}")
            sys.exit(1)

    print("\n" + "="*40)
    print("✨ 모든 파이프라인이 성공적으로 완료되었습니다!")
    print("📈 결과물: lstm_model.h5, scaler_x.pkl, scaler_y.pkl, 각종 .npy 파일")
    print("="*40)

if __name__ == "__main__":
    run_project()
