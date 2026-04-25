# -*- coding: utf-8 -*-
"""
evaluate_full.py — 종합 평가 + Baseline 비교 (단일 그래프 깔끔 버전)
────────────────────────────────────────────────────────────
사용법:
  python evaluate_full.py                     # week_traffic.csv
  python evaluate_full.py ddos_event.csv      # 원하는 CSV 입력
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from baseline import evaluate_baselines
from schema_adapter import SmartCSVLoader

warnings.filterwarnings('ignore', category=UserWarning)


def get_korean_font():
    """한글 폰트 설정 (깨짐 방지)"""
    candidates = ['NanumGothic', 'AppleGothic', 'Malgun Gothic', 'NanumBarunGothic']
    available  = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None


def evaluate_model_full(csv_path: str = 'week_traffic.csv'):
    print(f"📊 [{os.path.basename(csv_path)}] 종합 평가 시작...")

    # ── 아티팩트 로드 ──────────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model
        model    = load_model('lstm_model.h5', compile=False)
        X_train  = np.load('X_train.npy')
        y_train  = np.load('y_train.npy')
        X_test   = np.load('X_test.npy')
        y_test   = np.load('y_test.npy')
        with open('scaler_y.pkl', 'rb') as f: scaler_y = pickle.load(f)
        with open('scaler_x.pkl', 'rb') as f: scaler_x = pickle.load(f)
    except Exception as e:
        print(f"❌ 파일 없음: {e}")
        print("   preprocess_lstm.py → train_lstm.py 순서로 먼저 실행하세요.")
        return

    # ── 예측 및 역변환 ─────────────────────────────────────────────────────
    def inv_y(arr):
        return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

    y_tr_pred = inv_y(model.predict(X_train, verbose=0).flatten())
    y_te_pred = inv_y(model.predict(X_test,  verbose=0).flatten())
    y_tr_true = inv_y(y_train.flatten())
    y_te_true = inv_y(y_test.flatten())

    y_te_pred = np.maximum(y_te_pred, 0)
    y_tr_pred = np.maximum(y_tr_pred, 0)

    y_true = np.concatenate([y_tr_true, y_te_true])
    y_pred = np.concatenate([y_tr_pred, y_te_pred])

    mae      = float(np.mean(np.abs(y_te_true - y_te_pred)))
    rmse     = float(np.sqrt(np.mean((y_te_true - y_te_pred)**2)))
    nonzero  = y_te_true != 0
    mape     = float(np.mean(np.abs((y_te_true[nonzero] - y_te_pred[nonzero]) / y_te_true[nonzero])) * 100)
    p_thresh = float(np.quantile(y_te_true, 0.80))
    p_mask   = y_te_true >= p_thresh
    peak_mae = float(np.mean(np.abs(y_te_true[p_mask] - y_te_pred[p_mask])))

    print(f"\n{'='*45}")
    print(f"  [테스트셋 성능]")
    print(f"  MAE      : {mae:.2f} RPS")
    print(f"  RMSE     : {rmse:.2f} RPS")
    print(f"  MAPE     : {mape:.2f} %")
    print(f"  피크 MAE : {peak_mae:.2f} RPS  (≥{p_thresh:.0f} RPS, 상위 20%)")
    print(f"{'='*45}\n")

    # ── Baseline 비교 (수치만 출력) ──────────────────────────────────────────
    baseline_df = evaluate_baselines(y_te_true, lstm_pred=y_te_pred)

    # ── 시각화 (단일 그래프로 깔끔하게) ──────────────────────────────────────
    font_name = get_korean_font()
    if font_name:
        plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.close('all') # 기존 그래프 초기화

    plt.figure(figsize=(15, 6)) # 넓고 시원한 단일 프레임
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # 타이틀 설정
    plt.title(f'트래픽 예측 결과 ({base_name}) | MAE: {mae:.1f}, RMSE: {rmse:.1f}', 
              fontsize=16, fontweight='bold', pad=15)

    split_idx = len(y_tr_true)

    # 실제 트래픽과 예측 트래픽 선 그리기
    plt.plot(y_true, color='#B0BEC5', alpha=0.8, lw=1.5, label='실제 트래픽 (Actual)')
    plt.plot(y_pred, color='#1E88E5', ls='--', lw=1.5, label='LSTM 예측 (Predicted)')
    
    # Train / Test 경계선
    plt.axvline(x=split_idx, color='#E53935', ls=':', lw=2, label='Test 데이터 시작점')

    # 축 라벨 및 범례, 그리드 설정
    plt.xlabel('시간 (Time Steps)', fontsize=11)
    plt.ylabel('트래픽 (RPS)', fontsize=11)
    plt.legend(loc='upper right', fontsize=10, shadow=True, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    
    # 파일명에 원본 CSV 이름을 반영하여 저장
    output_filename = f"result_{base_name}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    
    print(f"✅ 단일 그래프 저장 완료: {output_filename}")
    plt.show()

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv'
    evaluate_model_full(csv_path=csv_arg)
