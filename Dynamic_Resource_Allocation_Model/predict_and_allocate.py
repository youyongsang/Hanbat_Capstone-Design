# -*- coding: utf-8 -*-
"""
predict_and_allocate.py — CSV 전환 지원 + 실시간 시뮬레이션
────────────────────────────────────────────────────────────
[문제 3 해결] CSV 전환
  - argparse로 --csv 옵션 추가 → 터미널에서 바로 전환 가능
    예) python predict_and_allocate.py --csv sale_event_traffic.csv
  - resource_policy 임포트 실패해도 내장 fallback 정책으로 동작

사용법:
  python predict_and_allocate.py                            # 기본 (week_traffic.csv)
  python predict_and_allocate.py --csv sale_event_traffic.csv
  python predict_and_allocate.py --csv week_traffic.csv --window 60
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

# ── resource_policy 안전 임포트 ─────────────────────────────────────────────
try:
    from resource_policy import allocate_resource
    print("✅ resource_policy.py 로드 성공")
except ImportError:
    print("⚠️  resource_policy.py 없음 → 내장 정책 사용")
    def allocate_resource(pred_rps: float):
        """내장 fallback: 100 RPS당 레플리카 1개 + 여유 1개"""
        replicas = min(max(2, int(np.ceil(pred_rps / 100)) + 1), 15)
        cpu = round(replicas * 0.5, 1)
        return cpu, replicas


def load_artifacts(window_size: int):
    """모델, 스케일러, 피처 컬럼 목록 로드"""
    from tensorflow.keras.models import load_model
    try:
        model = load_model('lstm_model.h5', compile=False)
        with open('scaler_x.pkl', 'rb') as f: sx = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f: sy = pickle.load(f)
        # 피처 컬럼 메타데이터 (preprocess 에서 저장)
        if os.path.exists('feature_cols.pkl'):
            with open('feature_cols.pkl', 'rb') as f: feat_cols = pickle.load(f)
        else:
            feat_cols = ['target_rps', 'diff', 'diff2', 'day_of_week', 'is_event', 'is_weekend']
        return model, sx, sy, feat_cols
    except Exception as e:
        print(f"❌ 모델/스케일러 로드 실패: {e}")
        print("   lstm_model.h5, scaler_x.pkl, scaler_y.pkl 이 필요합니다.")
        sys.exit(1)


def build_inference_features(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """preprocess_lstm.py 와 동일한 피처 생성 로직 (일관성 보장)"""
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 0
    if 'is_event' not in df.columns:
        df['is_event'] = 1 if ('sale' in os.path.basename(csv_path).lower()
                               or 'scenario' in df.columns) else 0
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['diff']  = df['target_rps'].diff().fillna(0)
    df['diff2'] = df['diff'].diff().fillna(0)
    return df


def run_simulation(csv_path: str, window_size: int = 60):
    """
    CSV 전체 데이터를 슬라이딩 윈도우로 예측 + 자원할당 시뮬레이션.
    결과를 터미널에 출력하고 CSV 로도 저장.
    """
    print(f"\n{'='*55}")
    print(f"🚀 시뮬레이션 시작: {csv_path}")
    print(f"{'='*55}\n")

    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} 파일이 없습니다.")
        sys.exit(1)

    model, sx, sy, feat_cols = load_artifacts(window_size)

    df = pd.read_csv(csv_path)
    df = build_inference_features(df, csv_path)

    n_features = len(feat_cols)
    data = df[feat_cols].values  # (N, 6)

    if len(data) < window_size:
        print(f"⚠️  데이터 부족: {len(data)} 행 < window_size {window_size}")
        sys.exit(1)

    # 스케일링
    scaled = sx.transform(data)

    results = []
    print(f"{'Idx':>6} | {'Actual':>8} | {'Predicted':>9} | {'CPU':>5} | {'Replicas':>8} | {'요일':>4} | {'이벤트':>4}")
    print("-" * 60)

    for i in range(len(scaled) - window_size):
        window = scaled[i : i + window_size].reshape(1, window_size, n_features)
        pred_scaled = model.predict(window, verbose=0)
        pred_rps = float(sy.inverse_transform(pred_scaled)[0][0])
        pred_rps = max(0.0, pred_rps)  # 음수 방지

        actual_rps  = float(df['target_rps'].iloc[i + window_size])
        day_of_week = int(df['day_of_week'].iloc[i + window_size])
        is_event    = int(df['is_event'].iloc[i + window_size])

        cpu, replicas = allocate_resource(pred_rps)

        day_names = ['월','화','수','목','금','토','일']
        day_str = day_names[day_of_week] if day_of_week < 7 else str(day_of_week)
        evt_str = "✅" if is_event else "  "

        # 100개마다 / 마지막 행 출력 (터미널 과부하 방지)
        if i % 100 == 0 or i == len(scaled) - window_size - 1:
            print(f"{i:>6} | {actual_rps:>8.1f} | {pred_rps:>9.1f} | "
                  f"{cpu:>5.1f} | {replicas:>8d} | {day_str:>4} | {evt_str:>4}")

        results.append({
            'index':       i,
            'actual_rps':  actual_rps,
            'pred_rps':    round(pred_rps, 2),
            'cpu':         cpu,
            'replicas':    replicas,
            'day_of_week': day_of_week,
            'is_event':    is_event,
        })

    result_df = pd.DataFrame(results)

    mae  = np.mean(np.abs(result_df['actual_rps'] - result_df['pred_rps']))
    rmse = np.sqrt(np.mean((result_df['actual_rps'] - result_df['pred_rps'])**2))

    print(f"\n{'='*55}")
    print(f"📊 평가 결과 ({os.path.basename(csv_path)})")
    print(f"   MAE  : {mae:.2f} RPS")
    print(f"   RMSE : {rmse:.2f} RPS")
    print(f"{'='*55}\n")

    out_name = f"result_{os.path.splitext(os.path.basename(csv_path))[0]}.csv"
    result_df.to_csv(out_name, index=False)
    print(f"💾 결과 저장: {out_name}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="LSTM 오토스케일링 시뮬레이터"
    )
    parser.add_argument(
        '--csv', type=str, default='week_traffic.csv',
        help="사용할 CSV 파일 (기본: week_traffic.csv)"
    )
    parser.add_argument(
        '--window', type=int, default=60,
        help="슬라이딩 윈도우 크기 (기본: 60)"
    )
    args = parser.parse_args()

    run_simulation(csv_path=args.csv, window_size=args.window)


if __name__ == "__main__":
    main()
