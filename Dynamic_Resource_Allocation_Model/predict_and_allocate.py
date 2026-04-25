# -*- coding: utf-8 -*-
"""
predict_and_allocate.py — 자동 Window Size 매칭 & 8-Feature 완벽 호환
"""
import argparse, os, pickle, sys, warnings
import numpy as np
import pandas as pd
from schema_adapter import SmartCSVLoader, STANDARD_FEATURES

try:
    from resource_policy import allocate_resource
except ImportError:
    def allocate_resource(pred_rps: float):
        replicas = min(max(2, int(np.ceil(pred_rps / 100)) + 1), 15)
        return round(replicas * 0.5, 1), replicas

def load_model_artifacts():
    from tensorflow.keras.models import load_model
    try:
        model = load_model('lstm_model.h5', compile=False)
        with open('scaler_x.pkl', 'rb') as f: sx = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f: sy = pickle.load(f)
        feat_cols = STANDARD_FEATURES
        if os.path.exists('feature_cols.pkl'):
            with open('feature_cols.pkl', 'rb') as f: feat_cols = pickle.load(f)
            
        # [핵심] 학습된 모델의 Input Shape에서 Window Size를 자동으로 추출합니다!
        model_window_size = model.input_shape[1] 
        return model, sx, sy, feat_cols, model_window_size
    except Exception as e:
        print(f"❌ 모델/스케일러 로드 실패: {e}")
        sys.exit(1)

def run_simulation(csv_path: str):
    print(f"\n{'='*55}")
    print(f"🚀 실시간 시뮬레이션 시작: {csv_path}")
    print(f"{'='*55}\n")

    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} 파일이 없습니다.")
        sys.exit(1)

    model, sx, sy, feat_cols, window_size = load_model_artifacts()
    print(f"🔍 모델에서 자동 감지된 Window Size: {window_size}")

    loader = SmartCSVLoader(csv_path, verbose=False) # 리포트 간소화
    df = loader.load()
    data = df[feat_cols].values

    if len(data) < window_size:
        print(f"⚠️ 데이터 부족: {len(data)} < {window_size}")
        sys.exit(1)

    # 스케일러 호환성 에러 방지 (Standard Scaler든 Adaptive든 작동하도록)
    try:
        scaled = sx.transform(data, check_ood=True)
    except TypeError:
        scaled = sx.transform(data) # check_ood를 지원하지 않는 일반 Scaler일 경우

    results = []
    print(f"\n{'Idx':>5} | {'Actual':>7} | {'Predicted':>9} | {'CPU':>4} | {'Rep':>3} | {'이벤트':>4}")
    print("-" * 55)

    for i in range(len(scaled) - window_size):
        window     = scaled[i : i + window_size].reshape(1, window_size, len(feat_cols))
        pred_sc    = model.predict(window, verbose=0)
        pred_rps   = max(0.0, float(sy.inverse_transform(pred_sc.reshape(-1,1)).flatten()[0]))

        actual_rps  = float(df['target_rps'].iloc[i + window_size])
        is_event    = int(df['is_event'].iloc[i + window_size])
        cpu, replicas = allocate_resource(pred_rps)
        evt_str = "🔴" if is_event else "  "

        if i % 100 == 0 or i == len(scaled) - window_size - 1:
            print(f"{i:>5} | {actual_rps:>7.1f} | {pred_rps:>9.1f} | {cpu:>4.1f} | {replicas:>3d} | {evt_str:>4}")

        results.append({
            'index': i, 'actual_rps': actual_rps, 'pred_rps': round(pred_rps, 2),
            'cpu': cpu, 'replicas': replicas, 'is_event': is_event
        })

    result_df = pd.DataFrame(results)
    mae  = float(np.mean(np.abs(result_df['actual_rps'] - result_df['pred_rps'])))
    rmse = float(np.sqrt(np.mean((result_df['actual_rps'] - result_df['pred_rps'])**2)))

    print(f"\n📊 [시뮬레이션 결과] MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    out = f"sim_result_{os.path.splitext(os.path.basename(csv_path))[0]}.csv"
    result_df.to_csv(out, index=False)
    print(f"💾 결과 저장: {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='week_traffic.csv', help="CSV 파일 경로")
    # window 인자는 모델에서 자동 추출하므로 제거했습니다.
    args = parser.parse_args()
    run_simulation(csv_path=args.csv)

if __name__ == "__main__":
    main()
