# -*- coding: utf-8 -*-
"""
predict_and_allocate.py — CSV 전환 + OOD 경고 + 파이프라인 자동화 (단점 3, 5, 6)
────────────────────────────────────────────────────────────
사용법:
  python predict_and_allocate.py                              # 기본 week_traffic.csv
  python predict_and_allocate.py --csv sale_event_traffic.csv
  python predict_and_allocate.py --csv my_custom.csv         # 어떤 CSV든 OK
  python predict_and_allocate.py --csv sale_event_traffic.csv --window 60
"""
import argparse, os, pickle, sys, warnings
import numpy as np
import pandas as pd

from schema_adapter import SmartCSVLoader, STANDARD_FEATURES

# resource_policy 안전 임포트 (없으면 내장 fallback)
try:
    from resource_policy import allocate_resource
    print("✅ resource_policy.py 로드 성공")
except ImportError:
    def allocate_resource(pred_rps: float):
        replicas = min(max(2, int(np.ceil(pred_rps / 100)) + 1), 15)
        return round(replicas * 0.5, 1), replicas


def load_model_artifacts(window_size: int):
    from tensorflow.keras.models import load_model
    try:
        model = load_model('lstm_model.h5', compile=False)
        with open('scaler_x.pkl', 'rb') as f: sx = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f: sy = pickle.load(f)
        feat_cols = STANDARD_FEATURES
        if os.path.exists('feature_cols.pkl'):
            with open('feature_cols.pkl', 'rb') as f: feat_cols = pickle.load(f)
        return model, sx, sy, feat_cols
    except Exception as e:
        print(f"❌ 모델/스케일러 로드 실패: {e}")
        sys.exit(1)


def run_simulation(csv_path: str, window_size: int = 60):
    print(f"\n{'='*55}")
    print(f"🚀 시뮬레이션 시작: {csv_path}")
    print(f"{'='*55}\n")

    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} 파일이 없습니다.")
        sys.exit(1)

    model, sx, sy, feat_cols = load_model_artifacts(window_size)

    # SmartCSVLoader → 어떤 CSV든 자동 변환
    loader = SmartCSVLoader(csv_path, verbose=True)
    df = loader.load()

    data = df[feat_cols].values
    if len(data) < window_size:
        print(f"⚠️  데이터 부족: {len(data)} < {window_size}")
        sys.exit(1)

    # OOD 탐지 포함 스케일링 (AdaptiveScaler의 경고 자동 출력)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaled = sx.transform(data, check_ood=True)
        for w in caught:
            if issubclass(w.category, UserWarning):
                print(str(w.message))

    results = []
    print(f"\n{'Idx':>6} | {'Actual':>8} | {'Predicted':>9} | {'CPU':>5} | {'Replicas':>8} | {'요일':>4} | {'이벤트':>4}")
    print("-" * 60)

    day_names = ['월', '화', '수', '목', '금', '토', '일']

    for i in range(len(scaled) - window_size):
        window     = scaled[i : i + window_size].reshape(1, window_size, len(feat_cols))
        pred_sc    = model.predict(window, verbose=0)
        pred_rps   = max(0.0, float(sy.inverse_transform(pred_sc.reshape(-1,1)).flatten()[0]))

        actual_rps  = float(df['target_rps'].iloc[i + window_size])
        day_of_week = int(df['day_of_week'].iloc[i + window_size])
        is_event    = int(df['is_event'].iloc[i + window_size])

        cpu, replicas = allocate_resource(pred_rps)

        day_str = day_names[day_of_week] if day_of_week < 7 else str(day_of_week)
        evt_str = "✅" if is_event else "  "

        if i % 100 == 0 or i == len(scaled) - window_size - 1:
            print(f"{i:>6} | {actual_rps:>8.1f} | {pred_rps:>9.1f} | "
                  f"{cpu:>5.1f} | {replicas:>8d} | {day_str:>4} | {evt_str:>4}")

        results.append({
            'index': i, 'actual_rps': actual_rps, 'pred_rps': round(pred_rps, 2),
            'cpu': cpu, 'replicas': replicas,
            'day_of_week': day_of_week, 'is_event': is_event,
        })

    result_df = pd.DataFrame(results)
    mae  = float(np.mean(np.abs(result_df['actual_rps'] - result_df['pred_rps'])))
    rmse = float(np.sqrt(np.mean((result_df['actual_rps'] - result_df['pred_rps'])**2)))

    print(f"\n{'='*55}")
    print(f"📊 평가: {os.path.basename(csv_path)}")
    print(f"   MAE  : {mae:.2f} RPS")
    print(f"   RMSE : {rmse:.2f} RPS")
    print(f"{'='*55}\n")

    out = f"result_{os.path.splitext(os.path.basename(csv_path))[0]}.csv"
    result_df.to_csv(out, index=False)
    print(f"💾 결과 저장: {out}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="LSTM 오토스케일링 시뮬레이터")
    parser.add_argument('--csv',    type=str, default='week_traffic.csv',
                        help="CSV 파일 경로 (기본: week_traffic.csv)")
    parser.add_argument('--window', type=int, default=60,
                        help="슬라이딩 윈도우 크기 (기본: 60)")
    args = parser.parse_args()
    run_simulation(csv_path=args.csv, window_size=args.window)


if __name__ == "__main__":
    main()
