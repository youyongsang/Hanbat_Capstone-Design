# -*- coding: utf-8 -*-
"""
preprocess_lstm.py
─────────────────────────────────────────────────────────────────
sale_event / week_traffic 두 CSV 모두 완벽 추종을 위한 분기 로직

[sale_event] window=20, phase분할(peak→test포함), 데이터 4배 증강
[week_traffic] window=60, 시간순 8:2 분할 (기존 방식 유지)
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = ['target_rps','diff','diff2','ma5','ma20',
                'day_of_week','is_event','is_weekend']

def get_config(csv_path):
    fname = os.path.basename(csv_path).lower()
    if 'sale' in fname or 'event' in fname:
        return {'window': 20, 'split': 'phase', 'augment': True,  'aug_copies': 4}
    else:
        return {'window': 60, 'split': 'time',  'augment': False, 'aug_copies': 0}

def build_features(df, csv_path):
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = (df.index // 2001 % 7).astype(int) if 'time_sec' in df.columns else 0
    df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)

    if 'is_event' not in df.columns:
        fname = os.path.basename(csv_path).lower()
        df['is_event'] = 1 if ('sale' in fname or 'scenario' in df.columns or 'phase' in df.columns) else 0
    df['is_event'] = pd.to_numeric(df['is_event'], errors='coerce').fillna(0).astype(int)

    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_weekend'] = pd.to_numeric(df['is_weekend'], errors='coerce').fillna(0).astype(int)

    df['diff']  = df['target_rps'].diff().fillna(0)
    df['diff2'] = df['diff'].diff().fillna(0)
    df['ma5']   = df['target_rps'].rolling(5,  min_periods=1).mean()
    df['ma20']  = df['target_rps'].rolling(20, min_periods=1).mean()
    return df

def create_windows(x, y, window):
    X, Y = [], []
    for i in range(len(x) - window):
        X.append(x[i:i+window])
        Y.append(y[i+window])
    return np.array(X), np.array(Y)

def augment(X, y, n_copies):
    rng = np.random.default_rng(42)
    Xs, Ys = [X], [y]
    for _ in range(n_copies):
        noise = rng.normal(0, 0.008, X.shape)
        scale = rng.uniform(0.95, 1.05, (len(X), 1, 1))
        Xa = np.clip(X + noise, 0, 1) * scale
        ya = np.clip(y * scale.reshape(-1, 1), 0, 1)
        Xs.append(np.clip(Xa, 0, 1))
        Ys.append(ya)
    return np.concatenate(Xs), np.concatenate(Ys)

def phase_split(df, window):
    TRAIN_P = {'normal','baseline','ramp_up','warmup','stable_low'}
    phase_col = 'phase' if 'phase' in df.columns else None

    if phase_col is None:
        split = int(len(df) * 0.4)
        return df.iloc[:split].reset_index(drop=True), \
               pd.concat([df.iloc[max(0,split-window):split], df.iloc[split:]], ignore_index=True)

    train_mask = df[phase_col].str.lower().isin(TRAIN_P)
    if train_mask.sum() < window * 2:
        split = int(len(df) * 0.4)
        return df.iloc[:split].reset_index(drop=True), \
               pd.concat([df.iloc[max(0,split-window):split], df.iloc[split:]], ignore_index=True)

    train_df = df[train_mask].reset_index(drop=True)
    test_df  = pd.concat([df[train_mask].tail(window), df[~train_mask]], ignore_index=True)
    print(f"   train phases: {df[train_mask][phase_col].unique().tolist()} → {len(train_df)}행")
    print(f"   test  phases: {df[~train_mask][phase_col].unique().tolist()} → {len(df[~train_mask])}행")
    return train_df, test_df

def preprocess_ultimate(csv_path):
    print(f"\n{'='*50}\n🛠️  전처리: {os.path.basename(csv_path)}\n{'='*50}")
    if not os.path.exists(csv_path):
        print(f"❌ 파일 없음: {csv_path}"); return False

    cfg    = get_config(csv_path)
    window = cfg['window']
    print(f"   window={window}  split={cfg['split']}  augment={cfg['augment']}")

    df = pd.read_csv(csv_path)
    df = build_features(df, csv_path)

    if cfg['split'] == 'phase':
        train_df, test_df = phase_split(df, window)
        train_x = train_df[FEATURE_COLS].values
        train_y = train_df['target_rps'].values.reshape(-1, 1)
        test_x  = test_df[FEATURE_COLS].values
        test_y  = test_df['target_rps'].values.reshape(-1, 1)
    else:
        split   = int(len(df) * 0.8)
        data_x  = df[FEATURE_COLS].values
        data_y  = df['target_rps'].values.reshape(-1, 1)
        train_x, train_y = data_x[:split], data_y[:split]
        test_x,  test_y  = data_x[split-window:], data_y[split-window:]
        print(f"   시간순 8:2: train={split}행  test={len(df)-split}행")

    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_x.fit(train_x);  scaler_y.fit(train_y)

    X_train, y_train = create_windows(scaler_x.transform(train_x), scaler_y.transform(train_y), window)
    X_test,  y_test  = create_windows(scaler_x.transform(test_x),  scaler_y.transform(test_y),  window)

    if cfg['augment'] and len(X_train) < 2000:
        before = len(X_train)
        X_train, y_train = augment(X_train, y_train, cfg['aug_copies'])
        print(f"   📈 증강: {before} → {len(X_train)} 샘플")

    np.save('X_train.npy', X_train);  np.save('y_train.npy', y_train)
    np.save('X_test.npy',  X_test);   np.save('y_test.npy',  y_test)
    with open('scaler_x.pkl','wb') as f: pickle.dump(scaler_x, f)
    with open('scaler_y.pkl','wb') as f: pickle.dump(scaler_y, f)
    with open('feature_cols.pkl','wb') as f: pickle.dump(FEATURE_COLS, f)
    with open('metadata.pkl','wb') as f:
        pickle.dump({'feature_cols':FEATURE_COLS,'window_size':window,'csv_path':csv_path}, f)

    print(f"\n✅ 완료: X_train={X_train.shape}  X_test={X_test.shape}")
    return True

if __name__ == "__main__":
    preprocess_ultimate(sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv')
