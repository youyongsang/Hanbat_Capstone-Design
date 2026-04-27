# -*- coding: utf-8 -*-
"""
preprocess_lstm.py
──────────────────────────────────────────────────────────────────
[sale_event]  window=5,  phase 분할(peak→test포함), scaler=전체fit
[week_traffic] window=60, 시간순 8:2,              scaler=train fit

사용법:
  python preprocess_lstm.py sale_event_traffic.csv
  python preprocess_lstm.py week_traffic.csv
"""
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = ['target_rps', 'diff', 'diff2']


def is_sale(path): 
    return 'sale' in os.path.basename(path).lower()


def add_features(df):
    df['diff']  = df['target_rps'].diff().fillna(0)
    df['diff2'] = df['diff'].diff().fillna(0)
    return df


def make_windows(x, y, w):
    X, Y = [], []
    for i in range(len(x) - w):
        X.append(x[i:i+w])
        Y.append(y[i+w])
    return np.array(X), np.array(Y)


def preprocess_ultimate(csv_path):
    print(f"\n{'='*50}\n🛠️  {os.path.basename(csv_path)}\n{'='*50}")
    df = pd.read_csv(csv_path)
    df = add_features(df)
    dx = df[FEATURE_COLS].values
    dy = df['target_rps'].values.reshape(-1, 1)

    if is_sale(csv_path):
        # ── sale_event: window=5, phase 분할 ──────────────────────────
        W = 5
        TRAIN_P = {'normal', 'ramp_up'}
        pc = 'phase' if 'phase' in df.columns else None

        if pc and df[pc].str.lower().isin(TRAIN_P).sum() > W * 2:
            tm = df[pc].str.lower().isin(TRAIN_P)
            ti = df[tm].index.tolist()
            ei = df[~tm].index.tolist()
            ei_full = df[tm].tail(W).index.tolist() + ei
            print(f"   train: {df[tm][pc].unique().tolist()} ({len(ti)}행)")
            print(f"   test:  {df[~tm][pc].unique().tolist()} ({len(ei)}행)")
            tx, ty = dx[ti], dy[ti]
            ex, ey = dx[ei_full], dy[ei_full]
        else:
            s = int(len(df)*0.4)
            tx, ty = dx[:s], dy[:s]
            ex, ey = dx[max(0,s-W):], dy[max(0,s-W):]

        # 전체 데이터로 fit (원래 잘 됐던 방식)
        sx, sy = MinMaxScaler(), MinMaxScaler()
        sx.fit(dx);  sy.fit(dy)

        X_train, y_train = make_windows(sx.transform(tx), sy.transform(ty), W)
        X_test,  y_test  = make_windows(sx.transform(ex), sy.transform(ey), W)

    else:
        # ── week_traffic: window=60, 시간순 8:2 ───────────────────────
        W = 60
        s = int(len(df) * 0.8)
        tx, ty = dx[:s], dy[:s]
        ex, ey = dx[s-W:], dy[s-W:]
        sx, sy = MinMaxScaler(), MinMaxScaler()
        sx.fit(tx);  sy.fit(ty)
        X_train, y_train = make_windows(sx.transform(tx), sy.transform(ty), W)
        X_test,  y_test  = make_windows(sx.transform(ex), sy.transform(ey), W)
        print(f"   window=60, train={s}행, test={len(df)-s}행")

    np.save('X_train.npy', X_train);  np.save('y_train.npy', y_train)
    np.save('X_test.npy',  X_test);   np.save('y_test.npy',  y_test)
    with open('scaler_x.pkl','wb') as f: pickle.dump(sx, f)
    with open('scaler_y.pkl','wb') as f: pickle.dump(sy, f)
    with open('metadata.pkl','wb') as f:
        pickle.dump({'feature_cols':FEATURE_COLS,'window_size':W,'csv_path':csv_path}, f)

    print(f"✅ X_train={X_train.shape}  X_test={X_test.shape}")
    return True


if __name__ == "__main__":
    preprocess_ultimate(sys.argv[1] if len(sys.argv) > 1 else 'week_traffic.csv')
