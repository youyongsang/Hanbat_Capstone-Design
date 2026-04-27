# -*- coding: utf-8 -*-
import os, sys, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

CONFIGS = {
    'sale_event':  {'units':(128,64), 'dense':(64,32), 'batch':32,  'lr':0.0005, 'patience':15, 'peak_q':0.65, 'peak_w':3.0},
    'week_traffic':{'units':(64,32),  'dense':(32,16), 'batch':256, 'lr':0.0008, 'patience':15, 'peak_q':0.70, 'peak_w':2.5},
    'default':     {'units':(64,32),  'dense':(32,16), 'batch':128, 'lr':0.001,  'patience':10, 'peak_q':0.70, 'peak_w':2.5},
}

def get_cfg(csv_path=''):
    fname = os.path.basename(str(csv_path)).lower()
    for k in CONFIGS:
        if k in fname: return CONFIGS[k]
    return CONFIGS['default']

def peak_loss(y_train, q, w):
    thr = float(np.quantile(y_train, q))
    def loss(y_true, y_pred):
        err = tf.square(y_true - y_pred)
        wt  = tf.where(y_true > thr, tf.ones_like(y_true)*w, tf.ones_like(y_true))
        return tf.reduce_mean(err * wt)
    loss.__name__ = 'peak_mse'
    return loss

def train_ultimate(csv_path=None):
    if csv_path is None and os.path.exists('metadata.pkl'):
        with open('metadata.pkl','rb') as f:
            csv_path = pickle.load(f).get('csv_path','')

    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except FileNotFoundError:
        print("❌ X_train.npy 없음. preprocess_lstm.py 먼저 실행하세요."); return

    cfg = get_cfg(csv_path)
    u1, u2 = cfg['units']
    d1, d2 = cfg['dense']
    ws, nf = X_train.shape[1], X_train.shape[2]

    print(f"🚀 학습 시작 | units=({u1},{u2}) batch={cfg['batch']} lr={cfg['lr']}")

    model = Sequential([
        Bidirectional(LSTM(u1, return_sequences=True), input_shape=(ws, nf)),
        Dropout(0.2),
        Bidirectional(LSTM(u2, return_sequences=False)),
        Dropout(0.1),
        Dense(d1, activation='relu'),
        Dense(d2, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(cfg['lr']),
                  loss=peak_loss(y_train, cfg['peak_q'], cfg['peak_w']))

    cbs = [
        EarlyStopping(monitor='val_loss', patience=cfg['patience'],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=cfg['patience']//2, min_lr=1e-6, verbose=1),
        ModelCheckpoint('lstm_model.h5', monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    print(f"\n🧠 학습 중... (max 200 epochs, EarlyStopping patience={cfg['patience']})")
    hist = model.fit(X_train, y_train, epochs=200, batch_size=cfg['batch'],
                     validation_split=0.15, shuffle=True, callbacks=cbs, verbose=1)

    print(f"\n✅ 완료: {len(hist.history['loss'])} epochs | best val_loss={min(hist.history['val_loss']):.6f}")

if __name__ == "__main__":
    train_ultimate(sys.argv[1] if len(sys.argv) > 1 else None)
