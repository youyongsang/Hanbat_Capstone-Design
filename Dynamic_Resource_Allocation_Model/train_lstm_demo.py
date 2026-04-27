# -*- coding: utf-8 -*-
import os, sys, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

CONFIGS = {
    'sale_event': {
        'units': (128, 64), 
        'dense': (64, 32), 
        'batch': 32, 
        'lr': 0.0004,     # 💡 학습률을 살짝 낮춰 더 정밀하게 고점 탐색
        'patience': 25, 
        'peak_q': 0.40,   # 💡 타겟 구간을 좁혀서 고점에 더 집중
        'peak_w': 15.0    # 💡 고점 가중치를 15배로 강화 (400초대 피크 견인)
    },
    'week_traffic': {
        'units': (64, 32), 'dense': (32, 16), 'batch': 256, 'lr': 0.0008, 
        'patience': 15, 'peak_q': 0.70, 'peak_w': 2.5
    },
    'default': {
        'units': (64, 32), 'dense': (32, 16), 'batch': 128, 'lr': 0.001, 
        'patience': 10, 'peak_q': 0.70, 'peak_w': 2.5
    },
}

def get_cfg(csv_path=''):
    fname = os.path.basename(str(csv_path)).lower()
    for k in CONFIGS:
        if k in fname: return CONFIGS[k]
    return CONFIGS['default']

# 💡 피크는 더 높게, 과잉 하락은 방지하는 [강화형 듀얼 로스]
def super_enhanced_loss(y_train, q, w):
    thr = float(np.quantile(y_train, q))
    def loss(y_true, y_pred):
        err = tf.square(y_true - y_pred)
        
        # 1. 고점 강화: 실제보다 낮게 예측하면 아주 강력한 벌점 (400초대 해결)
        wt_high = tf.where((y_true > thr) & (y_true > y_pred), 
                           tf.ones_like(y_true) * w * 1.5, # 가중치의 1.5배 추가 적용
                           tf.ones_like(y_true))
        
        # 2. 과잉 하락 방지: 실제보다 더 낮게 예측할 때 벌점 부여 (700초대 해결)
        # 모델이 실제값보다 0.03 이상 더 아래로 내려가면 패널티
        wt_low = tf.where((y_true < y_pred - 0.03), 
                          tf.ones_like(y_true) * 5.0, 
                          tf.ones_like(y_true))
        
        # 고점 가중치와 하락 방지 가중치 중 강한 놈 적용
        final_wt = tf.maximum(wt_high, wt_low)
        return tf.reduce_mean(err * final_wt)
    
    loss.__name__ = 'super_enhanced_mse'
    return loss

def train_ultimate(csv_path=None):
    if csv_path is None and os.path.exists('metadata.pkl'):
        with open('metadata.pkl','rb') as f:
            csv_path = pickle.load(f).get('csv_path','')

    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    cfg = get_cfg(csv_path)
    u1, u2 = cfg['units']; d1, d2 = cfg['dense']
    ws, nf = X_train.shape[1], X_train.shape[2]

    print(f"🚀 [최종 강화 모드] 학습 시작 | peak_w={cfg['peak_w']}")

    model = Sequential([
        Bidirectional(LSTM(u1, return_sequences=True), input_shape=(ws, nf)),
        Dropout(0.15),
        Bidirectional(LSTM(u2, return_sequences=False)),
        Dropout(0.1),
        Dense(d1, activation='relu'),
        Dense(d2, activation='relu'),
        Dense(1)
    ])
    
    # 💡 강화된 로스 함수 적용
    model.compile(optimizer=Adam(cfg['lr']), 
                  loss=super_enhanced_loss(y_train, cfg['peak_q'], cfg['peak_w']))
    
    cbs = [
        EarlyStopping(monitor='val_loss', patience=cfg['patience'], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=cfg['patience']//2, min_lr=1e-6, verbose=1),
        ModelCheckpoint('lstm_model.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    model.fit(X_train, y_train, epochs=200, batch_size=cfg['batch'],
              validation_split=0.15, shuffle=True, callbacks=cbs, verbose=1)
    
    print("\n✅ 강화 학습 완료! 다시 한번 결과를 확인해봅시다.")

if __name__ == "__main__":
    train_ultimate(sys.argv[1] if len(sys.argv) > 1 else None)
