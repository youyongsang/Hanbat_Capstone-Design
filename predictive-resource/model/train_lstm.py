import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential          # type: ignore
from tensorflow.keras.layers import LSTM, Dense        # type: ignore
from tensorflow.keras.callbacks import EarlyStopping   # type: ignore

# =========================
# 경로 설정
# =========================

MODEL_DIR = Path(__file__).resolve().parent   # predictive-resource/model/

def train():
    X_train_path = MODEL_DIR / "X_train.npy"
    if not X_train_path.exists():
        raise FileNotFoundError(
            "학습 데이터 없음. 먼저 preprocess_lstm.py를 실행하세요."
        )

    X_train = np.load(MODEL_DIR / "X_train.npy")
    y_train = np.load(MODEL_DIR / "y_train.npy")
    X_val   = np.load(MODEL_DIR / "X_val.npy")
    y_val   = np.load(MODEL_DIR / "y_val.npy")

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("학습 시작...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        shuffle=False,       # 시계열 순서 유지
        callbacks=[early_stop],
        verbose=1,
    )

    save_path = MODEL_DIR / "lstm_model.h5"
    model.save(str(save_path))
    print(f"모델 저장 완료: {save_path}")


if __name__ == "__main__":
    train()
