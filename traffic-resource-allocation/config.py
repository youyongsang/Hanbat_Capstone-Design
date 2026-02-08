import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "traffic_log.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

INTERVAL_SEC = 5
WINDOW_SIZE = 12
NUM_NODES = 10
NUM_FEATURES = 1
PRED_HORIZON = 1

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
TOTAL_SAMPLES = 5000

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

TOTAL_CAPACITY = 100
SCALE_OUT_TH = 0.8
SCALE_IN_TH = 0.2
