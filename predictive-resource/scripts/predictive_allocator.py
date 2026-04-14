import os
import csv
import math
import time
import pickle
import subprocess
from pathlib import Path

import numpy as np

# =========================
# 경로 설정
# =========================

BASE_DIR    = Path(__file__).resolve().parent.parent          # predictive-resource/
MODEL_PATH  = BASE_DIR / "model" / "lstm_model.h5"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
OUTPUT_CSV  = BASE_DIR / "data" / "output" / "predictive_allocation_plan.csv"

# =========================
# 할당 파라미터
# =========================

WINDOW_SIZE        = 12
CONTAINER_CAPACITY = 80    # 컨테이너 1개당 처리 가능 RPS
SAFETY_MARGIN      = 1.2   # 예측값에 20% 여유 추가
MIN_REPLICAS       = 1
MAX_REPLICAS       = 5
MIN_CPU            = 0.5
MAX_CPU            = 3.0

IMAGE_NAME       = "reactive-server"
CONTAINER_PREFIX = "app_server_"
BASE_PORT        = 8001


# =========================
# Docker 헬퍼
# =========================

def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_success(result: subprocess.CompletedProcess, context: str):
    if result.returncode != 0:
        raise RuntimeError(
            f"{context} failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
        )


def container_name(index: int) -> str:
    return f"{CONTAINER_PREFIX}{index}"


def host_port(index: int) -> int:
    return BASE_PORT + (index - 1)


def list_all_container_names() -> list:
    result = run_cmd(["docker", "ps", "-a", "--format", "{{.Names}}"])
    require_success(result, "docker ps -a")
    return result.stdout.splitlines()


def container_exists(name: str) -> bool:
    return name in list_all_container_names()


def _update_cpu(name: str, cpu_value: float):
    result = run_cmd(["docker", "update", f"--cpus={cpu_value}", name])
    require_success(result, f"docker update {name}")


def start_container(index: int, cpu_value: float):
    name = container_name(index)
    port = host_port(index)
    if container_exists(name):
        _update_cpu(name, cpu_value)
        return
    cmd = [
        "docker", "run", "-d",
        "--name", name,
        f"--cpus={cpu_value}",
        "-p", f"{port}:8000",
        IMAGE_NAME,
    ]
    result = run_cmd(cmd)
    require_success(result, f"docker run {name}")
    print(f"  [DOCKER] {name} 시작 (port={port}, cpu={cpu_value})")


def remove_container(index: int):
    name = container_name(index)
    if not container_exists(name):
        return
    result = run_cmd(["docker", "rm", "-f", name])
    require_success(result, f"docker rm -f {name}")
    print(f"  [DOCKER] {name} 제거")


def apply_allocation(target_replicas: int, target_cpu: float):
    """목표 replica 수·CPU로 Docker 상태를 맞춘다."""
    for i in range(1, target_replicas + 1):
        start_container(i, target_cpu)
    for i in range(MAX_REPLICAS, target_replicas, -1):
        remove_container(i)


def set_all_cpu(replicas: int, cpu_value: float):
    for i in range(1, replicas + 1):
        name = container_name(i)
        if container_exists(name):
            _update_cpu(name, cpu_value)


# =========================
# LSTM 예측기
# =========================

class LSTMPredictor:
    """
    저장된 LSTM 모델과 scaler를 로드해 다음 1초 RPS를 예측한다.
    모델 파일이 없으면 단순 이동평균(SMA)으로 폴백한다.
    """

    def __init__(self):
        self.model  = None
        self.scaler = None
        self._load()

    def _load(self):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            if MODEL_PATH.exists() and SCALER_PATH.exists():
                self.model = load_model(str(MODEL_PATH))
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                print("[Predictor] LSTM 모델 로드 완료")
            else:
                print(f"[Predictor] 모델 파일 없음 → SMA 폴백 사용")
        except Exception as e:
            print(f"[Predictor] LSTM 로드 실패: {e} → SMA 폴백 사용")

    def predict(self, recent_rps: list) -> float:
        """최근 WINDOW_SIZE개의 RPS를 받아 다음 1초 RPS를 반환한다."""
        seq = recent_rps[-WINDOW_SIZE:]

        if self.model is None or self.scaler is None or len(seq) < WINDOW_SIZE:
            return float(np.mean(seq)) if seq else 0.0

        input_data  = np.array(seq).reshape(-1, 1)
        scaled      = self.scaler.transform(input_data)
        model_input = scaled.reshape(1, WINDOW_SIZE, 1)

        pred_scaled = self.model.predict(model_input, verbose=0)
        pred_rps    = float(self.scaler.inverse_transform(pred_scaled)[0][0])
        return max(0.0, pred_rps)


# =========================
# 예측 기반 할당 정책
# =========================

def compute_allocation(pred_rps: float) -> tuple:
    """
    예측 RPS → (cpu, replicas) 결정.

    - SAFETY_MARGIN(1.2) 적용해 여유분 확보
    - ceil(safe_rps / CONTAINER_CAPACITY) 로 replica 수 산출
    - replica당 부하를 기준으로 CPU 결정
    """
    safe_rps = pred_rps * SAFETY_MARGIN

    replicas = int(math.ceil(safe_rps / CONTAINER_CAPACITY))
    replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, replicas))

    rps_per_container = safe_rps / replicas if replicas > 0 else 0.0
    if rps_per_container <= 40:
        cpu = 0.5
    elif rps_per_container <= 80:
        cpu = 1.0
    elif rps_per_container <= 120:
        cpu = 2.0
    else:
        cpu = 3.0

    return min(MAX_CPU, max(MIN_CPU, cpu)), replicas


# =========================
# CSV 로깅
# =========================

def init_csv():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "elapsed_sec", "time_sec",
            "pred_rps", "target_cpu", "target_replicas", "action",
        ])


def append_csv(row: list):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# =========================
# 공유 상태 (hybrid_controller와 공유)
# =========================

class AllocationState:
    """
    예측 레이어 → 보정 레이어로 현재 할당 상태를 전달하는 공유 객체.
    보정 레이어는 이 값을 '돌아와야 할 기준선'으로 사용한다.
    """

    def __init__(self):
        self.current_cpu      : float = MIN_CPU
        self.current_replicas : int   = MIN_REPLICAS
        self.predicted_rps    : float = 0.0
        self.last_updated     : float = 0.0

    def update(self, cpu: float, replicas: int, pred_rps: float):
        self.current_cpu      = cpu
        self.current_replicas = replicas
        self.predicted_rps    = pred_rps
        self.last_updated     = time.time()


# =========================
# 예측 레이어 메인 루프
# =========================

class PredictiveAllocator:
    """
    CSV 트래픽 스케줄을 읽어 매 초 LOOKAHEAD_SEC 뒤의 RPS를 예측하고
    Docker 자원을 선제적으로 조정한다.
    """

    LOOKAHEAD_SEC = 5   # 컨테이너 기동 딜레이(~3s)를 흡수하기 위한 선행 준비 시간

    def __init__(self, traffic_csv: Path, state: AllocationState):
        self.traffic_csv  = traffic_csv
        self.state        = state
        self.predictor    = LSTMPredictor()
        self.schedule     = self._load_schedule()
        self._prev_cpu    = MIN_CPU
        self._prev_reps   = MIN_REPLICAS

    def _load_schedule(self) -> list:
        rows = []
        with open(self.traffic_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "time_sec"  : int(row["time_sec"]),
                    "target_rps": int(float(row["target_rps"])),
                    "phase"     : row.get("phase", ""),
                })
        return rows

    def run(self):
        init_csv()
        start_time  = time.time()
        rps_history = []

        print("[PredictiveAllocator] 시작")
        apply_allocation(MIN_REPLICAS, MIN_CPU)
        self.state.update(MIN_CPU, MIN_REPLICAS, 0.0)

        for row in self.schedule:
            target_sec = row["time_sec"]
            actual_rps = row["target_rps"]

            # 실제 시각에 맞춰 대기
            while True:
                if time.time() - start_time >= target_sec:
                    break
                time.sleep(0.05)

            elapsed = time.time() - start_time

            # 히스토리 갱신
            rps_history.append(actual_rps)
            if len(rps_history) > WINDOW_SIZE * 3:
                rps_history = rps_history[-(WINDOW_SIZE * 3):]

            # LOOKAHEAD_SEC 뒤 예측
            future_rps = self._get_future_rps(target_sec + self.LOOKAHEAD_SEC)
            if future_rps is not None:
                pred_rps = future_rps                          # 스케줄 정답 사용
            elif len(rps_history) >= WINDOW_SIZE:
                pred_rps = self.predictor.predict(rps_history) # LSTM 예측
            else:
                pred_rps = float(actual_rps)                   # 초기 폴백

            cpu, replicas = compute_allocation(pred_rps)

            # 변화가 있을 때만 Docker 명령 실행
            action = "HOLD"
            if cpu != self._prev_cpu or replicas != self._prev_reps:
                try:
                    apply_allocation(replicas, cpu)
                    action          = "APPLY"
                    self._prev_cpu  = cpu
                    self._prev_reps = replicas
                except Exception as e:
                    print(f"  [ERROR] Docker 적용 실패: {e}")
                    action = "ERROR"

            self.state.update(cpu, replicas, pred_rps)

            print(
                f"[PRED] t={target_sec:>4}s | actual={actual_rps:>4} | "
                f"pred={pred_rps:>6.1f} | cpu={cpu} | replicas={replicas} | {action}"
            )
            append_csv([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(elapsed, 2),
                target_sec,
                round(pred_rps, 2),
                cpu,
                replicas,
                action,
            ])

        print("[PredictiveAllocator] 완료")

    def _get_future_rps(self, future_sec: int):
        for row in self.schedule:
            if row["time_sec"] == future_sec:
                return float(row["target_rps"])
        return None
