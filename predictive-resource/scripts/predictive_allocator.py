import csv
import time
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PLAN_CSV = BASE_DIR / "data" / "output" / "resource_allocation_plan.csv"
OUTPUT_CSV = BASE_DIR / "data" / "output" / "predictive_allocation_log.csv"

MIN_REPLICAS = 1
MAX_REPLICAS = 5
MIN_CPU = 0.5

IMAGE_NAME = "reactive-server"
CONTAINER_PREFIX = "app_server_"
BASE_PORT = 8001


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


def update_cpu(name: str, cpu_value: float):
    result = run_cmd(["docker", "update", f"--cpus={cpu_value}", name])
    require_success(result, f"docker update {name}")


def start_container(index: int, cpu_value: float):
    name = container_name(index)
    port = host_port(index)
    if container_exists(name):
        update_cpu(name, cpu_value)
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
    for i in range(1, target_replicas + 1):
        start_container(i, target_cpu)
    for i in range(MAX_REPLICAS, target_replicas, -1):
        remove_container(i)


def init_csv():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "elapsed_sec", "time_sec",
            "predicted_rps", "planned_cpu", "planned_replicas", "action",
        ])


def append_csv(row: list):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


class AllocationState:
    def __init__(self):
        self.current_cpu: float = MIN_CPU
        self.current_replicas: int = MIN_REPLICAS
        self.predicted_rps: float = 0.0
        self.last_updated: float = 0.0

    def update(self, cpu: float, replicas: int, pred_rps: float):
        self.current_cpu = cpu
        self.current_replicas = replicas
        self.predicted_rps = pred_rps
        self.last_updated = time.time()


class PredictiveAllocator:
    """
    미리 생성된 자원 계획 CSV를 읽어 시간축에 맞춰 선제 할당을 수행한다.
    실행 중 실제 오차는 HybridController가 따로 보정한다.
    """

    def __init__(self, plan_csv: Path, state: AllocationState):
        self.plan_csv = plan_csv
        self.state = state
        self.plan = self._load_plan()
        self._prev_cpu = MIN_CPU
        self._prev_reps = MIN_REPLICAS

    def _load_plan(self) -> list:
        if not self.plan_csv.exists():
            raise FileNotFoundError(f"자원 계획 CSV가 없습니다: {self.plan_csv}")

        rows = []
        with open(self.plan_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"time_sec", "predicted_rps", "planned_cpu", "planned_replicas"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"계획 CSV에 필요한 컬럼이 없습니다: {required}")

            for row in reader:
                rows.append({
                    "time_sec": int(float(row["time_sec"])),
                    "predicted_rps": float(row["predicted_rps"]),
                    "planned_cpu": float(row["planned_cpu"]),
                    "planned_replicas": int(float(row["planned_replicas"])),
                })
        return rows

    def run(self):
        init_csv()
        start_time = time.time()

        print("[PredictiveAllocator] 계획 기반 선제 할당 시작")
        apply_allocation(MIN_REPLICAS, MIN_CPU)
        self.state.update(MIN_CPU, MIN_REPLICAS, 0.0)

        for row in self.plan:
            target_sec = row["time_sec"]
            pred_rps = row["predicted_rps"]
            cpu = row["planned_cpu"]
            replicas = row["planned_replicas"]

            while True:
                if time.time() - start_time >= target_sec:
                    break
                time.sleep(0.05)

            elapsed = time.time() - start_time
            action = "HOLD"

            if cpu != self._prev_cpu or replicas != self._prev_reps:
                try:
                    apply_allocation(replicas, cpu)
                    action = "APPLY"
                    self._prev_cpu = cpu
                    self._prev_reps = replicas
                except Exception as e:
                    print(f"  [ERROR] Docker 적용 실패: {e}")
                    action = "ERROR"

            self.state.update(cpu, replicas, pred_rps)

            print(
                f"[PLAN] t={target_sec:>4}s | "
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

        print("[PredictiveAllocator] 계획 실행 완료")


if __name__ == "__main__":
    state = AllocationState()
    PredictiveAllocator(DEFAULT_PLAN_CSV, state).run()
