import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONTROLLER = BASE_DIR / "scripts" / "hybrid_reactive_controller.py"
LOAD_GENERATOR = BASE_DIR / "scripts" / "hybrid_load_generator.py"

CONTROLLER_STARTUP_DELAY_SEC = 5


def terminate_process(process: subprocess.Popen):
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    print("[Main] Hybrid Reactive Controller start")
    controller = subprocess.Popen([sys.executable, str(CONTROLLER)], cwd=BASE_DIR)

    try:
        print(f"[Main] Waiting {CONTROLLER_STARTUP_DELAY_SEC}s for container startup")
        time.sleep(CONTROLLER_STARTUP_DELAY_SEC)

        if controller.poll() is not None:
            print(f"[ERROR] Controller exited early with code {controller.returncode}")
            sys.exit(controller.returncode or 1)

        print("[Main] Load Generator start")
        load_generator = subprocess.Popen([sys.executable, str(LOAD_GENERATOR)], cwd=BASE_DIR)
        load_code = load_generator.wait()

        print(f"[Main] Load Generator finished with code {load_code}")
        if load_code != 0:
            sys.exit(load_code)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted")
        sys.exit(130)
    finally:
        print("[Main] Stopping Hybrid Reactive Controller")
        terminate_process(controller)

    print("[Main] Hybrid Reactive experiment finished")


if __name__ == "__main__":
    main()
