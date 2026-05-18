from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import CURRICULUM_STAGES, DEFAULT_SEED, DEFAULT_TOTAL_TIMESTEPS, TRAINED_MODEL_PATH
from data_utils import load_merged_schedule
from resource_env import ResourceAllocationEnv


def main():
    parser = argparse.ArgumentParser(description="PPO 기반 자원 할당 정책 학습")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-curriculum", action="store_true", help="단일 full 구간으로만 학습")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3가 필요합니다. requirements.txt를 먼저 설치하세요.") from exc

    schedule_df = load_merged_schedule()

    stages = [("full", 1.0)] if args.no_curriculum else CURRICULUM_STAGES
    model = None
    for stage_name, stage_ratio in stages:
        stage_steps = max(1, int(args.timesteps * stage_ratio))
        env = DummyVecEnv([lambda stage=stage_name: ResourceAllocationEnv(schedule_df, mode=stage)])
        if model is None:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                seed=args.seed,
                n_steps=256,
                batch_size=64,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
            )
        else:
            model.set_env(env)

        print(f"[TRAIN] stage={stage_name} steps={stage_steps}")
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=False)

    TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(TRAINED_MODEL_PATH)
    print(f"[DONE] 학습 완료: {TRAINED_MODEL_PATH}")


if __name__ == "__main__":
    main()
