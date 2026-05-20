from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    CURRICULUM_STAGES,
    DEFAULT_SEED,
    DEFAULT_TOTAL_TIMESTEPS,
    IMITATION_BATCH_SIZE,
    IMITATION_ACTION_WEIGHTS,
    IMITATION_EPOCHS,
    TRAINED_MODEL_PATH,
)
from data_utils import build_teacher_dataset, load_merged_schedule
from resource_env import ResourceAllocationEnv


def pretrain_with_teacher(model, teacher_obs, teacher_actions, epochs: int, batch_size: int, seed: int):
    import torch
    import torch.nn.functional as F

    if len(teacher_obs) == 0:
        return

    policy = model.policy
    device = policy.device
    optimizer = policy.optimizer
    obs_tensor = torch.as_tensor(teacher_obs, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(teacher_actions, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    action_counts = np.bincount(teacher_actions, minlength=model.action_space.n)
    class_weights = np.ones(model.action_space.n, dtype=np.float32)
    nonzero_counts = action_counts[action_counts > 0]
    if len(nonzero_counts) > 0:
        base_scale = float(len(teacher_actions)) / float(len(nonzero_counts))
        for action_id, count in enumerate(action_counts):
            if count > 0:
                class_weights[action_id] = base_scale / float(count)
    for action_id, multiplier in IMITATION_ACTION_WEIGHTS.items():
        class_weights[action_id] *= float(multiplier)
    class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    print(f"[IMITATION] samples={len(teacher_obs)} epochs={epochs} batch_size={batch_size}")
    print(f"[IMITATION] action_counts={action_counts.tolist()}")
    for epoch in range(epochs):
        indices = rng.permutation(len(teacher_obs))
        losses = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_obs = obs_tensor[batch_indices]
            batch_actions = action_tensor[batch_indices]
            distribution = policy.get_distribution(batch_obs)
            logits = distribution.distribution.logits
            loss = F.cross_entropy(logits, batch_actions, weight=class_weights_tensor)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[IMITATION] epoch={epoch + 1}/{epochs} loss={avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="PPO 기반 자원 할당 정책 학습")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-curriculum", action="store_true", help="단일 full 구간으로만 학습")
    parser.add_argument("--no-imitation", action="store_true", help="teacher imitation warm-start를 건너뜀")
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

        if model is not None and not args.no_imitation and stage_name == stages[0][0]:
            try:
                teacher_obs, teacher_actions = build_teacher_dataset()
                pretrain_with_teacher(
                    model,
                    teacher_obs=teacher_obs,
                    teacher_actions=teacher_actions,
                    epochs=IMITATION_EPOCHS,
                    batch_size=IMITATION_BATCH_SIZE,
                    seed=args.seed,
                )
            except Exception as exc:
                print(f"[IMITATION] warm-start를 건너뜁니다: {exc}")

        print(f"[TRAIN] stage={stage_name} steps={stage_steps}")
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=False)

    TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(TRAINED_MODEL_PATH)
    print(f"[DONE] 학습 완료: {TRAINED_MODEL_PATH}")


if __name__ == "__main__":
    main()
