"""Train PPO on Mappy pixels. Adapted from stable_retro/examples/ppo.py.

Usage:
  python scripts/train.py                          # 10M steps, 8 workers
  python scripts/train.py --timesteps 2e6 --num-envs 4
  python scripts/train.py --resume checkpoints/ppo_mappy_1000000_steps.zip

Monitor with: tensorboard --logdir tb_logs
Expect hours for multi-million-step runs on a laptop; checkpoints land in
checkpoints/ every 100k total steps (save_freq is divided by num_envs).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import mappy_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=float, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--resume", default=None, help="checkpoint .zip to continue from")
    args = parser.parse_args()

    venv = VecTransposeImage(
        VecFrameStack(
            SubprocVecEnv([mappy_env.make_env(train=True)] * args.num_envs),
            n_stack=4,
        )
    )

    if args.resume:
        model = PPO.load(args.resume, env=venv, tensorboard_log="tb_logs")
        print(f"resumed from {args.resume} at {model.num_timesteps} steps")
    else:
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            learning_rate=lambda f: f * 2.5e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="tb_logs",
        )

    checkpoint = CheckpointCallback(
        save_freq=max(100_000 // args.num_envs, 1),
        save_path="checkpoints",
        name_prefix="ppo_mappy",
    )
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=checkpoint,
        log_interval=1,
        reset_num_timesteps=args.resume is None,
    )
    model.save("checkpoints/ppo_mappy_final")
    print("saved checkpoints/ppo_mappy_final.zip")


if __name__ == "__main__":
    main()
