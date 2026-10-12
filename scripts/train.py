"""Train PPO on Mappy pixels. Adapted from stable_retro/examples/ppo.py.

Every invocation writes into its own run directory, runs/<run-name>/:

  config.json           args, hyperparameters, env description, git commit
  logs/                 SB3 logger: log.txt (console tables), progress.csv, tensorboard
  checkpoints/          ppo_mappy_<steps>_steps.zip every --checkpoint-every steps
  best/best_model.zip   best mean raw score on the periodic evaluation (--eval-every)
  eval/evaluations.npz  per-evaluation scores and episode lengths
  final_model.zip       the model when training ends

Usage:
  python scripts/train.py --run-name clipreward_v2            # 10M steps, 8 workers
  python scripts/train.py --run-name big --num-envs 32 --timesteps 2e7
  python scripts/train.py --run-name clipreward_v2 --resume runs/clipreward_v2/final_model.zip

Afterwards: python scripts/archive_run.py runs/<run-name>    # curve, videos, tables, RUN.md
Watch live: tensorboard --logdir runs
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import mappy_env

HYPERPARAMS = dict(
    policy="CnnPolicy",
    learning_rate=lambda f: f * 2.5e-4,  # linear decay to 0
    n_steps=128,
    batch_size=32,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
)


def make_vec(num_envs, **env_kwargs):
    return VecTransposeImage(
        VecFrameStack(SubprocVecEnv([mappy_env.make_env(**env_kwargs)] * num_envs), n_stack=4)
    )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty"], cwd=mappy_env.PROJECT_DIR, text=True
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None,
                        help="folder name under runs/ (default: timestamp)")
    parser.add_argument("--timesteps", type=float, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--resume", default=None, help="model .zip to continue from")
    parser.add_argument("--checkpoint-every", type=int, default=500_000,
                        help="total env steps between checkpoints")
    parser.add_argument("--eval-every", type=int, default=250_000,
                        help="total env steps between evaluations that track best_model")
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(mappy_env.PROJECT_DIR, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    venv = make_vec(args.num_envs, train=True)
    # Evaluation env: same observations, Monitor on, no reward clipping -> raw score.
    eval_env = make_vec(1, train=False, monitor=True)

    if args.resume:
        model = PPO.load(args.resume, env=venv)
        print(f"resumed from {args.resume} at {model.num_timesteps} steps")
        log_dir = os.path.join(run_dir, f"logs_from_{model.num_timesteps}")
        config_path = os.path.join(run_dir, f"config_resume_{model.num_timesteps}.json")
    else:
        model = PPO(env=venv, verbose=1, **HYPERPARAMS)
        log_dir = os.path.join(run_dir, "logs")
        config_path = os.path.join(run_dir, "config.json")
    model.set_logger(configure(log_dir, ["stdout", "log", "csv", "tensorboard"]))

    config = {
        "run_name": run_name,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "hyperparams": {k: (v if not callable(v) else "linear 2.5e-4 -> 0") for k, v in HYPERPARAMS.items()},
        "env": mappy_env.ENV_DESCRIPTION,
        "eval": {"episodes": args.eval_episodes, "deterministic": False, "metric": "mean raw score"},
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=1)

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.num_envs, 1),
            save_path=os.path.join(run_dir, "checkpoints"),
            name_prefix="ppo_mappy",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best"),
            log_path=os.path.join(run_dir, "eval"),
            eval_freq=max(args.eval_every // args.num_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=False,  # watch.py samples too; argmax policies tend to loop
            render=False,
        ),
    ]
    print(f"run dir: {run_dir}")
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=callbacks,
        log_interval=1,
        reset_num_timesteps=args.resume is None,
    )
    final_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_path)
    print(f"saved {final_path}")
    eval_env.close()
    venv.close()


if __name__ == "__main__":
    main()
