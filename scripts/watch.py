"""Watch a trained agent play Mappy.

Usage:
  python scripts/watch.py checkpoints/ppo_mappy_final.zip                # live window
  python scripts/watch.py checkpoints/ppo_mappy_final.zip --episodes 5
  python scripts/watch.py checkpoints/ppo_mappy_final.zip --record       # .bk2 + .mp4

Recording writes .bk2 replays to recordings/ and converts them to .mp4 with
stable-retro's playback tool (needs ffmpeg on PATH).
"""

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import mappy_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="argmax actions (default: sample — deterministic tends to loop)")
    args = parser.parse_args()

    record_dir = os.path.join(mappy_env.PROJECT_DIR, "recordings")
    if args.record:
        os.makedirs(record_dir, exist_ok=True)

    venv = VecTransposeImage(
        VecFrameStack(
            DummyVecEnv([
                mappy_env.make_env(
                    train=False,
                    render_mode=None if args.record else "human",
                    record=record_dir if args.record else False,
                )
            ]),
            n_stack=4,
        )
    )
    model = PPO.load(args.checkpoint)

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = venv.step(action)
            total += float(reward[0])
            done = bool(dones[0])
        print(f"episode {ep + 1}: reward {total:.0f}")
    venv.close()

    if args.record:
        for bk2 in sorted(glob.glob(os.path.join(record_dir, "*.bk2"))):
            print(f"converting {bk2} ...")
            subprocess.run(
                [sys.executable, "-m", "stable_retro.scripts.playback_movie", bk2],
                check=False,
            )
        print(f"recordings in {record_dir}")


if __name__ == "__main__":
    main()
