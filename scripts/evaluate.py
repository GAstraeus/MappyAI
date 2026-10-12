"""Headless evaluation of Mappy checkpoints: scores, rounds reached, optional video.

Unlike watch.py this needs no display and no ffmpeg (video is written with
OpenCV), so it runs on a headless box.

Usage:
  python scripts/evaluate.py checkpoints/ppo_mappy_final.zip
  python scripts/evaluate.py checkpoints/ppo_mappy_final.zip --episodes 5 \
      --video eval/final.mp4 --frames eval/final_frames.png
  python scripts/evaluate.py checkpoints/ppo_mappy_{1000000,5000000}_steps.zip \
      --episodes 3 --media-dir eval/media --json eval/results.json
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import mappy_env

STEP_SECONDS = 4 / 60.0  # frame-skip 4 at 60 fps
ACTION_NAMES = ["+".join(c) if c else "NOOP" for c in mappy_env.ACTION_COMBOS]


def run_episode(model, venv, deterministic, frame_sink=None):
    raw = venv.envs[0].unwrapped
    obs = venv.reset()
    done = False
    steps = 0
    deaths = 0
    max_round = 0
    first_point = None
    prev_lives = None
    prev_score = 0
    actions = collections.Counter()
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        actions[int(action[0])] += 1
        obs, reward, dones, infos = venv.step(action)
        info = infos[0]
        steps += 1
        if frame_sink is not None and not dones[0]:
            frame_sink(raw.get_screen())  # after done the VecEnv has already reset
        lives = int(info["lives"])
        if prev_lives is not None and lives < prev_lives:
            deaths += 1
        prev_lives = lives
        max_round = max(max_round, int(info["round"]))
        score = int(info["score"])
        if first_point is None and score > prev_score:
            first_point = steps
        prev_score = score
        done = bool(dones[0])
    truncated = bool(info.get("TimeLimit.truncated", False))
    if not truncated:
        deaths += 1  # the game-over death: lives jumps 0 -> 3 instead of decrementing
    return {
        "score": int(info["score"]),
        "steps": steps,
        "seconds": steps * STEP_SECONDS,
        "max_round": max_round,
        "deaths": deaths,
        "truncated": truncated,
        "first_point_step": first_point,
        "actions": {ACTION_NAMES[a]: n for a, n in sorted(actions.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--video", help="write episode 1 of the first checkpoint to this .mp4")
    parser.add_argument("--frames", help="write a strip of 8 frames from episode 1 to this .png")
    parser.add_argument("--media-dir", help="write <checkpoint>.mp4 and <checkpoint>_frames.png "
                                            "from episode 1 of every checkpoint into this directory")
    parser.add_argument("--json", help="append per-episode results to this JSON file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(4)
    venv = VecTransposeImage(
        VecFrameStack(DummyVecEnv([mappy_env.make_env(train=False)]), n_stack=4)
    )
    venv.seed(args.seed)

    all_results = {}
    for ci, ckpt in enumerate(args.checkpoints):
        model = PPO.load(ckpt, device="cpu")
        stem = os.path.splitext(os.path.basename(ckpt))[0] + ("_det" if args.deterministic else "")
        results = []
        for ep in range(args.episodes):
            video = frames_path = None
            if ep == 0 and args.media_dir:
                video = os.path.join(args.media_dir, f"{stem}.mp4")
                frames_path = os.path.join(args.media_dir, f"{stem}_frames.png")
            elif ep == 0 and ci == 0:
                video, frames_path = args.video, args.frames
            frames = []
            r = run_episode(model, venv, args.deterministic, frames.append if (video or frames_path) else None)
            results.append(r)
            acts = ", ".join(f"{k}={n / r['steps']:.0%}" for k, n in r["actions"].items())
            print(
                f"{os.path.basename(ckpt)} ep{ep + 1}: score {r['score']:>6}  "
                f"steps {r['steps']:>4} ({r['seconds']:.0f}s)  round {r['max_round']}  "
                f"deaths {r['deaths']}  {'TIMEOUT' if r['truncated'] else 'GAMEOVER'}  "
                f"first point @ step {r['first_point_step']}  actions: {acts}",
                flush=True,
            )
            if frames:
                save_media(frames, video, frames_path)
        scores = np.array([r["score"] for r in results])
        steps = np.array([r["steps"] for r in results])
        print(
            f"== {os.path.basename(ckpt)}: score mean {scores.mean():.0f} ± {scores.std():.0f} "
            f"(min {scores.min()}, max {scores.max()}), steps mean {steps.mean():.0f}, "
            f"max round {max(r['max_round'] for r in results)}, "
            f"timeouts {sum(r['truncated'] for r in results)}/{len(results)}",
            flush=True,
        )
        all_results[stem] = {
            "checkpoint": ckpt,
            "num_timesteps": int(model.num_timesteps),
            "deterministic": args.deterministic,
            "seed": args.seed,
            "episodes": results,
        }
    venv.close()

    if args.json:
        existing = {}
        if os.path.exists(args.json):
            with open(args.json) as f:
                existing = json.load(f)
        existing.update(all_results)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(existing, f, indent=1)
        print(f"wrote {args.json}")


def save_media(frames, video_path, frames_path):
    import cv2

    if video_path:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        h, w = frames[0].shape[:2]
        fps = 1 / STEP_SECONDS  # one wrapped step per video frame -> real time
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h * 2))
        for f in frames:
            out.write(cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (w * 2, h * 2), interpolation=cv2.INTER_NEAREST))
        out.release()
        print(f"wrote {video_path} ({len(frames)} frames, {len(frames) * STEP_SECONDS:.0f}s)")
    if frames_path:
        os.makedirs(os.path.dirname(os.path.abspath(frames_path)), exist_ok=True)
        idx = np.linspace(0, len(frames) - 1, 8).astype(int)
        strip = np.concatenate([frames[i] for i in idx], axis=1)
        cv2.imwrite(frames_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print(f"wrote {frames_path} (steps {[int(i) for i in idx]})")


if __name__ == "__main__":
    main()
