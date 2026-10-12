"""Summarize a training run: learning curve, per-model evaluation with videos, tables, RUN.md.

Works on a run directory written by scripts/train.py (runs/<name>/) and on
manually assembled ones, as long as checkpoints live in <run>/checkpoints/ and
tensorboard event files are somewhere below the directory.

Usage:
  python scripts/archive_run.py runs/<run-name>                 # 4 milestones + best + final, 3 episodes each
  python scripts/archive_run.py runs/<run-name> --episodes 5 --milestones 6
  python scripts/archive_run.py runs/<run-name> --episodes 0    # curve and RUN.md only

Writes into the run directory:
  learning_curve.png          training score / length with periodic evaluations overlaid
  eval/results.json           per-episode detail for every evaluated model
  eval/summary.md             one row per model
  eval/media/<model>.mp4      episode 1 of each model at real-time speed, plus an 8-frame strip
  eval/eval_<model>.log       evaluate.py console output
  RUN.md                      setup from config.json + table + a Findings section to fill in
                              (created only if missing, so your notes are never overwritten)
Heavy files (mp4, zip, events, log.gz) are kept out of git by runs/.gitignore.
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

import plot_curve


def step_of(path):
    m = re.search(r"_(\d+)_steps\.zip$", path)
    return int(m.group(1)) if m else None


def pick_models(run_dir, milestones):
    ckpts = sorted(
        (p for p in glob.glob(os.path.join(run_dir, "checkpoints", "*.zip")) if step_of(p) is not None),
        key=step_of,
    )
    models = []
    if milestones and ckpts:
        n = min(milestones, len(ckpts))
        idx = sorted({round(i * (len(ckpts) - 1) / max(n - 1, 1)) for i in range(n)})
        models += [ckpts[i] for i in idx]
    for p in (
        os.path.join(run_dir, "best", "best_model.zip"),
        os.path.join(run_dir, "final_model.zip"),
        os.path.join(run_dir, "checkpoints", "ppo_mappy_final.zip"),  # pre-run-dir layout
    ):
        if os.path.exists(p) and p not in models:
            models.append(p)
    return models


def evaluate(run_dir, jobs, episodes):
    """jobs: list of (model_path, deterministic). Runs evaluate.py once per job, in parallel."""
    eval_dir = os.path.join(run_dir, "eval")
    media = os.path.join(eval_dir, "media")
    os.makedirs(media, exist_ok=True)

    def run(job):
        model, det = job
        stem = os.path.splitext(os.path.basename(model))[0] + ("_det" if det else "")
        part = os.path.join(eval_dir, f"results_{stem}.json")
        cmd = [sys.executable, os.path.join(SCRIPTS, "evaluate.py"), model,
               "--episodes", str(episodes), "--media-dir", media, "--json", part]
        if det:
            cmd += ["--deterministic", "--seed", "1"]
        with open(os.path.join(eval_dir, f"eval_{stem}.log"), "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        return part

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        parts = list(ex.map(run, jobs))
    results_path = os.path.join(eval_dir, "results.json")
    merged = json.load(open(results_path)) if os.path.exists(results_path) else {}
    for part in parts:
        if os.path.exists(part):
            merged.update(json.load(open(part)))
            os.remove(part)
        else:
            print(f"WARNING: no results for {part} (see the eval log next to it)")
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=1)
    return merged


def model_label(run_dir, path):
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(run_dir))
    return os.path.basename(path) if rel.startswith("..") else rel


def make_table(run_dir, merged):
    rows = [
        "| model | trained steps | policy | episodes | score mean ± sd | min / max | mean length (steps) | hit time cap | max round | deaths / ep |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for stem in sorted(merged, key=lambda s: (merged[s].get("num_timesteps") or 0, "det" in s, s)):
        r = merged[stem]
        eps = r["episodes"]
        sc = [e["score"] for e in eps]
        trained = r.get("num_timesteps")
        rows.append(
            f"| `{model_label(run_dir, r['checkpoint'])}` | {trained / 1e6:.3g}M | " if trained else
            f"| `{model_label(run_dir, r['checkpoint'])}` | ? | "
        )
        rows[-1] += (
            f"{'argmax' if r['deterministic'] else 'sampled'} | {len(eps)} | "
            f"{st.mean(sc):,.0f} ± {st.pstdev(sc):,.0f} | {min(sc):,} / {max(sc):,} | "
            f"{st.mean(e['steps'] for e in eps):,.0f} | {sum(e['truncated'] for e in eps)}/{len(eps)} | "
            f"{max(e['max_round'] for e in eps)} | {st.mean(e['deaths'] for e in eps):.1f} |"
        )
    return "\n".join(rows)


def write_summary(run_dir, table):
    path = os.path.join(run_dir, "eval", "summary.md")
    with open(path, "w") as f:
        f.write("# Evaluation summary\n\nEach model replayed from Level1.state by `scripts/evaluate.py` "
                "(sampled rows seed 0, argmax rows seed 1). Episode 1 of every row is in "
                "`media/<model>.mp4` (real-time speed) with an 8-frame strip in `media/<model>_frames.png`.\n\n"
                + table + "\n")
    return path


def write_run_md(run_dir, table):
    path = os.path.join(run_dir, "RUN.md")
    if os.path.exists(path):
        print(f"{path} exists; left untouched")
        return path
    name = os.path.basename(os.path.normpath(run_dir))
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    lines = [f"# Run {name}", "",
             "_Generated by `scripts/archive_run.py`. Fill in the Findings section; re-running the script "
             "refreshes the curve and tables but never overwrites this file._", "", "## Setup", ""]
    if cfg:
        lines += [f"- Started {cfg.get('started')} at commit `{cfg.get('git_commit')}`",
                  f"- Command: `{cfg.get('command')}`",
                  f"- PPO: `{json.dumps(cfg.get('hyperparams'))}`",
                  f"- Env: `{json.dumps(cfg.get('env'))}`",
                  f"- Periodic evaluation: `{json.dumps(cfg.get('eval'))}`"]
        resumes = sorted(glob.glob(os.path.join(run_dir, "config_resume_*.json")))
        for r in resumes:
            rc = json.load(open(r))
            lines.append(f"- Resumed {rc.get('started')}: `{rc.get('command')}`")
    else:
        lines.append("- No config.json: this run predates per-run directories in scripts/train.py.")
    lines += ["", "## Training curve", "", "![learning curve](learning_curve.png)", "",
              "## Evaluation", "", table or "_not evaluated yet (run archive_run.py with --episodes > 0)_", "",
              "Videos: `eval/media/<model>.mp4`; per-episode detail: `eval/results.json`.", "",
              "## Findings", "", "_What happened, why, and what the next run should change._", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--episodes", type=int, default=3, help="episodes per model (0 = skip evaluation)")
    parser.add_argument("--milestones", type=int, default=4, help="evenly spaced checkpoints to evaluate")
    parser.add_argument("--no-argmax", action="store_true", help="skip the extra argmax run of the best/final model")
    args = parser.parse_args()
    run_dir = os.path.normpath(args.run_dir)

    try:
        print("wrote", plot_curve.plot(run_dir, os.path.join(run_dir, "learning_curve.png")))
    except SystemExit as e:
        print(f"skipping curve: {e}")

    table = ""
    results_path = os.path.join(run_dir, "eval", "results.json")
    if args.episodes > 0:
        models = pick_models(run_dir, args.milestones)
        if not models:
            print("no models found to evaluate")
        else:
            jobs = [(m, False) for m in models]
            if not args.no_argmax:
                headline = next((m for m in models if "best_model" in m), models[-1])
                jobs.append((headline, True))
            print(f"evaluating {len(jobs)} model runs x {args.episodes} episodes in parallel ...")
            merged = evaluate(run_dir, jobs, args.episodes)
            table = make_table(run_dir, merged)
    elif os.path.exists(results_path):
        table = make_table(run_dir, json.load(open(results_path)))
    if table:
        print("wrote", write_summary(run_dir, table))
        print(table)
    print("wrote", write_run_md(run_dir, table))


if __name__ == "__main__":
    main()
