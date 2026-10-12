"""Plot the training curve (mean episode score and length) for a run directory.

Reads every tensorboard event file under the directory, so resumed segments
(logs/, logs_from_<steps>/) and the legacy tb_logs/PPO_n layout both work, and
overlays the periodic evaluation (eval/mean_reward, eval/mean_ep_length) when
train.py recorded one.

Usage:
  python scripts/plot_curve.py runs/<run-name>          # -> runs/<run-name>/learning_curve.png
  python scripts/plot_curve.py tb_logs/PPO_3 -o eval/curve.png
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRAIN_COLOR, EVAL_COLOR = "#2a78d6", "#eb6834"
TAGS = ("rollout/ep_rew_mean", "rollout/ep_len_mean", "eval/mean_reward", "eval/mean_ep_length")


def load(run_dir):
    files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents*"), recursive=True)
    if not files:
        raise SystemExit(f"no tensorboard event files under {run_dir}")
    series = {t: {} for t in TAGS}
    for d in sorted({os.path.dirname(f) for f in files}):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        for t in TAGS:
            if t in ea.Tags()["scalars"]:
                for e in ea.Scalars(t):
                    series[t][e.step] = e.value  # later segments win on equal steps
    return {t: (sorted(d), [d[k] for k in sorted(d)]) for t, d in series.items()}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.margins(x=0.02)


def plot(run_dir, out, episode_cap=4500):
    data = load(run_dir)
    has_eval = bool(data["eval/mean_reward"][0])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, facecolor=SURFACE)
    panels = [
        (axes[0], "rollout/ep_rew_mean", "eval/mean_reward", "Mean episode score (game points)"),
        (axes[1], "rollout/ep_len_mean", "eval/mean_ep_length", "Mean episode length (agent steps, 4 frames each)"),
    ]
    for ax, train_tag, eval_tag, title in panels:
        x, y = data[train_tag]
        style(ax)
        ax.plot(x, y, color=TRAIN_COLOR, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
        if x:
            ax.scatter([x[-1]], [y[-1]], s=50, color=TRAIN_COLOR, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.annotate(f"{y[-1]:,.0f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        if has_eval:
            ex, ey = data[eval_tag]
            ax.scatter(ex, ey, s=30, color=EVAL_COLOR, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        if not has_eval:
            title += " — 100-episode rolling mean"
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_ylim(bottom=0)
    axes[1].axhline(episode_cap, color=AXIS, linewidth=1, zorder=1)
    axes[1].annotate(f"episode cap ({episode_cap:,} steps)", (data["rollout/ep_len_mean"][0][0], episode_cap),
                     xytext=(0, 4), textcoords="offset points", fontsize=8, color=MUTED)
    axes[1].set_ylim(0, episode_cap * 1.12)
    xmax = max(data["rollout/ep_rew_mean"][0] or [0])
    unit, div = ("M", 1e6) if xmax >= 1e6 else ("k", 1e3)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / div:g}{unit}"))
    axes[1].set_xlabel("training steps", color=MUTED, fontsize=9)
    if has_eval:
        handles = [
            Line2D([], [], color=TRAIN_COLOR, linewidth=2, label="training episodes (100-episode rolling mean, clipped-reward policy rollouts)"),
            Line2D([], [], color=EVAL_COLOR, marker="o", linestyle="", markersize=6,
                   markeredgecolor=SURFACE, label="periodic evaluation (raw score, sampled policy)"),
        ]
        axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.suptitle(f"PPO on Mappy (NES) — {os.path.basename(os.path.normpath(run_dir))}",
                 x=axes[0].get_position().x0, ha="left", fontsize=13, color=INK, fontweight="bold")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None, help="run directory (default: newest under runs/)")
    parser.add_argument("-o", "--out", default=None, help="output PNG (default: <run_dir>/learning_curve.png)")
    parser.add_argument("--episode-cap", type=int, default=4500)
    args = parser.parse_args()
    run_dir = args.run_dir or max(glob.glob("runs/*/"), key=os.path.getmtime)
    out = args.out or os.path.join(run_dir, "learning_curve.png")
    print(f"wrote {plot(run_dir, out, args.episode_cap)} from {run_dir}")


if __name__ == "__main__":
    main()
