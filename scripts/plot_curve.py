"""Plot the training curve (mean episode score and length) from tensorboard logs.

Usage:
  python scripts/plot_curve.py                         # latest run in tb_logs/
  python scripts/plot_curve.py tb_logs/PPO_3 -o eval/learning_curve.png
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]


def load(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
        s = ea.Scalars(tag)
        out[tag] = ([e.step for e in s], [e.value for e in s])
    return out


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
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.margins(x=0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("-o", "--out", default="eval/learning_curve.png")
    parser.add_argument("--episode-cap", type=int, default=4500)
    args = parser.parse_args()
    run_dir = args.run_dir or max(glob.glob("tb_logs/*"), key=os.path.getmtime)
    data = load(run_dir)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, facecolor=SURFACE)
    panels = [
        (axes[0], "rollout/ep_rew_mean", "Mean episode score (game points, 100-episode rolling mean)", SERIES[0]),
        (axes[1], "rollout/ep_len_mean", "Mean episode length (agent steps, 4 frames each)", SERIES[1]),
    ]
    for ax, tag, title, color in panels:
        x, y = data[tag]
        style(ax)
        ax.plot(x, y, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
        ax.scatter([x[-1]], [y[-1]], s=50, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{y[-1]:,.0f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_ylim(bottom=0)
    cap = args.episode_cap
    axes[1].axhline(cap, color=AXIS, linewidth=1, zorder=1)
    axes[1].annotate(f"episode cap ({cap:,} steps)", (data["rollout/ep_len_mean"][0][0], cap),
                     xytext=(0, 4), textcoords="offset points", fontsize=8, color=MUTED)
    axes[1].set_ylim(0, cap * 1.12)
    axes[1].set_xlabel("training steps", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.suptitle(f"PPO on Mappy (NES) — {os.path.basename(run_dir)}", x=axes[0].get_position().x0,
                 ha="left", fontsize=13, color=INK, fontweight="bold")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out} from {run_dir}")


if __name__ == "__main__":
    main()
