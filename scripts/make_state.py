"""Recreate Level1.state deterministically from power-on.

Usage: python scripts/make_state.py

Boots the ROM, waits out the title fade-in, holds START for 30 frames
(Actions.ALL is required — FILTERED drops START on this core), then saves
the state at the first frame where the player has control. NES emulation is
deterministic, so this produces an identical state on any machine.

Control detection: branch the emulator state, hold LEFT vs RIGHT for 20
frames each, and look for any pixel divergence. During the round intro the
branches are identical; once the player can move they differ.
"""

import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import stable_retro as retro

import mappy_env

STATE_PATH = os.path.join(mappy_env.INTEGRATIONS_DIR, "Mappy-Nes", "Level1.state")


def main():
    env = mappy_env.make_raw_env(state=retro.State.NONE, actions=retro.Actions.ALL)
    buttons = env.buttons
    env.reset()

    def run(names, frames):
        action = [b in names for b in buttons]
        obs = None
        for _ in range(frames):
            obs, _, _, _, _ = env.step(action)
        return obs

    run([], 240)          # title screen fade-in
    run(["START"], 30)    # start a 1-player game
    base = env.em.get_state()

    def controllable(offset):
        env.em.set_state(base)
        run([], offset)
        mid = env.em.get_state()
        obs_l = run(["LEFT"], 20)
        env.em.set_state(mid)
        obs_r = run(["RIGHT"], 20)
        return np.any(obs_l != obs_r)

    offset = 0
    while offset <= 900 and not controllable(offset):
        offset += 15
    if offset > 900:
        sys.exit("Never gained control after START — did the game start?")

    env.em.set_state(base)
    run([], offset)
    lives = env.get_ram()[0x70]
    if lives != 2:
        sys.exit(f"Unexpected lives value {lives} at save point (expected 2).")
    with gzip.open(STATE_PATH, "wb") as f:
        f.write(env.em.get_state())
    env.close()
    print(f"Saved {STATE_PATH} at {offset} frames after START (lives byte = {lives}).")


if __name__ == "__main__":
    main()
