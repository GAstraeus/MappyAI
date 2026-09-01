"""Play Mappy with the keyboard; save states and RAM logs.

Usage:
  python scripts/play.py                 # boot from power-on (title screen)
  python scripts/play.py --state Level1  # start from the saved state

Keys:
  arrows      move          ENTER  START        X  A (attack button)
  Z           B             RSHIFT SELECT       S  save integrations/.../<name>.state
  Q / ESC     quit (writes RAM log to ram_logs/)

Every frame of RAM (10240 bytes: 2KB CPU RAM first) is logged and written to
ram_logs/play_<timestamp>.npz on exit, for scripts/find_score.py.
"""

import argparse
import gzip
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import stable_retro as retro

import mappy_env

KEYMAP = {
    "UP": "K_UP", "DOWN": "K_DOWN", "LEFT": "K_LEFT", "RIGHT": "K_RIGHT",
    "A": "K_x", "B": "K_z", "START": "K_RETURN", "SELECT": "K_RSHIFT",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=None,
                        help="state name to load (default: power-on boot)")
    parser.add_argument("--save-name", default="Level1",
                        help="name used when saving a state with S")
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    import pygame

    state = args.state if args.state else retro.State.NONE
    env = mappy_env.make_raw_env(state=state, actions=retro.Actions.ALL)
    buttons = env.buttons
    obs, _ = env.reset()

    pygame.init()
    h, w = obs.shape[0], obs.shape[1]
    screen = pygame.display.set_mode((w * args.scale, h * args.scale))
    pygame.display.set_caption("Mappy — S: save state, Q: quit")
    clock = pygame.time.Clock()

    rams = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_s:
                    path = os.path.join(mappy_env.INTEGRATIONS_DIR, "Mappy-Nes",
                                        f"{args.save_name}.state")
                    with gzip.open(path, "wb") as f:
                        f.write(env.em.get_state())
                    print(f"saved {path}")

        pressed = pygame.key.get_pressed()
        action = [pressed[getattr(pygame, KEYMAP[b])] if b in KEYMAP else False
                  for b in buttons]
        obs, _, terminated, truncated, info = env.step(action)
        rams.append(env.get_ram().copy())
        if terminated or truncated:
            print(f"episode done (info={info}); resetting")
            obs, _ = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        surf = pygame.transform.scale(surf, screen.get_size())
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()
    if rams:
        os.makedirs(os.path.join(mappy_env.PROJECT_DIR, "ram_logs"), exist_ok=True)
        out = os.path.join(mappy_env.PROJECT_DIR, "ram_logs",
                           f"play_{int(time.time())}.npz")
        np.savez_compressed(out, ram=np.array(rams))
        print(f"wrote {len(rams)} frames of RAM to {out}")


if __name__ == "__main__":
    main()
