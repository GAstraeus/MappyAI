# Mappy (NES) — RL agent via stable-retro + PPO

Trains a PPO agent (Stable-Baselines3, CNN policy, pixel observations) to play
Mappy from a cartridge-dumped ROM, using a custom stable-retro integration.

## Setup

```sh
python3.12 -m venv .venv          # any Python 3.10–3.14
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/setup_rom.py path/to/rom.nes   # installs ROM, sanity-boots it
```

`integrations/Mappy-Nes/Level1.state` is committed; regenerate it any time with
`python scripts/make_state.py` (deterministic from power-on).

## Usage

```sh
python scripts/train.py                     # 10M steps, 8 workers, checkpoints/ + tb_logs/
python scripts/watch.py checkpoints/ppo_mappy_final.zip            # live window
python scripts/watch.py checkpoints/ppo_mappy_final.zip --record   # .bk2 + .mp4 (needs ffmpeg)
python scripts/play.py --state Level1       # play with keyboard, logs RAM to ram_logs/
python scripts/find_score.py ram_logs/play_<ts>.npz   # hunt RAM variables in a log
```

Training on pixels needs millions of frames — expect hours. Watch progress in
tensorboard (`tensorboard --logdir tb_logs`, `rollout/ep_rew_mean`).

## Integration: what was reverse-engineered and how

All addresses found empirically by logging `env.get_ram()` during scripted /
search-driven play and matching against the rendered score display:

| variable | address | type | meaning |
|---|---|---|---|
| `score` | 0x72–0x74 (114) | `>d3` | big-endian packed BCD, equals displayed score (verified at 0, 40, 60, 80, 200, 3210+) |
| `lives` | 0x70 (112) | `\|u1` | **reserve** lives: 2 while playing the first life, 0 on the last, resets to 3 in attract mode |
| `gameover` | 0x491 (1169) | `\|u1` | 0 during play; 12 while the GAME OVER banner shows (~4 s) |

- **Reward** = score delta (`scenario.json`); `ClipRewardEnv` sign-clips it during training.
- **Done** = `gameover != 0 AND lives == 3`. Lives alone can't express game over
  (it never goes negative; it jumps 0→3 when attract mode starts, and 3 lives can
  legitimately occur in-game after the 20000-point extra life). The banner byte
  0x491 stayed 0 through all observed play, deaths, and item pickups.
- Episodes start from `Level1.state` and truncate after 4500 wrapped steps
  (~5 min of game time) via `TimeLimit`.

## Gotchas discovered (beyond the agent brief)

- **`Actions.FILTERED` silently drops START on this NES core.** Anything that
  must press START (play.py, make_state.py) uses `Actions.ALL`. Training never
  needs START since episodes begin from the save state.
- The attract demo mimics real gameplay (same lives display, live sprites).
  Sanity checks that "the game is running" must key off `lives == 2` after
  START, not off the screen looking like gameplay.
- START needs a long hold at the title (30 frames used); short 10-frame taps
  were not always registered.
- `env.get_ram()` returns 10240 bytes; the 2KB NES CPU RAM comes first, and
  `data.json` addresses index into it directly (no 0x8000-style offsets).
- The game-mode byte 0x135 (17=play, 3/73=death, 1=game over) looked usable for
  `done` but takes extra values (61, 196, 232) during normal play — avoided.

## Action space

`Discretizer` maps to 6 combos: `[[], LEFT, RIGHT, A, LEFT+A, RIGHT+A]`.
A-vs-B was settled empirically (branch the emulator, press each, diff the
pixels): **A and B are interchangeable** — both toggle doors — so only A is
in the action space.
