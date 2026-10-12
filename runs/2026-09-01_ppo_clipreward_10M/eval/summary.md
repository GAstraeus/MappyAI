# Evaluation summary

Each checkpoint replayed from Level1.state with `scripts/evaluate.py` (seed 0; argmax run seed 1). Episode 1 of every row is recorded in `media/<checkpoint>.mp4` (real-time speed) with an 8-frame strip in `media/<checkpoint>_frames.png`.

| checkpoint | trained steps | policy | episodes | score mean ± sd | min / max | mean length (steps) | hit time cap | max round | items collected |
|---|---|---|---|---|---|---|---|---|---|
| `ppo_mappy_1000000_steps.zip` | 1M | sampled | 3 | 1,333 ± 482 | 760 / 1,940 | 1,271 | 0/3 | 1 | 0 |
| `ppo_mappy_2500000_steps.zip` | 2.5M | sampled | 3 | 2,517 ± 968 | 1,730 / 3,880 | 2,094 | 0/3 | 1 | 0 |
| `ppo_mappy_5000000_steps.zip` | 5M | sampled | 3 | 3,307 ± 720 | 2,720 / 4,320 | 2,735 | 0/3 | 1 | 0 |
| `ppo_mappy_7500000_steps.zip` | 7.5M | sampled | 3 | 3,287 ± 749 | 2,470 / 4,280 | 2,693 | 0/3 | 1 | 0 |
| `ppo_mappy_final.zip` | 10M | sampled | 5 | 5,088 ± 850 | 3,540 / 5,870 | 4,007 | 2/5 | 1 | 0 |
| `ppo_mappy_final.zip` | 10M | argmax | 3 | 5,933 ± 17 | 5,910 / 5,950 | 4,500 | 3/3 | 1 | 0 |

'items collected' is inferred: every score delta observed was +10 (the trampoline-bounce award); items are worth 100+.
