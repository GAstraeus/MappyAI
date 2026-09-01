"""Locate score-like variables in a RAM log recorded by play.py.

Usage: python scripts/find_score.py ram_logs/play_<ts>.npz [--min-increases 2]

Scans every address for multi-byte groups that decode to a monotonically
non-decreasing number under common NES encodings, and ranks candidates by
number of increases. Match the final decoded value against the score you
remember from the session.

Note: Mappy's score was already located this way — big-endian packed BCD in
bytes 0x72-0x74, i.e. data.json type ">d3" at address 114. This script is
kept for finding further variables (round number, item flags, ...).
"""

import argparse

import numpy as np


def decode(block, kind):
    """block: (frames, width) uint8 array -> (frames,) int64 or None."""
    b = block.astype(np.int64)
    if kind == "le-int":
        return sum(b[:, i] << (8 * i) for i in range(b.shape[1]))
    if kind == "be-int":
        w = b.shape[1]
        return sum(b[:, i] << (8 * (w - 1 - i)) for i in range(w))
    if kind in ("le-bcd", "be-bcd"):
        if ((b >> 4) > 9).any() or ((b & 0xF) > 9).any():
            return None
        digits = (b >> 4) * 10 + (b & 0xF)
        order = range(b.shape[1]) if kind == "le-bcd" else range(b.shape[1] - 1, -1, -1)
        out = np.zeros(b.shape[0], dtype=np.int64)
        for i in order:
            out = out * 100 + digits[:, i]
        # le-bcd: least-significant byte first => reverse accumulation order
        if kind == "le-bcd":
            out = np.zeros(b.shape[0], dtype=np.int64)
            for i in range(b.shape[1] - 1, -1, -1):
                out = out * 100 + digits[:, i]
        return out
    if kind in ("le-digit", "be-digit"):
        if (b > 9).any():
            return None
        order = range(b.shape[1] - 1, -1, -1) if kind == "le-digit" else range(b.shape[1])
        out = np.zeros(b.shape[0], dtype=np.int64)
        for i in order:
            out = out * 10 + b[:, i]
        return out
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--min-increases", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=4)
    args = parser.parse_args()

    ram = np.load(args.log)["ram"]
    print(f"log: {ram.shape[0]} frames x {ram.shape[1]} bytes")

    results = []
    for addr in range(ram.shape[1] - args.max_width):
        for width in range(1, args.max_width + 1):
            block = ram[:, addr:addr + width]
            for kind in ("le-int", "be-int", "le-bcd", "be-bcd", "le-digit", "be-digit"):
                if width == 1 and kind.startswith("be"):
                    continue  # same as le for width 1
                vals = decode(block, kind)
                if vals is None:
                    continue
                d = np.diff(vals)
                if (d >= 0).all() and (d > 0).sum() >= args.min_increases and vals[-1] > vals[0]:
                    results.append((int((d > 0).sum()), addr, width, kind, int(vals[0]), int(vals[-1])))
                    break  # one encoding hit per (addr, width) is enough

    results.sort(reverse=True)
    print(f"{'incr':>5} {'addr':>6} {'width':>5} {'enc':>9} {'first':>8} {'final':>8}")
    for inc, addr, width, kind, v0, v1 in results[:25]:
        print(f"{inc:5d} 0x{addr:04x} {width:5d} {kind:>9} {v0:8d} {v1:8d}")
    if not results:
        print("no monotone candidates — play longer and collect more points")


if __name__ == "__main__":
    main()
