"""Install a Mappy ROM into the integration and sanity-check that it boots.

Usage: python scripts/setup_rom.py [path/to/rom.nes]

If no path is given, uses integrations/Mappy-Nes/rom.nes if already present.
Writes rom.sha (SHA1 of the ROM minus its 16-byte iNES header) and boots the
emulator for a few hundred frames with no save state to confirm the ROM runs.
"""

import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import stable_retro as retro

import mappy_env

INTEGRATION_DIR = os.path.join(mappy_env.INTEGRATIONS_DIR, "Mappy-Nes")
ROM_DEST = os.path.join(INTEGRATION_DIR, "rom.nes")


def main():
    if len(sys.argv) > 1:
        src = sys.argv[1]
        if os.path.abspath(src) != os.path.abspath(ROM_DEST):
            shutil.copyfile(src, ROM_DEST)
            print(f"Copied {src} -> {ROM_DEST}")
    if not os.path.exists(ROM_DEST):
        sys.exit(f"No ROM found. Usage: python scripts/setup_rom.py path/to/rom.nes")

    data = open(ROM_DEST, "rb").read()
    if data[:4] != b"NES\x1a":
        sys.exit("File does not look like an iNES ROM (bad magic).")
    sha1 = hashlib.sha1(data[16:]).hexdigest()
    with open(os.path.join(INTEGRATION_DIR, "rom.sha"), "w") as f:
        f.write(sha1 + "\n")
    print(f"rom.sha written: {sha1}")

    env = mappy_env.make_raw_env(state=retro.State.NONE, render_mode=None)
    env.reset()
    for _ in range(300):
        obs, _, _, _, info = env.step([False] * env.action_space.n)
    env.close()
    print(f"Boot OK: 300 frames stepped, obs shape {obs.shape}, info {info}")


if __name__ == "__main__":
    main()
