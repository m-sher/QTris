"""Tag an untagged spectate replay with its round boundaries.

A spectate replay with no round markers is one session of many rounds. A round start
is a frame whose board is empty with b2b and combo both -1; a perfect clear also
empties the board but leaves both non-negative.
The piece stream is an independent check: it cannot be continuous across a boundary.
Match boundaries are not recoverable, so every round is tagged game 1.

    uv run python scripts/tag_replay_rounds.py PATH [--dry-run] [--out OTHER]
"""

import argparse
import json
import os
import sys

import numpy as np
from recover_spectate_keys import _expected_after, _queue_match


def round_starts(frames):
    """Frame indices that begin a round, by empty board plus a reset scorer."""
    starts = []
    for i, frame in enumerate(frames):
        empty = not np.any(np.asarray(frame["board"]))
        b2b, combo, _ = frame["b2b_combo_garbage"]
        if i == 0 or (empty and int(b2b) == -1 and int(combo) == -1):
            starts.append(i)
    return starts


def queue_breaks(frames):
    """Frame indices the piece stream cannot reach from the frame before."""
    out = []
    for i in range(len(frames) - 1):
        if not any(
            _queue_match(
                _expected_after(frames[i]["pieces"], hold), frames[i + 1]["pieces"]
            )
            for hold in (False, True)
        ):
            out.append(i + 1)
    return out


def tag(frames, starts):
    bounds = starts + [len(frames)]
    for r, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
        for frame in frames[a:b]:
            frame["game"] = 1
            frame["round"] = r
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="replay JSON to tag")
    ap.add_argument("--out", default=None, help="output path (default: in place)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument(
        "--force", action="store_true", help="tag even if the checks disagree"
    )
    args = ap.parse_args(argv)

    with open(args.path) as f:
        data = json.load(f)
    frames = data["frames"]
    starts = round_starts(frames)
    breaks = queue_breaks(frames)
    print(f"{os.path.basename(args.path)}: {len(frames)} frames")
    print(f"round starts:        {starts}")
    print(f"queue discontinuity: {breaks}")

    if breaks != starts[1:]:
        print(
            "the two signals disagree"
            + ("; tagging anyway" if args.force else "; not tagging")
        )
        if not args.force:
            return 1

    bounds = starts + [len(frames)]
    for r, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
        print(f"  round {r}: frames {a}-{b - 1} ({b - a})")
    if args.dry_run:
        return 0

    dest = args.out or args.path
    data["mode"] = data.get("mode") or "spectate"
    data["frames"] = tag(frames, starts)
    with open(dest, "w") as f:
        json.dump(data, f)
    print(f"wrote {dest} ({len(starts)} rounds, mode={data['mode']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
