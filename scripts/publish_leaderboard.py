"""Publish the 1v1 AZ ELO ratings to the Cloudflare leaderboard Worker.

Reads the trainer's `pool/elo.json`, builds a compact ranked payload, and POSTs it to the
Worker's `/api/publish`. Stdlib only, so it runs as a bare cron job without the training env.

Env (or flags): LEADERBOARD_URL (the Worker base URL), LEADERBOARD_TOKEN (the publish secret).
Publishes only when the substantive payload changes, and not more than once per --min-interval,
to stay under Workers KV's free-tier 1,000 writes/day.
"""

import argparse
import glob
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

DEFAULT_POOL = "checkpoints/1v1_placement_az/pool"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path):
    """Fill os.environ from a simple KEY=VALUE .env file; real env vars still win."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, val.strip().strip("\"'"))


def _present_gens(pool_dir):
    """Basenames of pool snapshots currently on disk (gen_*.index)."""
    return {
        os.path.basename(p)[: -len(".index")]
        for p in glob.glob(os.path.join(pool_dir, "gen_*.index"))
    }


def build_payload(pool_dir):
    """Ranked leaderboard payload from elo.json, or None if it is missing/mid-write."""
    elo_path = os.path.join(pool_dir, "elo.json")
    try:
        with open(elo_path) as f:
            book = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    ratings, games = book["ratings"], book["games"]
    learner, anchor = book["learner"], book["anchor"]
    present = _present_gens(pool_dir)

    entries = [
        {
            "id": pid,
            "rating": round(float(rating), 2),
            "games": int(games.get(pid, 0)),
            # learner and the pinned anchor are always live; other gens only if on disk.
            "present": pid in (learner, anchor) or pid in present,
            "is_learner": pid == learner,
            "is_reference": pid == anchor,
        }
        for pid, rating in ratings.items()
    ]
    entries.sort(key=lambda e: e["rating"], reverse=True)

    pool = [
        e["rating"] for e in entries if not e["is_learner"] and not e["is_reference"]
    ]
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_pool": max(pool) if pool else None,
        "entries": entries,
    }


def _digest(payload):
    """Hash of the substantive content (ignores updated_at so the clock alone never publishes)."""
    body = {k: v for k, v in payload.items() if k != "updated_at"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def publish(payload, url, token):
    req = urllib.request.Request(
        url.rstrip("/") + "/api/publish",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            # Cloudflare's bot check (error 1010) blocks the default Python-urllib UA.
            "User-Agent": "qtris-leaderboard-publisher/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status


def main():
    ap = argparse.ArgumentParser(
        description="Publish ELO ratings to the leaderboard Worker."
    )
    ap.add_argument(
        "--pool-dir", default=DEFAULT_POOL, help="dir holding elo.json + gen_*.index"
    )
    ap.add_argument("--url", default=None, help="Worker base URL (or LEADERBOARD_URL)")
    ap.add_argument(
        "--token", default=None, help="publish secret (or LEADERBOARD_TOKEN)"
    )
    ap.add_argument(
        "--env-file",
        default=os.path.join(_REPO_ROOT, ".env"),
        help="dotenv file holding LEADERBOARD_URL / LEADERBOARD_TOKEN",
    )
    ap.add_argument("--once", action="store_true", help="publish once and exit")
    ap.add_argument(
        "--interval", type=int, default=120, help="loop poll seconds (default 120)"
    )
    ap.add_argument(
        "--min-interval",
        type=int,
        default=120,
        help="min seconds between writes, KV free-tier guard (default 120)",
    )
    args = ap.parse_args()

    # Precedence: explicit flag > already-exported env > .env file.
    _load_dotenv(args.env_file)
    url = args.url or os.environ.get("LEADERBOARD_URL")
    token = args.token or os.environ.get("LEADERBOARD_TOKEN")
    if not url or not token:
        ap.error(
            "set LEADERBOARD_URL and LEADERBOARD_TOKEN (in .env, the environment, "
            "or via --url/--token)."
        )

    last_digest = None
    last_write = 0.0
    while True:
        payload = build_payload(args.pool_dir)
        if payload is None:
            print(f"no readable elo.json in {args.pool_dir}, skipping", flush=True)
        else:
            digest = _digest(payload)
            now = time.monotonic()
            if digest == last_digest:
                pass
            elif now - last_write < args.min_interval and not args.once:
                pass  # changed, but hold off to respect the write cap
            else:
                status = publish(payload, url, token)
                last_digest, last_write = digest, now
                n = len(payload["entries"])
                print(
                    f"published {n} entries -> {status} at {payload['updated_at']}",
                    flush=True,
                )

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
