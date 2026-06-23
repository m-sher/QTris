# QTris ELO leaderboard

A free, live-updating leaderboard of the 1v1 AlphaZero opponent-pool ELO ratings.

- **Worker** (`worker.js`): serves the page and a KV-backed JSON API.
  - `GET /api/leaderboard` returns the latest payload from KV.
  - `POST /api/publish` writes a new payload (bearer-token auth).
- **Page** (`public/index.html`): polls `/api/leaderboard` every 30s and renders a ranked table.
- **Publisher** (`../scripts/publish_leaderboard.py`): on the training machine, reads
  `checkpoints/1v1_placement_az/pool/elo.json` and POSTs to `/api/publish` on a timer.

Free-tier note: Workers KV allows 1,000 writes/day. The publisher only writes when the ratings
change and at most once per `--min-interval` (default 120s), so it stays well under the cap.

## One-time deploy (Cloudflare)

Requires a free Cloudflare account. Use `npx wrangler` (no global install needed).

```bash
cd leaderboard
npx wrangler login

# Create the KV namespace, then paste the printed id into wrangler.toml (kv_namespaces.id).
npx wrangler kv namespace create LEADERBOARD

# Set the shared publish secret (any long random string; reuse it for the publisher below).
npx wrangler secret put PUBLISH_TOKEN

npx wrangler deploy
# Note the printed URL, e.g. https://qtris-leaderboard.<subdomain>.workers.dev
```

## Run the publisher (training machine)

Put your URL and token in a `.env` at the repo root (copy the template, gitignored so the
token is never committed):

```bash
cp .env.example .env
# then edit .env:
#   LEADERBOARD_URL=https://qtris-leaderboard.<subdomain>.workers.dev
#   LEADERBOARD_TOKEN=<same secret you set above>
```

The publisher auto-loads `.env` from the repo root (precedence: `--url`/`--token` flag >
exported env var > `.env`), so no `export` is needed:

```bash
# One-shot (verify it works):
uv run python scripts/publish_leaderboard.py --once

# Background loop, detached (survives closing the terminal):
scripts/run_leaderboard_publisher.sh
```

Or via cron (stdlib only, so plain `python3` works) every 2 minutes; it reads the same `.env`:

```cron
*/2 * * * * cd /path/to/QTris && python3 scripts/publish_leaderboard.py --once >> logs/leaderboard_cron.log 2>&1
```

Then open the Worker URL.

## Local dev

```bash
cd leaderboard && npx wrangler dev          # serves on http://localhost:8787
# In another shell (PUBLISH_TOKEN here matches your local .dev.vars or a test value):
LEADERBOARD_URL=http://localhost:8787 LEADERBOARD_TOKEN=test \
  uv run python scripts/publish_leaderboard.py --once
```
