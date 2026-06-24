# QTris ELO leaderboard

A live-updating web leaderboard of the 1v1 AlphaZero opponent-pool ELO ratings.

- **Worker** (`worker.js`): serves the page and a small JSON API. `GET /api/leaderboard` returns
  the latest published payload; `POST /api/publish` stores a new one (bearer-token auth).
- **Page** (`public/index.html`): polls `/api/leaderboard` every 30s and renders a ranked table of
  the learner, the frozen `gen_0` reference, and the pool snapshots.
- **Publisher** (`../scripts/publish_leaderboard.py`): runs on the training machine, reads
  `checkpoints/1v1_placement_az/pool/elo.json`, and POSTs it to the Worker on a timer, writing only
  when the ratings change.

The learner's displayed rating is a restart-safe EMA that smooths the raw, noisy per-generation
Elo (hover the value for the raw number); this is display-only and never alters `elo.json`.
