# QTris rating leaderboard

A live-updating web leaderboard of the 1v1 AlphaZero opponent-pool ratings.

- **Worker** (`worker.js`): serves the page and a small JSON API. `GET /api/leaderboard` returns
  the latest published payload; `POST /api/publish` stores a new one (bearer-token auth).
- **Page** (`public/index.html`): polls `/api/leaderboard` every 30s and renders a ranked table of
  the learner, the frozen `gen_0` reference, and the pool snapshots.
- **Publisher** (`../scripts/publish_leaderboard.py`): runs on the training machine, reads
  `checkpoints/1v1_placement_az/pool/ratings.json`, and POSTs it to the Worker on a timer, writing
  only when the displayed content changes (plus an hourly liveness publish).

Ratings come from the trainer's whole-history rating (WHR) batch refit: each value is the MAP
estimate with a ±1σ Laplace uncertainty, anchored at `gen_0` = 1500. The learner's displayed
rating is the latest point of its fitted trajectory; the random-walk prior in the fit is the
smoother, so the page shows the posterior directly (no display-side EMA). It moves within ~±1σ
between refreshes by construction.
