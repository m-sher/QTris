I moved the datasets into the dataset folder but have not updated the paths to them yet

I eventually want to move the AR runner from the environment to this repo.

I need to update the pretrainer buffer size

## Phase 1 refactor (2026-05-15)

`Oneshot/` will be deleted in Phase 1c (still present as of end-of-1b).
The Flat variant in `src/qtris/models/flat/` supersedes it.

Recover the legacy `Oneshot/` tree from git with:

    git checkout b4952c6 -- Oneshot/

(`b4952c6` is the HEAD of `big-ol-refactor` immediately prior to the
Phase 1 migration.)

## Phase 1b deferred items (recorded 2026-05-15)

- **`src/qtris/data/convert.py` was NOT refactored** into `main(args)` shape.
  It still has its own internal argparse and an `if __name__ == "__main__":`
  guard. It is NOT wired into `cli/datagen.py`. Run directly via
  `uv run python -m qtris.data.convert --direction flat_to_ar`. Phase 1c or
  later should decide whether to fold it into `datagen` (e.g.
  `uv run datagen --convert flat_to_ar`) or leave standalone.

- **`qtris/logging/` subpackage was deleted** during Phase 1b smoke testing.
  Its empty `__init__.py` shadowed Python's stdlib `logging` (which TF imports
  internally as `from logging import DEBUG`), breaking every `import` of a
  module that imports TF. Phase 3's wandb helper should live under a
  non-shadowing name — candidates: `qtris/run_logging/`, `qtris/wandb_setup/`,
  or just `qtris/wandb_helpers.py` (single module instead of subpackage).

- **`_1v1.py` uses a runtime `global USE_FLAT` mutation pattern** to switch
  between AR and Flat 1v1 training. `main(args)` does
  `global USE_FLAT; USE_FLAT = args.family == "flat"` then continues with
  the existing `if USE_FLAT:` branches throughout the file. Both AR + Flat
  imports happen unconditionally at module load (`Py1v1TetrisRunner`,
  `Py1v1TetrisRunnerFlat`, `PolicyModel`, `FlatPolicyModel`). This is
  pragmatic but hacky. Phase 2 or later should consider either:
  (a) splitting into `_ar_1v1.py` + `_flat_1v1.py` with shared helpers; or
  (b) refactoring the body to thread `use_flat: bool` as a function arg.

- **CLI surface is minimal.** Only a handful of args are exposed
  (--family, --dataset, --num-epochs, --batch-size, --policy-only,
  --num-generations, --mode, --checkpoint, --opponent, --left, --right,
  --steps, --output, --policy-checkpoint, --dagger, --seed). Other
  hyperparams (model depth, num_heads, gamma, search_depth, beam_width,
  garbage_chance, etc.) remain as module-level constants or are hidden
  inside `main()` via a `SimpleNamespace` shim. Phase 5 (config
  centralization) will replace these with `qtris.config` dataclass fields
  + CLI overrides via `dataclasses.replace`.

- **`pygame_widgets` and `imageio` added to pyproject deps** during Phase 1b.7.
  They were used by demos but had been installed out-of-band previously
  (legacy `requirements.txt`, not via uv).

- **`tf_agents.system.multiprocessing.handle_main` is invoked from
  `cli/train.py`,** not the trainer modules themselves. The trainer
  modules have `def main(args)` with no `__main__` guard. cli/train.py
  wraps with `handle_main(lambda _argv: run(args), argv=[sys.argv[0]])`
  to keep `_INITIALIZED[0]` flag for `ParallelPyEnvironment`. Pretrainers,
  demos, and datagen do NOT need this wrapping (single-process).

- **Phase 1c smoke test must use `checkpoints/1v1_ar_policy_22k/`** (or
  21k) as the demo checkpoint, NOT the plan's outdated reference to
  `1v1_ar_policy_14k/`. The 14k checkpoint never existed on disk during
  this refactor; only 21k and 22k are present. No live `1v1_flat_*`
  checkpoint exists yet — flat 1v1 demo smoke-test will need a freshly
  trained checkpoint first.

- **`data/dagger.py` and `demo/ar_1v1.py` use a `SimpleNamespace` shim**
  inside `main(cli_args)` to backfill defaults for parameters the CLI
  doesn't expose. The remaining `args.X` references inside their bodies
  are unchanged. If future CLI work needs to override one of those
  hidden parameters, add a CLI flag in `cli/datagen.py` or `cli/demo.py`
  and overwrite the corresponding SimpleNamespace field.

## Phase 1c outcome (2026-05-18)

`Autoregressive/` and `Oneshot/` no longer exist. All live checkpoint
dirs now live at canonical names under `checkpoints/`:

- `checkpoints/ar_policy/`           (was `Autoregressive/policy_checkpoints/`)
- `checkpoints/ar_policy_445k/`      (was `Autoregressive/policy_checkpoints_445k/`)
- `checkpoints/ar_value/`            (was `Autoregressive/value_checkpoints/`)
- `checkpoints/ar_pretrained_policy/`
- `checkpoints/ar_pretrained_value/`
- `checkpoints/flat_pretrained_policy/`
- `checkpoints/1v1_ar_policy_21k/`   (was `Autoregressive/checkpoints/1v1_ar_policy_21k/`)
- `checkpoints/1v1_ar_policy_22k/`

`Autoregressive/checkpoints/orig_checkpoints/` (3.4 GB of historical
snapshots) was deleted per user authorization — `ar_policy_445k` was
kept as the sole retained historical snapshot. `Demo.gif` at the repo
root now points at the newer 6.6 MB render (was the Autoregressive/
copy); the old 3.7 MB root version was deleted.

### Smoke verification status

Done (lightweight):
- `uv sync` clean
- `uv run pretrain --help`, `train --help`, `demo --help`, `datagen --help` all OK
- All 13 migrated modules importable
- No stale `1v1_ar_policy_14k` / `1v1_flat_policy_17k` references in code
- Dataset paths present at `datasets/tetris_expert_dataset_b2b` and `datasets/tetris_expert_dataset_flat`

### Deferred to a later verification pass (Phase 1d? Phase 2 pre-work?)

User explicitly deferred checkpoint-load verification. Concrete items
to run when picking this up:

- `uv run pretrain ar   --dataset datasets/tetris_expert_dataset_b2b --num-epochs 1 --policy-only`
- `uv run pretrain flat --dataset datasets/tetris_expert_dataset_flat --num-epochs 1`
- `uv run train ar   --num-generations 1`  # confirms PPO loop + restore from `checkpoints/ar_policy`, `ar_value`, `ar_pretrained_value`
- `uv run train flat --num-generations 1`
- `uv run train ar   --mode 1v1 --num-generations 1`
- `uv run train flat --mode 1v1 --num-generations 1`  # NEW dual-family path; bootstraps from scratch
- `uv run demo ar   --checkpoint checkpoints/1v1_ar_policy_22k`  # the canonical "does the AR policy still play sensibly?" test
- `uv run demo vs   --left checkpoints/1v1_ar_policy_21k --right checkpoints/1v1_ar_policy_22k`
- `uv run demo ar   --mode 1v1 --checkpoint checkpoints/1v1_ar_policy_22k --opponent checkpoints/1v1_ar_policy_21k`
- `uv run datagen ar   --steps 100 --output datasets/_smoke_ar`
- `uv run datagen flat --steps 100 --output datasets/_smoke_flat`
- `uv run datagen ar   --dagger --policy-checkpoint checkpoints/1v1_ar_policy_22k --steps 50`

The most important among these is `uv run demo ar --checkpoint checkpoints/1v1_ar_policy_22k`
— it doubles as the Phase 2 checkpoint-parity oracle. If the model produces
nonsense moves vs pre-refactor, `tf.train.Checkpoint.restore()` silently
mismatched variables and Phase 2 layer extraction has more risk.

Also create `scripts/smoke.sh` to pin this list (the plan's
cross-phase verification gate) — deferred along with the runtime tests.

Open at end of Phase 1c:
- `flat 1v1 demo` (demo flat --mode 1v1) is not yet implemented in cli/demo.py
  (it errors out). Will need a `demo/flat_1v1.py` module + dispatcher entry.
  Currently blocked on having a trained flat 1v1 checkpoint at all.
- `src/qtris/data/convert.py` still standalone (run via
  `uv run python -m qtris.data.convert --direction flat_to_ar`).

## Phase 2 pre-flight (2026-05-18)

Parity test landed at `tests/test_checkpoint_parity.py`. Captured
bit-stable goldens (max-abs-diff = 0.0 over 2 consecutive runs) for:

- `PolicyModel` <- `checkpoints/ar_policy/` -> `tests/golden/ar_policy.npy`
- `ValueModel`  <- `checkpoints/ar_value/`  -> `tests/golden/ar_value.npy`

Run with `uv run python tests/test_checkpoint_parity.py`. After every
Phase 2 extraction step, this must still show `max-abs-diff = 0.000e+00`
for both pairs.

### FlatPolicyModel ckpt staleness (deferred work)

`checkpoints/flat_pretrained_policy/ckpt-40` is INCOMPLETE relative to
the current `FlatPolicyModel` definition — `assert_existing_objects_matched()`
reports 8 unmatched variables (e.g. `dense_17/kernel` shape (256, 128)
and friends). The model has grown since the pretrain ckpt was saved.
Unrestored vars get fresh init each run, which makes forward-pass output
non-deterministic (saw ~1.13 diff between runs).

For Phase 2 we therefore demote FlatPolicyModel to a BUILD-ONLY smoke
in the parity test (verifies it constructs + forward-pass shape).
Justification: the shared primitives Phase 2 extracts (PosEncoding,
attention, encoders, process_obs base) are USED by FlatPolicyModel,
and AR parity at max-abs-diff=0.0 will catch any regression in them.

Action when picking up flat work: re-pretrain to capture a complete
`flat_pretrained_policy` checkpoint, then re-enable the FlatPolicyModel
parity branch in `tests/test_checkpoint_parity.py`.

### Deviation from Phase 2 plan

Plan said use `checkpoints/1v1_ar_policy_22k/` and `checkpoints/ar_pretrained_value/`.
User chose `checkpoints/ar_policy/` and `checkpoints/ar_value/` — the live PPO
checkpoints — which is the stronger oracle since it's the actively-trained
model. Goldens reflect this choice.

## Phase 2 outcome (2026-05-18)

Shared model primitives extracted; AR parity preserved at max-abs-diff = 0.000e+00 after every step.

New modules:
- `src/qtris/nn/attention.py` (SelfAttention, CrossAttention, FeedForward)
- `src/qtris/nn/transformer.py` (positional_encoding, PosEncoding, CrossAttentionLayer, EncoderLayer, DecoderLayer)
- `src/qtris/models/encoders.py` (make_patches factory, tokenize_bcg helper)
- `src/qtris/models/base.py` (QtrisModelBase with shared `process_obs` and `_tokenize_bcg` methods)
- `src/qtris/models/value.py` (ValueModel; moved out of ar/model.py so flat/ stops cross-importing)

Line-count delta:
- `src/qtris/models/ar/model.py`: 861 → 394 (-467)
- `src/qtris/models/flat/model.py`: 225 → 148 (-77)
- Shared modules: ~390 LOC total
- Net: code deduplicated, transformer primitives now have a single canonical home

Parity test scaffolding (`tests/`, `tests/golden/`) was deleted at end of phase per user direction ("don't keep these tests"). The bit-stable AR-pair parity oracle held through all 4 extraction steps including the ValueModel move.

### Phase 2 deferred items

- **Back-compat shim: `from qtris.models.value import ValueModel` re-exported in `src/qtris/models/ar/model.py`.** Added so 4 existing import sites (`from qtris.models.ar.model import ValueModel` in pretraining/ar.py, pretraining/flat.py, training/ar.py, training/flat.py) keep working without simultaneous edits. Tagged inline with `# back-compat shim, remove by end of refactor`. To retire: update the 4 import sites to `from qtris.models.value import ValueModel`, then delete the re-export line. Must be gone before the refactor closes (see memory: no-back-compat-by-end-of-refactor).
- **AR parity vs Phase-2-output models reusing checkpoints** still holds end-to-end. Confirmed by `uv run demo ar --checkpoint checkpoints/ar_policy` playing sensible Tetris (carried-over evidence from earlier in the session).
- Phase 2 did NOT touch `AsymmetricValueModel` beyond inheriting `QtrisModelBase`. Its `process_obs` IS now the shared one; its trunks and asymmetric two-player call() are still inline. If Phase 3+ extracts more, consider whether `AsymmetricValueModel` should move to its own module too.

## Observability phase outcome (2026-05-18)

Single wandb logging module landed at `src/qtris/observability/` (2 files):

- `models.py` — pydantic v1 schemas:
  - `PPOConfigBase` (10 shared PPO knobs) → `SingleAgentTrainConfig` (+ expert_coef, early_stopping) and `OneVsOneTrainConfig` (+ flat, b2b_gap_coef, pool_save_interval, max_pool_size)
  - `PPOLogBase` (20 shared per-step metrics + image fields) → `SingleAgentPPOLog` (+ avg_reward, avg_deaths, expert_*) and `OneVsOnePPOLog` (+ APP_*, win/loss outcomes)
  - `PPOLogBase.to_wandb_payload()` wraps `board` / `scores` numpy arrays in `wandb.Image` at log time so callers don't import wandb directly for normal logging
- `wandb_backend.py` — thin helpers: `init_run(*, project, config)`, `log_step(metrics, *, step=None)`, `finish(run)`

All 3 trainers rewired (`training/{ar,flat,_1v1}.py`):
- Replaced inline `config = {...}` dict + `wandb.init(config=config)` with typed config + `init_run(...)`
- Replaced `wandb.log({...})` calls with `log_step(SingleAgentPPOLog(...) | OneVsOnePPOLog(...))`
- Replaced `wandb_run.finish()` with `finish(wandb_run)`
- Dropped dead `import wandb` and `import tf_agents` lines from trainer files (handle_main is invoked from `cli/train.py`, not the trainer modules)

Pydantic version: pinned to v1.10 because tf-agents 0.19.0 transitively requires `typing-extensions==4.5.0`, which blocks pydantic v2. The v1 syntax used:
- `class Config: arbitrary_types_allowed = True` (would become `model_config = ConfigDict(...)` under v2)
- `instance.dict()` (would become `instance.model_dump()` under v2)
- Inline notes in models.py + wandb_backend.py mark the v1→v2 translation points.

Pydantic candidates left for later (per scope decision):
- `SimpleNamespace` shims in `data/dagger.py` and `demo/ar_1v1.py` — natural pydantic BaseModel candidates
- `qtris/config.py` empty dataclass stubs — Phase 5 target; pydantic-settings could replace argparse for CLI override

## Phase 3 outcome (2026-05-18)

Shared training primitives extracted; trainers + pretrainers thinner.

New modules:
- `src/qtris/training/gae.py` — `compute_gae_and_returns`, `compute_raw_returns` (verbatim, parametrized by `num_collection_steps` + `num_envs` instead of module-globals)
- `src/qtris/training/ppo_loss.py` — `clipped_surrogate(ratio, adv, ppo_clip) -> (surrogate, clipped_ratio)` element-wise primitive; `clipped_value_loss(values, old_values, returns, value_clip) -> scalar` reduced loss
- `src/qtris/pretraining/base.py` — `surge_correction`, `correct_and_clip`, plus `RETURN_CLIP_LOW`/`HIGH` constants

Callsite migrations:
- 3 trainers each lose their inline `compute_gae_and_returns` (and ar/flat lose `compute_raw_returns`)
- 4 PPO-loss callsites collapse: ar.py, flat.py, _1v1.train_step_ar (sequence variant), _1v1.train_step_flat (scalar variant). The element-wise `clipped_surrogate` handles both — caller does its own reduction (mean for scalar, mask-then-mean-mean for sequence). Each site lost ~5 lines of `surr1`/`surr2`/`clip_by_value` plumbing.
- 4 value-loss callsites collapse to a single function call. ~8 lines saved each.
- 2 pretrainers' `_surge_correction` and `_correct_and_clip` `@staticmethod`s dropped; bodies replaced with `surge_correction(...)` / `correct_and_clip(...)` from the shared base module.

Net dedup: roughly 90 lines across trainers + 30 lines across pretrainers. No behavior change (math is bit-identical).

Wandb logging was also extracted in this phase under `src/qtris/observability/` (see Observability phase outcome above).

Skipped / not extracted in Phase 3:
- Trainer `train_step` loops themselves (entropy, kl, expert_loss, gradient apply) — too entangled with per-trainer logging and shape assertions. Phase 4-5 may revisit if a "training step base class" pattern emerges.
- Per-trainer dataset loader pipelines (Pretrainer `_load_dataset`) — structurally similar but differ in expected fields (AR uses `actions`/`masks`, Flat uses `action_indices`/`valid_masks`). Could share a skeleton in Phase 4+.

## Cleanup audit (2026-05-18)

- **ValueModel back-compat shim retired.** The re-export in `ar/model.py` (`from qtris.models.value import ValueModel`) has been deleted. All 4 callsites now import directly from `qtris.models.value`. The re-export in `flat/model.py` (was unused) also deleted.
- **Zero `from qtris.models.ar` in `qtris/models/flat/`** — the "flat stops cross-importing from ar" goal from Phase 2 is fully achieved.
- **No stale TODOs, phase-refs, dead imports, or back-compat comments** anywhere in `src/qtris/`.
- **`_1v1.py` USE_FLAT pattern** remains. It's functional (not back-compat), and replacing it is non-trivial (see notes.md Phase 1b deferred). Acceptable as-is until a natural rewrite opportunity.

## Phase 4 outcome (2026-05-26)

Shared demo modules created and wired:
- `src/qtris/demo/constants.py` — PIECE_COLORS, READABLE_KEYS, BCG_COLORS_RGB, BCG_LABELS
- `src/qtris/demo/rendering.py` — compute_bcg_heatmaps, draw_garbage_bar, colorize_piece_sidebar
- `src/qtris/demo/utils.py` — load_checkpoint, load_piece_display, save_frames_as_video

All 4 demos fully wired:
- Inline constant definitions (PIECE_COLORS, READABLE_KEYS, BCG_COLORS_RGB, BCG_LABELS) deleted
- BCG heatmap computation → `compute_bcg_heatmaps(scores)`
- Garbage bar → `draw_garbage_bar(py_env, height=24, width=10)`
- Sidebar colorization → `colorize_piece_sidebar(piece_display, pieces_array, PIECE_COLORS)`
- Frame recording → `save_frames_as_video(frames, "Demo.mp4")`
- Dead `import imageio` lines removed from ar/flat/vs (imageio now imported only in utils.py)

Line counts before/after (demo files only):
- ar.py: 540 → 456 (-84)
- flat.py: 521 → 446 (-75)
- vs.py: 993 → 896 (-97)
- ar_1v1.py: 498 → 434 (-64)
- Shared modules: +130 (constants 29 + rendering 58 + utils 43)
- Net: 2553 → 2362 (-191 lines, ~7.5% reduction)

vs.py still has the dominant-piece attention colorization inline (left + right
variants) because that block's output shape feeds into demo-specific surface
creation. Could extract as a `colorize_attention_scores(scores, pieces_array,
PIECE_COLORS)` helper in a future pass — marginal value (~20 lines × 2).

## Phase 5 outcome (2026-05-27)

`config.py` populated with 5 pydantic v1 models carrying canonical defaults:
- `ModelConfig` (10 fields: piece_dim=8, key_dim=12, depth=64, ...)
- `EnvConfig` (8 fields: max_holes=50, max_height=18, max_steps=512, ...)
- `PPOConfig` (13 fields: gamma=0.99, lam=0.95, ppo_clip=0.2, ...)
- `PretrainConfig` (5 fields: return_clip_low=-150, batch_size=512, lr=3e-4, ...)
- `DataGenConfig` (6 fields: search_depth=8, beam_width=96, ...)

Defaults match AR trainer values. Flat/1v1 override via construction:
`PPOConfig(num_collection_steps=256, target_kl=0.02, expert_coef=0.1)`.

Proof-of-concept wiring:
- `training/ar.py` — module-level constants now sourced from config models
  (unpacked to module scope for tf.function tracing compatibility)
- `data/dagger.py` — SimpleNamespace shim fields now sourced from config models
  (SimpleNamespace itself kept for args.X reference compatibility throughout body)

Remaining files can be wired incrementally using the same pattern.

Intentional per-family differences (NOT bugs — verified via survey):
- target_kl: 0.03 (AR) vs 0.02 (Flat)
- expert_coef: 0.005 (AR) vs 0.1 (Flat)
- num_collection_steps: 64 (AR) vs 256 (Flat/1v1)
- max_steps: 512 (AR/1v1) vs 1024 (Flat)
- search_depth: 8 (AR datagen) vs 7 (Flat datagen)
- dropout_rate: 0.0 (training) vs 0.1 (demo — higher for inference robustness)
- max_height: 18 (all) vs 19 (demo/vs.py — likely legacy; worth auditing)

## Stale/dup audit pass (2026-05-27)

- **Deleted dead `RewardNormalizer`** (`training/reward_normalizer.py`) — zero references; trainers compute `return_var` inline via EMA.
- **Extracted `PretrainerBase`** into `pretraining/base.py`. `Pretrainer` + `FlatPretrainer` now inherit `__init__`, `_load_dataset`, `load_expert_dataset` (~95 duplicated lines collapsed). Stale `DataGen.py`/`DataGenFlat.py` error strings fixed to generic "Run `uv run datagen`".
- **Wired `load_checkpoint`** (Phase 4 helper that was created but never used) into all 6 demo checkpoint-load sites: ar.py, flat.py, vs.py (×2 left/right), ar_1v1.py's `load_policy`.
- Dead-function scan now clean: no uncalled top-level functions anywhere in `src/qtris/`.
- ruff format + ruff check both pass.

NOT done this pass (deferred, marginal value): `colorize_attention_scores` demo helper (the dominant-piece colored_scores block still inline in ar/flat/vs×2).

## Dependency modernization (deferred — stack-level)

The dependency stack is dated and pinned together by `tf-agents==0.19.0`. To
modernize, the chain is:

1. `tf-agents 0.19.0` pins `typing-extensions==4.5.0`, which blocks `pydantic>=2`
   (needs typing-extensions>=4.6.1) and probably other modern libs too.
2. tf-agents is currently used by:
   - `src/qtris/runners/flat.py` — `ParallelPyEnvironment` (multi-process env collection)
     and `TFPyEnvironment` (TF wrapper around a py env)
   - All 4 `src/qtris/demo/*.py` — `TFPyEnvironment` for inference-time env wrap
   - `src/qtris/cli/train.py` — `tf_agents.system.multiprocessing.handle_main`
     bootstraps the multiprocessing protocol for `ParallelPyEnvironment`
3. Replacing tf-agents = re-implementing those wrappers + a multiprocessing
   manager for parallel env rollouts. Real scope (~few hundred LOC + parity gate
   confirming rollouts produce identical trajectories on a fixed seed).

Why bother:
- Unblocks **pydantic v2** (current syntax is v1 because of #1; observability
  module's `class Config: arbitrary_types_allowed = True` would become
  `model_config = ConfigDict(...)` plus other v1→v2 API changes).
- Unblocks upgrading **Python (3.11.11 → 3.12/3.13)** and **TensorFlow
  (2.15 → 2.18+)** — both are aging fast as of 2026.
- Removes a transitive dep that's barely maintained (tf-agents has had
  spotty releases for a couple of years).

When to act: probably after Phase 5, as its own follow-up project. Treat as
"modernization phase", not part of this refactor. Until then, stay on
pydantic v1.

## TFTetrisEnv vendored as git subtree (2026-05-28)

`tetrisenv/` is now a squashed git subtree of `m-sher/TFTetrisEnv` @ main
(ebb5b2a). The uv dep was switched from a git source to `{ path = "tetrisenv" }`
in `pyproject.toml`; the lockfile resolves it from `directory = "tetrisenv"`.

Gotcha hit on integration: upstream main had **committed stale `.so` binaries**
(`pathfinder`/`hole_finder` `.cpython-311...so`) inside `TetrisEnv/`. The wheel
build packaged them byte-identical instead of compiling fresh from the `.c`
sources, so the ctypes wrappers loaded a wrong-ABI library. Removed them from
the subtree and added `*.so` to root `.gitignore`. NOTE: these are plain C
shared libs loaded via `ctypes.CDLL` (no `PyInit_*`), not CPython extension
modules — `import TetrisEnv.pathfinder` is *expected* to fail; the real entry
points are `CKeySequenceFinder` / `CHoleFinder` / `CB2BSearch`.

Deferred:
- A future `git subtree pull` from upstream will re-introduce the committed
  `.so` files unless upstream deletes them. Fix at the source repo, or re-run
  the `.so` removal after each pull.
- Vendored `tetrisenv/` is non-idiomatic style; keep it out of ruff's scope so
  it isn't reformatted (would diverge from upstream + break subtree pulls).

## Triangle/TETR.IO bot migration (2026-05-28)

Migrated the two TETR.IO "Triangle" bot harnesses (kept in the gitignored
`tetrio/` dir, per user). Shared JSON-protocol / replay / move-loop code now
lives in `tetrio/_bot.py` (`TriangleBot`, `make_bot_env`, `PIECE_MAP`,
`KEY_MAP`); `tetrio/triangle_integration.py` (AR) and
`tetrio/triangle_integration_flat.py` (Flat) just build their model + load a
checkpoint + run the bot. Run with `uv run python tetrio/triangle_integration.py`.

Changes from the originals:
- Imports retargeted to `qtris.models.{ar,flat}.model` + the shared
  `qtris.demo.utils.load_checkpoint`; dropped the old `sys.path` hack + shebangs.
- Checkpoints retargeted to canonical `checkpoints/ar_policy` and
  `checkpoints/flat_policy`.
- Fixed a Flat bug: it built `_garbage_queue` as 2-tuples, but the current env
  uses 3-tuples `(rows, col, timer)`. Both bots now use 3-tuples.
- `pieces` is now sliced to `queue_size+2` in both (was AR-only); required
  because the bot runs `auto_fill_queue=False` and the client feeds a long queue.

2026-06-11: added `tetrio/triangle_integration_placement.py` — the AZ placement bot.
Subclasses TriangleBot, overriding only `_predict_keys`: PUCT MCTS (PlacementMCTS,
128 sims, lpr 8, eps 0 / greedy-by-visits) picks a descriptor on the live bot env;
descriptor -> dense action (is_hold*160 + rot*40 + norm_col*4 + spin) -> key sequence
via env._enumerate_placement_candidates(); dead root -> forced hard drop. Loads
`checkpoints/placement_az` latest (policy+value) + the frozen return_scale from the
ckpt (falls back to 1.0 for BC/PPO ckpts); diagnostics go to stderr (stdout is the
protocol). Protocol-smoked: config+play round-trip returns a sensible spin placement.

Deferred:
- `checkpoints/flat_policy` does not exist yet (flat PPO never ran), so the Flat
  bot runs on random weights until a flat policy is trained. Same blocker noted
  under "Dependency modernization" / flat-1v1 items.
- Preview path retraces/breaks if the supplied queue is shorter than
  `count + queue_size` (pieces drains below 7). Real client queues are long
  enough; only an issue for synthetic short-queue inputs.

## b2b_search reduce-search — Phase 1 findings (2026-05-29)

Goal: cut search budget (currently needs depth>=8/width>=96) while holding
quality. Priority survival > b2b > APP. Regime: both, garbage-weighted.
APP target ~0.9+ (see memory qtris-b2b-app-target). All experiments via the
existing `decompose` / `set_weight` / `run_eval_games` C entry points; scratch
harness in /tmp/b2b_exp (not committed).

### Baseline budget surface (run_eval_games, 10 seeds, 120 steps)
- **No-garbage:** 100% survival everywhere; b2b climbs to ~42 by d8; APP flat
  ~0.45-0.52 regardless of budget. Low budget already solves no-garbage
  (d5w48 -> b2b 38). The >=8/>=96 requirement is NOT from no-garbage play.
- **Garbage 0.15:** APP scales with budget. Target cell **d8w96 -> APP 0.734,
  mean-max-b2b 30.6, 100% survival, 51.7 ms/move**. d10 is WORSE than d8
  (d10w96 APP 0.673) — negative returns once depth exceeds queue length
  (speculative-bag branching adds noise). So effective depth ceiling ~8.
- ms/move scales steeply: d8w96 52ms, d8w48 31ms, d6w48 23ms, d5w48 20ms.

### H2 — chain_rollout dominates cost (CONFIRMED, biggest win)
Gating `b2b_chain_rollout` off (W_CHAIN_ROLLOUT=0) at d8w96/0.15 dropped
**51.5 -> 22.5 ms/move (~56% of per-move cost)** with ~unchanged quality
(APP 0.71->0.68, b2b ~19, survival fine). The rollout is the single most
expensive leaf term AND its reward (10*len, max ~50) is negligible vs
b2b_linear (~400). Drop it or shrink K and re-scale; bank a ~2x speedup or
reinvest into depth/width.

### Eval landscape (decompose, mean |contribution| of best placement, 110 states)
Garbage game: height 156.8, **b2b_linear 150.7**, b2b_sqrt 19.3, immobile_clear
8.1, holes 8.0, hole_ceiling 7.5, bumpiness 6.2, b2b_flat 4.7, app 2.0,
attack 1.8, tslot 1.4, then everything else <1. **cascade / break_ready /
near_death == 0 across all 220 states (garb+nogarb).**

Core structural problem: the score is dominated by `b2b_linear` (20*b2b, up to
+760) and `height` (quartic+avg, up to -270). Every tactical/priming term meant
to let SHALLOW search find bursts (immobile_clear, downstack, cascade, surge_pot,
tslot, attack, app, chain_rollout) is ~4-15x too small to reorder the beam
across states that differ by even one b2b level (=20). So the leaf gives almost
no tactical gradient — good play (spin setups, combo bursts, when-to-break)
emerges ONLY from deep multi-ply search watching b2b_linear/height/near_death
change. THIS is why it needs depth>=8/width>=96.

### Why b2b breaks (decompose on the 2 break events)
At depth-0 EVERY candidate prefers maintaining b2b (b2b_linear +360 at b2b=18).
The bot broke (sending 21 = surge18 + combo) only because deep search saw that
NOT clearing -> 4 garbage pushes up -> death. Breaks are survival-forced via
lookahead and are invisible to the leaf eval. The garbage-gated priming terms
(surge_pot, downstack) that should pre-signal the cash-out are too small
(~10-40) to flip the leaf ordering against b2b_linear.

### H1 — aspiration prune (analytical, partly tested)
`cheap_prescore` includes the dominant terms (b2b_linear, height, holes) but
omits the positive tactical bonuses (immobile_clear, downstack, cascade,
surge_pot, chain_rollout) — up to ~tens-to-150 in primed states — while
ASPIRATION_SLACK=15. So the prune can discard exactly the burst-ready states at
low width (where topk_floor is high). At d8w96 disabling it didn't help (wide
beam), so the bite is at LOW width. IMPORTANT: if Phase 2 scales the priming
terms up (below), the prune must add an optimistic positive bound to
cheap_prescore or it will discard them harder.

### Phase 2 edit list (grounded in the above)
1. **Drop/shrink `b2b_chain_rollout`** (56% cost, negligible signal). Remove the
   fn + W_CHAIN_ROLLOUT + s.chain_rollout_length, or K=5->2 and re-scale.
2. **Scale up the priming terms** (downstack/cascade/surge_pot/break_ready/
   immobile_clear) into the ~20-40 range (~1-2 b2b levels) so the beam prefers
   burst-ready states — the key lever for APP at reduced width. Keep them
   garbage-gated; survival is still enforced by height/near_death (>> these).
3. **Fix aspiration prune** (H1): add optimistic max-positive-bonus constant to
   `cheap_prescore` (true lower bound) or widen ASPIRATION_SLACK; re-derive after
   step 2. Enables lower beam width.
4. **Collapse the 3 b2b store terms** (b2b_flat + b2b_sqrt + b2b_linear) to one
   tuned curve (keep linear-dominant per the surge economy). Removes 2 weights.
5. **Remove the near-3 immobile pass + W_FUTURE_B2B** (doubly-gated tiny term,
   pure cost) — fold into the single immobile scan or delete.
6. **Consolidate/remove dead weights**: cascade/break_ready never fired at 0.15
   (re-check under heavier garbage before deleting); well/hole_forgive/
   wasted_holes/combo/max_single all <1 contribution. Be conservative; verify
   per-term before deletion (no-back-compat by end).
7. **Cap effective depth ~8** (d10 worse) or fix the speculative-branch noise.
8. **Fix the misleading tuning comments** (W_B2B_LINEAR "~12/25" vs 20;
   W_MAX_SINGLE "1.5" vs 0.5).

### Instrumentation left in b2b_search.c (tag: EXPERIMENT-GATE / EXPERIMENT)
- `ASPIRATION_SLACK` made non-const + added to W_TABLE (runtime-tunable, H1).
- `b2b_chain_rollout`, near-immobile pass, hole_ceiling gated on their weights
  (skip compute when weight==0, H2). The gates are real optimizations (keep);
  the W_TABLE ASPIRATION_SLACK entry is experiment-only — decide whether to keep
  it exposed or revert in Phase 2.

## b2b_search reduce-search — Phase 2 outcome (2026-05-29)

Two verified changes to `tetrisenv/TetrisEnv/b2b_search.c`; result: near-baseline
quality at 2.7-5.4x lower per-move cost.

1. **Removed `b2b_chain_rollout` entirely** (fn, W_CHAIN_ROLLOUT, BoardStats
   field, the compute call, both eval + decompose use-sites). It was ~56% of
   per-move cost for a reward (10*len, max ~50) negligible vs b2b_linear (~400).
   Ablation showed removal is quality-neutral-to-positive at every budget.
2. **ASPIRATION_SLACK 15 -> 60** (made non-const + added to W_TABLE so it's
   runtime-tunable). `cheap_prescore` under-estimates true score by the omitted
   positive bonuses (downstack/cascade/surge/immobile, tens of points), so at
   slack=15 the prune discarded burst/survival states at low beam width. 60
   covers the typical omitted mass; disabling entirely (1e9) is WORSE (junk
   crowds the beam). Recovers low-width garbage APP + survival.

Kept as permanent optimizations: the weight-gates on the near-3 immobile pass
(W_FUTURE_B2B) and hole-ceiling (W_HOLE_CEILING) — skip the scan when the term
is off. Fixed the misleading W_B2B_LINEAR tuning comment (cited 12/25/9.3 vs the
real 20 and the surge economy).

### Verification (run_eval_games, 24 seeds, 140 steps, garbage 0.15)
- orig baseline d8w96: surv 100%, b2b 30.6, APP 0.734, 52 ms/move
- new   d8w96:         surv 92%,  b2b 31.0, APP 0.741, 34 ms (1.5x; APP parity)
- **new d8w48:         surv 100%, b2b 22.2, APP 0.710, 19 ms (2.7x)**  <- recommended
- new   d6w48:         surv 92%,  b2b 18.7, APP 0.675, 9.6 ms (5.4x)
No-garbage unaffected (100% survival, b2b ~33-42 across these cells). End-to-end
`search()`+PyTetrisEnv smoke OK; decompose still returns 22 components; 28 weights.

### Rejected by experiment (do NOT do)
- **Scaling up priming terms** (downstack/cascade/surge_pot/immobile_clear ~2x):
  trades APP DOWN for b2b UP (d6w48 g0.15 APP 0.64->0.59). Those terms reward the
  SETUP, so amplifying them makes the bot hold/hoard rather than cash out — wrong
  direction for APP. Priming weights left at defaults.

### Deferred (analytically flagged, not done — would need cleaner benchmarks)
- Collapsing b2b_flat + b2b_sqrt into b2b_linear: low upside (monotonic-in-b2b,
  doesn't change same-b2b ranking), skipped to avoid unverifiable behavior shift.
- Culling never-firing terms (cascade, break_ready at 0.15): they target HEAVY
  garbage downstacking; not removed without a heavy-garbage survival check
  (survival-first). well/hole_forgive/wasted_holes/combo/max_single are <1
  contribution — consolidation candidates if a knockout sweep is later run.
- The leaf eval still has little tactical gradient (good play emerges from
  lookahead, not the leaf); raising APP further at low budget likely needs a
  proper hold-vs-cash-out tuning pass with a low-noise benchmark, not done here.
- Decide whether to keep ASPIRATION_SLACK exposed in W_TABLE (it's a search knob,
  not an eval weight) or move it to a dedicated setter.

## b2b_search combo tracking + APP analysis (2026-05-29)

Goal: push APP toward 1.0. User insight: combo is the primary way to hold a
baseline APP while building b2b for the surge (combo multiplies attack:
`floor(base*(1+0.25*combo))`; b2b-maintaining spin-single clears chained = combo
AND b2b growth).

Added **combo tracking** to `run_eval_games`: `GameResult.max_combo` and
`avg_combo` (mean combo over clearing placements). Mirrored in CB2BSearch.py
GameResult struct. Harness prints `cmb(mx/av)`.

### Measured: the bot barely combos
- d8w96 g0.15: max_combo ~2.4-3.8, **avg_combo ~0.34-0.53**.
- d8w96 g0.00: max_combo ~1.5, **avg_combo ~0.15**.
Most clears are isolated (combo 0-1). This is the direct cause of the no-garbage
APP ceiling (~0.47): isolated b2b spin-singles deal ~1-2 attack each, no multiplier.

### W_COMBO scaling is NOT the unlock (tested 2.5/10/25/60 @ d8w96 g0.15, 14 seeds)
avg_combo rises only 0.38 -> 0.53; APP stays noisy ~0.70-0.76; b2b dips slightly.
COMBO_CAP=6 is not binding (avg combo << 6). The bot simply doesn't pursue
combos, and nudging the leaf combo weight doesn't change the strategy.

### Why (analysis)
Sustaining a combo means clearing a line EVERY piece. A b2b-maintaining combo
needs each piece to land a spin-clear, which requires a board with multiple
simultaneous spin-clear slots — structure the eval doesn't cultivate. The combo/
downstack/cascade "combo machinery" (downstack/cascade/surge_pot/well_aligned) is
garbage-gated (garbage_mul=0 in no-garbage), so in low-garbage play NOTHING
cultivates combo cascades. And boosting those terms rewards HAVING slots (setup)
-> hoarding, which measured APP DOWN, not comboing. So APP->1.0 via combo is a
strategic redesign, not a weight tweak. Candidate direction (needs a tuning pass
with a low-noise benchmark): a combo-CONTINUATION reward that values "just cleared
AND another spin-clear slot is ready", and/or partially ungating the downstack/
cascade cultivation in no-garbage — both require careful balancing vs b2b-hold.

## b2b_search combo reward + smooth garbage — outcome (2026-05-29)

Implemented the combo-reward + smooth-garbage design. Result: the requested
architecture change landed and is Pareto-safe with a MODEST APP gain at realistic
garbage; the APP->1.0 goal is NOT reached (structural — see below).

### Changes (`tetrisenv/TetrisEnv/b2b_search.c`)
1. **Smooth garbage signal replaces the hard gate.** `garbage_mul` (hard 0/1
   clamp) gone. New `burst_scale = clamp(GARBAGE_BURST_BASE + W_GARBAGE_URGENCY
   * garbage_remaining, 0, GARBAGE_BURST_MAX)` multiplies downstack / well_aligned
   / cascade / surge_pot / break_ready. `future_b2b` uses smooth
   `1/(1+W_GARBAGE_URGENCY*garbage_remaining)` instead of `(1-garbage_mul)`.
   Tuned defaults: BASE=0.35, URGENCY=0.18, MAX=1.1 — so burst terms fire at a
   small baseline with no garbage (cultivation) and rise to ~old strength by
   ~4 garbage lines, NOT amplified beyond it (the first try with MAX=2 regressed
   survival). Accepting garbage is never penalized; its risk is priced only by
   the survival height gradient (effective_h includes garbage_remaining).
2. **Combo-continuation reward** (new `W_COMBO_CHAIN`, default 3.0): when a combo
   is alive (`combo>=0`) and a b2b-maintaining spin-clear slot is ready, add
   `W_COMBO_CHAIN * min(immobile_clearing, COMBO_CHAIN_CAP=4) * (1+combo)`. Keys
   off immobile (b2b-maintaining) slots so it rewards CHAINING not hoarding; pays
   nothing on a dead-combo board. `COMBO_CAP` raised 6 -> 12.
3. Decompose mirror + `cheap_prescore` updated: cheap adds OPTIMISTIC bounds for
   the combo-chain and burst terms (using combo/b2b/garbage it already knows) so
   the aspiration prune can't discard mid-combo / burst-primed states.
4. New tunable weights in W_TABLE: W_COMBO_CHAIN, GARBAGE_BURST_BASE,
   W_GARBAGE_URGENCY, GARBAGE_BURST_MAX (count 28 -> 32).

### Verification (run_eval_games; OLD reconstructed in-binary via weights)
- **d8w48 g0.15, 24 seeds:** OLD APP0.701 b2b21.2 surv95.8 combo0.52 ->
  **NEW APP0.717 b2b21.0 surv100 combo0.57** — Pareto win (APP+combo+survival up,
  b2b held).
- d8w48 g0.20, 24 seeds: APP/b2b/survival held (0.70/16->17/87.5 both).
- d6w48 g0.20, 16 seeds: APP0.616->0.655, b2b 9.6->15.1, surv 100 both — win.
- **No-garbage: unchanged** (combo ~0.14, APP ~0.47, b2b ~37, surv 100).
- b2b_test smoke (d8w48, garbage): 3/3 survived, APP 0.662, sensible.

### Honest assessment — APP->1.0 NOT achieved
Combos are structurally a DOWNSTACK phenomenon: they need consecutive line
clears, which only happen when there are near-full rows to clear (garbage / built
cascades). No-garbage b2b-building is isolated spin-singles with nothing to chain,
so avg_combo stays ~0.14 and APP ~0.47 regardless of the combo reward. Under
"max-b2b strictly first" the combo term can't be scaled enough to force chaining
without risking b2b/survival, so the gain is confined to the garbage regime and
is small (~+0.016 APP at g0.15). The smooth-garbage refactor is the durable win
(cleaner, tunable, survival-neutral, cultivates burst setups in no-garbage too).

### To actually push APP toward 1.0 (future, bigger effort)
- Relax "max-b2b strictly first" to permit combo bursts (cash-out), OR
- Add a dedicated combo-structure objective (perpetual-spin / staircase) that
  builds boards where consecutive pieces each spin-clear — a real search/heuristic
  project, needs a low-noise benchmark to tune. Both flagged, not done.

### Cheap combo-SETUP-potential term — TRIED and REVERTED (2026-05-29)
Per "encourage states with b2b-building combo potential, keep b2b strictly first,
cheap structural signal only (no rollout, no height-forgiveness)": added a term
rewarding `combo_potential` = # distinct rows completable by a b2b-maintaining
(immobile/spin) placement, when >=2, faded by (1-h_ratio). Computed cheaply by
unioning cleared-row bits in the existing immobile scan (popcount).

A/B (W_COMBO_SETUP 0 vs 4, 16 seeds): **failed the Pareto-safe gate.**
- No-garbage: avg_combo FLAT (0.15->0.14), APP slightly DOWN — the height penalty
  vetoes building setup structure, so the term does nothing (predicted).
- Garbage 0.2: survival REGRESSED (d8w48 87.5->81.2, d6w48 100->93.8) and combo
  went DOWN (0.58->0.48) — it distorts toward riskier structure WITHOUT producing
  combos. "Potential" (>=2 spin-clearable rows) does not convert to actual combos:
  clearing one row often destroys the other's slot, so the cheap proxy can't see
  chainability, and nothing lets the bot build the substrate (height penalty).
Fully reverted (term + W_COMBO_SETUP + combo_potential/distinct_clearable_rows
plumbing). Conclusion: combo-potential needs BOTH a chainability check (a shallow
spin rollout, option B) AND height-forgiveness for about-to-clear rows (option C);
the cheap structural signal alone (option A) is insufficient and not Pareto-safe.

## b2b_search spin-channel (reusable-overhang) — TRIED, DISABLED (2026-05-29)

Pursued the mechanically-correct combo design: a b2b-building combo needs a
REUSABLE OVERHANG — a covered channel (gap column under a roof, near-full rows
alongside) where each spin completes+clears a row, the stack drops, the roof
descends over the next gap. Implemented in `tetrisenv/TetrisEnv/b2b_search.c`:
- `detect_spin_channel()` — finds the deepest COVERED (roofed), currently
  spin-clearable (intersect `clearing_cells`) run of >=9-filled rows sharing one
  gap column. New BoardStats `spin_channel_depth` + a `channel_cells` mask.
  (This uses the previously-UNUSED `blocked_9_rows` insight: covered near-full
  row == spin-required row == the combo substrate.)
- `W_SPIN_CHANNEL` reward (depth, (1-h_ratio)-faded).
- **Hole forgiveness**: channel cells passed to count_hole_sections /
  count_deep_holes / compute_hole_ceiling_weight instead of `no_exempt` (holes
  were previously penalized UNIFORMLY — W_HOLES up to ~70 vs spin reward ~8 — so
  building any covered structure was net-penalized; this was THE blocker).
- `W_CHANNEL_HEIGHT_RELIEF`: discount the channel's volume from avg_height ONLY
  (max_height/cliff untouched), so the linear volume penalty doesn't veto building.
- cheap_prescore optimistic bounds + decompose mirror; master switch (weight=0
  short-circuits detection + forgiveness).

### Result: FAILS — and reveals the real obstacle (no construction gradient)
A/B (channel on vs off, 16 seeds, d8w48+d6w48, garb 0.0/0.2):
- No-garbage: NEUTRAL, combos FLAT (~0.13->0.14), board stays low (~3.7) — the
  bot never builds a channel.
- Garbage: APP DOWN (~0.67->0.65, 0.70->0.64), combos ~flat, survival sometimes
  up — fails the Pareto gate (APP must rise).
- **Even with hole-forgiveness AND height-relief AND aspiration pruning disabled
  (slack 250), the no-garbage bot STILL won't build channels (height 3.8, combo
  0.11).**

Root cause: **there is no reward gradient along the CONSTRUCTION path.** The
channel reward only pays once a depth>=2 covered+live channel EXISTS, but every
intermediate step toward one (digging a covered hole, raising height, an overhang
in progress) is penalized and unrewarded. A greedy per-state heuristic + bounded
beam can't climb to the structure — the path is all-downside, only the
destination pays. Forgiving the destination's holes/height doesn't help because
the bot never reaches the destination.

Implication: a shallow spin-rollout likely ALSO won't help — it measures clearing
a structure that already exists, not the multi-step construction toward one. So
b2b-building combos appear to be **beyond what a hand-heuristic + greedy beam can
induce in this framework** (it's the kind of multi-step setup an RL policy would
learn, not a per-state eval term).

### Decision / state
Feature is **DISABLED by default** (`W_SPIN_CHANNEL=0`, master-switch short-
circuits all of it) so shipped behavior == the prior Pareto-safe state (smooth
garbage + combo-continuation). Code kept + tagged, pending a decision: remove it,
or pivot to a construction-aware approach (much bigger: RL, or a dedicated
multi-step setup search). The exhaustive negative results (generic combo
continuation, combo-setup, spin-channel) all point to the same construction-
gradient wall — recommend NOT spending more on per-state combo heuristics.

## b2b_search APP reframe + NON-DETERMINISM finding (2026-05-29)

### APP heuristic reframe (implemented, kept)
Per user suggestion, changed the APP term from `total_attack/pieces` to
`(total_attack + max(0, b2b)) / pieces` in evaluate_state, cheap_prescore, and
decompose (D_APP).  Rationale: stored b2b is pending surge attack (surge ~= b2b),
so APP now rewards efficient b2b-building-PER-PIECE + realized attack as one
quantity.  W_APP kept at default 3 (the formula change is the win; W_APP 3 vs 15
differ within noise and 3 keeps survival higher).  Appears to be a mild
Pareto-safe gain (b2b up, APP up slightly, survival held) but see noise caveat.

### CRITICAL: the parallel beam search is NON-DETERMINISTIC
Identical input (same seeds, weights, depth, beam) gives DIFFERENT results
run-to-run under OpenMP: measured d8w96 g0.15, 16 seeds, 4 runs ->
b2b {31.1, 30.0, 23.6, 27.2} (range 7.5), APP {0.695,0.741,0.745,0.731}.
OMP_NUM_THREADS=1 is bit-identical (b2b 18.0/18.0). Root cause in the
`#pragma omp parallel` beam expansion (b2b_search.c ~line 3460+):
- per-thread `topk_reset`/`g_topk_heap` => each thread prunes children against
  its OWN aspiration floor => non-deterministic SET of inserted children;
- `__atomic_fetch_add(&next_count,1)` slot assignment is thread-timing ordered;
- truncation (`next_count > max_next`) + top-K ties keep a race-dependent set.

Implications:
1. **Every A/B in this session was corrupted by this noise** (±0.05 APP / ±7 b2b).
   Most "neutral/mild/failed" combo conclusions are within the noise band — they
   are NOT reliable. Combo tuning cannot proceed reliably until this is fixed.
2. **Play robustness bug**: the bot sometimes makes much worse choices (b2b 23.6
   vs 31.1) purely from thread timing.

### Recommended next step (enabler for everything else)
Make the parallel beam DETERMINISTIC while keeping it fast: e.g. schedule(static)
parent partition + per-thread child buffers, a serial deterministic merge, a
single deterministic aspiration bound (not per-thread), and a stable top-K
tiebreak (score, then state hash). Then re-run the APP-reframe + combo A/Bs with
trustworthy numbers. OMP=1 tuning is not viable (~16x slower).

### Combo status (blocked on the above)
The spin-channel / fuel work (v1 stacked channel, v2 graded covered-near-full-row
fuel) is gated off (W_SPIN_CHANNEL=0). v2 DID make the bot build fuel (gradient
works) and raise combo, but combos broke b2b mid-chain AND the measurements were
inside the noise band. Re-evaluate combo levers once the search is deterministic.

## b2b_search beam DETERMINISM fix (2026-05-29) — major

Fixed the non-determinism (was b2b ±7.5 run-to-run). Now 4 identical-config runs
at 16 threads give bit-identical results (APP 0.6750, b2b 26.12, surv 100, every
run). The search is reproducible AND the play is robust (no thread-timing degraded
games). This is the enabler for reliable eval tuning.

How: removed the per-thread aspiration prune (the main nondeterminism source) and
made beam selection a strict TOTAL order.
- `expand_and_insert`: dropped cheap_prescore/topk aspiration; every surviving
  child gets a full eval (strictly MORE thorough) and caches `sort_hash`
  (state_hash) on the SearchState.
- New `finalize_beam`: qsort by (score desc, sort_hash, depth0_idx) -> dedupe
  adjacent equal-hash -> keep top beam_width. Replaces dedupe_beam +
  beam_select_top_k at both the depth-0 and per-depth sites. Final best-pick uses
  the same `beam_cmp`.
- Removed now-dead: `cheap_prescore`, `topk_*`/`g_topk_heap`, `ASPIRATION_SLACK`
  (+ W_TABLE entry + extern), `dedupe_beam`, `beam_select_top_k`, `beam_sift_down`,
  `HashSlot`, the `hash_table` alloc/frees. Net simplification.

Cost: ~+12% per-move (d8w96 33->37 ms) from full-evaluating all children (no
aspiration). Well within the chain_rollout-removal headroom.

KEY consequence: deterministic = the CORRECT top-K, landing at APP ~0.675 at
d8w96 g0.15 (16 seeds) — the eval's HONEST quality. The noisy runs that hit 0.74+
were eval-suboptimal moves getting lucky game outcomes. So the eval caps APP, and
it can now be tuned with a clean per-change signal. (All earlier A/B conclusions
in this file that hinged on <~0.05 APP / <~7 b2b deltas were within the old noise
band and are NOT reliable; re-tune on the deterministic build.)

Next: re-validate the APP reframe + revisit combo levers on the now-trustworthy
measurements. Determinism also means single 16-seed runs are reliable (no
run-to-run noise; only cross-seed generalization variance remains).

### APP reframe re-tuned on the deterministic build (W_APP 3 -> 10)
Clean deterministic sweep, d8w96 g0.15, 16 seeds:
  W_APP=3  -> APP0.702 b2b30.7 surv100 cmb0.45
  W_APP=10 -> APP0.781 b2b35.2 surv100 cmb0.53   <- new default
  W_APP=20 -> APP0.766 b2b31.8 surv100 cmb0.44
  W_APP=40 -> APP0.782 b2b30.6 surv 81  (survival breaks)
So W_APP=10 (with the (total_attack+max(0,b2b))/pieces reframe) is a reliable
Pareto win: APP 0.70->0.78, b2b 31->35, combo 0.45->0.53, survival 100%; combo
rising shows the reframe induces more b2b-building combos (the intended effect).
Set as default. End-to-end b2b_test smoke (real env, d8w96, garbage): 3/3
survived, APP 0.747. Spin-channel fuel work stays gated off (W_SPIN_CHANNEL=0)
pending revisit on the deterministic build.

### Deterministic combo-lever scan on top of W_APP=10 (d8w96 g0.15)
Run at 50 steps for speed, then validated the best at 100 steps (50-step results
mislead — b2b is still building). Single-lever deltas vs base:
- W_B2B_ATTACK=8: 50-step looked like +0.07 APP, but at 100 steps it's a
  NO-GARBAGE-only lever — garbage b2b 26.8->21.2 (-21%) with flat garbage APP,
  while no-garbage APP 0.495->0.561. It rewards CASHING b2b (attack while alive)
  over building it. REJECTED for the garbage priority (b2b is priority 2).
- generic W_COMBO, W_BLINEAR up: neutral. W_SPIN_CHANNEL=4: combo up, APP flat-to-down.
Conclusion (now reliable): the APP reframe (W_APP=10) is the lever that works for
the garbage priority; the combo-specific levers don't cleanly beat it there.
b2b_attack is available as a no-garbage APP boost IF that regime is wanted (costs
garbage b2b). Lesson: decide on >=100-step runs, not 50.

### Combo IS fuel-driven, but the bot's garbage combos BREAK b2b (det., 100 steps)
base (W_APP=10) d8w96 by garbage_chance:
  g0.15 APP0.725 b2b26.8 surv100 cmb0.49
  g0.30 APP0.775 b2b12.9 surv 88 cmb0.85
  g0.45 APP0.618 b2b 4.2 surv 38 cmb1.11
Combo rises to >1.0 with more garbage -> the bot CAN combo; combos are fuel-driven.
BUT b2b collapses as garbage rises (27->13->4): the bot downstacks garbage with
b2b-BREAKING regular-clear combos, then drowns at heavy garbage. The reference bot
instead downstacks garbage with b2b-MAINTAINING spin-clears (high combo AND high
b2b). Re-tested spin-channel deterministically (chan8 g0.15): combo 0.49->0.58 but
b2b 26.8->18.8, APP 0.725->0.629 — confirms (not noise) that the channel induces
b2b-BREAKING combos. So the precise unsolved capability is: SPIN-CLEAR garbage
(b2b-maintaining) rather than regular-clear it. A regular garbage clear is 1 piece
(drop into the open gap); a spin-clear needs the gap COVERED first (2 pieces:
cover then spin) -> slower -> under garbage pressure the bot regular-clears to
survive. Every combo reward lever tried (channel, b2b_attack, generic combo)
deterministically induces MORE b2b-breaking combos, not b2b-maintaining ones.
NOTE the original bot already had chain_rollout (K=5) and still combo'd ~0.45 — so
the rollout is NOT the missing lever.

### Tried (det.) toward spin-clear-garbage — all per-state rewards HOARD
- W_COVERED_CLEAR (reward blocked_9_rows = covered/spin-clearable near-full rows):
  at g0.15 combo DROPPED 0.50->0.39, APP flat. Rewarding the covered-row STATE
  makes the bot HOLD covered rows (not clear) -> fewer clears -> lower combo.
  Same hoarding pathology as channel/fuel. Gated OFF (W_COVERED_CLEAR=0 default).
- Removing the open-gap/downstack bias (W_DOWNSTACK/WELL_ALIGNED/CASCADE=0):
  neutral at g0.15, slightly better survival at g0.30, but b2b stays ~14 under
  garbage. Didn't unlock b2b-maintenance either.
Consistent conclusion: per-STATE structural rewards can't induce the multi-step
b2b-maintaining spin-clear EXECUTION — rewarding the setup state -> hoarding.

### ROOT-CAUSE LEAD: the search models incoming garbage as SOLID
`push_simulated_garbage` (the SEARCH's speculative garbage, b2b_search.c ~892)
fills FULLY SOLID rows `(1<<10)-1` (no gap), while the actual game
(`garb_push_one`) pushes garbage WITH a gap at one column. And b2b_search_c only
receives `total_garbage` (a count), not gap columns. So when the search plans
ahead, incoming garbage is UNCLEARABLE height — it literally cannot plan to clear
(let alone spin-clear) incoming garbage, only react to gaps already on the root
board. This structurally biases the search against b2b-maintaining garbage
downstacks and likely caps combo-under-garbage regardless of eval weights —
explaining why no per-state reward lever helped.

FIX (next, meatier — spans C API + env): thread the garbage gap column(s) into
b2b_search_c (extend CB2BSearch.search + PyTetrisEnv to pass them) and model the
gap in push_simulated_garbage, so the search can PLAN b2b-maintaining garbage
spin-clears. This is the most promising remaining lead and the only one that's a
concrete model fix rather than a reward tweak. Experimental combo weights
(W_COVERED_CLEAR, W_SPIN_CHANNEL, W_CHANNEL_HEIGHT_RELIEF) left gated off (=0);
remove if not pursuing.

### Search-side executed-combo reward ALSO fails (det., 100 steps)
Tried W_B2B_COMBO = reward max_b2b_combo (longest run of CONSECUTIVE b2b-maintaining
clears EXECUTED along the search path; path-cumulative, anti-hoarding — earned only
by executing, not by holding a setup). Result g0.15: Wbc=0 APP0.732/b2b28.5/cmb0.50;
Wbc=8 APP0.660/b2b23.4/cmb0.50; Wbc=20 APP0.694/b2b26.1/cmb0.51. Combo does NOT rise
(0.50->0.51) and APP/b2b drop. Decisive: longer b2b-maintaining combos are NOT
REACHABLE in the search tree (no continuous-spin structure to chain through), so
rewarding the metric just distorts. The limit is reachability, not the reward
formulation. (W_B2B_COMBO + the b2b_combo_len/max_b2b_combo SearchState tracking
left in, gated off at 0.)

### Overall combo conclusion (deterministic, robust)
Across BOTH per-state rewards (channel/fuel/covered-rows) AND path-cumulative
rewards (b2b_attack, max_b2b_combo) AND bias removal: none induces b2b-maintaining
combos. Root reasons: (a) per-state setup rewards -> hoarding; (b) continuous
b2b-spin-combo paths aren't reachable in the search (the structure isn't there and
building it hoards); (c) under garbage the 2-piece cover+spin rate can't keep pace
-> forced b2b-breaking regular clears. The reference bot's capability isn't
reproducible via reward tuning on this eval+search. SOLID WINS that stand:
deterministic search + APP reframe (W_APP=10): garbage d8w96 APP 0.73->0.78, b2b up,
survival 100%. Recommend consolidating these; the combo capability needs a method
beyond reward shaping (not identified).

## Datagen pivot to fusion-style search-aligned targets (2026-05-30)

Pivoted AR datagen from RL (behavior-clone best move + MC discounted-return value)
to fusion's supervised search-aligned lane: score EVERY candidate root move and use
the best score as the value target.

C (`b2b_search.c`): `b2b_search_c` gained optional per-root outputs
(`max_roots, out_num_roots, out_root_action_indices, out_root_scores,
out_root_sequences`). For every root placement, it reports the action index, RAW
best-descendant score, and reconstructed key sequence; value target = max score.
- Tracks each root's best score across ALL depths (recorded pre-prune at depth 0 +
  every depth-loop iteration) into `root_best[]`, gated on `want_roots`. This is
  observe-only — it does NOT touch the beam/pruning, so move quality + determinism
  are unchanged (verified: run_eval_games bit-identical across runs). Needed because
  the deep search converges HARD to one root (num_candidates=1 at d16/w200 if you
  only read the final beam); cross-depth tracking gives the full ~34-70 legal-move
  distribution instead.
- run_eval_games passes `0,NULL,NULL,NULL,NULL` (no per-root output).
- NO softmax in C/datagen — raw scores stored; caller normalizes.

CB2BSearch.py: argtypes extended; added `search_with_scores(...)` ->
(best_action, best_seq, cand_action_idxs, cand_scores, cand_sequences).

Datagen (`data/gen_ar.py`): REWRITTEN. Per position stores boards/pieces/
b2b_combo_garbage + `cand_sequences[64,15]` + `cand_scores[64]` (raw, PAD/0) +
`num_candidates` + `value`(=max score). MAX_CAND=64 (top-by-score). Dropped the
episode rollout / discounted-returns / death-trim entirely (value is the search
score now; reward function irrelevant). Existing-dataset merge keyed on
`cand_scores`; old-schema datasets start fresh. beam 200 / depth 16.

Demo (`b2b_demo.py`): headless prints the softmaxed top-5 move distribution per
turn; GUI renders a "MOVE DIST" panel with probability bars. Softmax is
viz-only. Added `softmax_dist()` + `decode_action()` (160-based encoding; also
fixed the GUI's stale //80 decode). Added a temperature knob (`--dist-temp` /
GUI "Dist Temp" spinner, default 30) — raw scores are O(tens-hundreds) so temp 1
is near-argmax; temp scales the spread. NOTE: the demo softmaxes only the TOP-5
(readable panel), so its temp is NOT comparable to a full-candidate softmax.

### Softmax-temperature sharpness measurement (informs the trainer, not the data)
CAVEAT measured on a 25-step set: best-move prob over ALL ~64 candidates is much
softer than a top-5 softmax at the same temp — T20: 94%/95%, T50: 61%/79%,
T100: **20%**/54%, T150: 9%/42% (all-cand / top-5). So a full-candidate temp 100
is a SOFT target (best move only ~20%). This is a TRAINER hyperparameter (applied
on the fly to cand_scores), NOT stored in the dataset — picked T~50 -> best ~61%
if a peaked target is wanted. (Briefly stored a per-row `policy_temp`/`value`/
`num_candidates`; REMOVED — all derivable from cand_scores, so the oracle dataset
stores only non-derivable outputs.)

### Dense 320-action target (gen_ar) + demo full-candidate softmax
Pivoted gen_ar from a packed top-64 candidate list to a DENSE action-indexed
target: per position, `cand_scores[320]` (float32, sentinel -1e30 for illegal/
unreached) + `cand_sequences[320,15]` (int8, PAD=11) indexed directly by action
index (is_hold*160+rot*40+norm_col*4+spin). Every position carries a score for all
320 possible placement-sequences (illegal masked by sentinel; softmax illegal
mass = 0, verified). NO C change — `b2b_search_c` already returns per-root
(action_idx, score, seq) for all legal roots via cross-depth root_best
(max_roots=512 >> ~120 raw placements); gen_ar SCATTERS into the dense array,
deduping colliding action indices by max score. Kept C untouched on purpose (it's
open in the IDE; a prior IDE-undo clobbered C edits). Storage ~6KB/pos (int8
seqs). Old packed-64 datasets auto-invalidated on merge (shape[1] != 320).
LEAN SCHEMA — dataset stores ONLY oracle outputs: boards, pieces,
b2b_combo_garbage, cand_sequences, cand_scores. The trainer derives the rest on
the fly: value target = cand_scores.max(), legal mask = score > -1e29, softmax
temperature. (No policy_temp/value/num_candidates stored — all derivable.)
Demo `softmax_dist` now softmaxes over ALL legal moves (deduped by action idx) to
MATCH the trainer's dense target — top-5 shown but probs over the full set (bars
sum <100%). Headless line shows "top5 of N legal". Demo dist-temp default stays 30
(its own viz knob).

### DAgger reworked to the dense gen_ar schema
`data/dagger.py` rewritten: rolls the trained policy forward (greedy, valid-seq
masked) and steps the env with the POLICY's choice (DAgger invariant), but labels
each visited state with `search_with_scores` -> the SAME dense 320-action target
as gen_ar (boards, pieces, b2b_combo_garbage, cand_sequences[320,15] int8,
cand_scores[320] f32). Dropped the entire old label/mask/returns/sample_weights/
gamma/death_trim/episode-buffer machinery and the ar/flat label branching — the
dense target is family-agnostic. `family` now only selects the rollout policy
(ar vs flat checkpoint). The dense scatter is shared via `gen_ar.dense_target()`
(single source of truth; gen_ar.collect uses it too). Search budget hardcoded to
depth16/beam200 to match gen_ar so labels are equivalent and datasets accumulate.
Was BROKEN before (imported gen_ar._build_mask, removed in the gen_ar rewrite).
Smoke (ar_pretrained_policy, 8 steps): 5-field dense save, policy≠beam=2, legal
68-70/row. Fixed: str() the --output PosixPath before tf.data.save.

### Datagen progress bars
All three collectors (gen_ar, gen_flat, dagger) wrap the step loop in a tqdm bar
with per-`log_every` postfix (transitions/deaths/max_b2b...). `datagen --headless`
disables the bar and emits the old periodic `Step N/M | ...` print lines instead
(for non-tty/log-file runs). Flag threaded CLI -> each main -> collect(headless=).

### AR pretrainer -> dense-target distillation (AR only; flat untouched)
`pretraining/ar.py` `_train_step` rewritten from hard next-token CE -> SEQUENCE-
LEVEL distribution distillation over the top-K candidate moves. Policy target =
softmax(top_k(cand_scores)/policy_temp); model dist = softmax of per-sequence
log-probs; loss = soft-CE (== KL + const). The OOM-avoidance hinges on the model's
encoder/decoder split: `process_obs` (board ViT, O(batch)) runs ONCE; only the
cheap `process_keys` key decoder pays the K multiplier (tile piece_dec to B*K,
score all K seqs as one batch). Diagnostics: Top1/Top3 candidate agreement.
VALUE: dropped returns/surge/correct_and_clip; target = max(cand_scores) scaled by
`_value_scale` (std of per-pos max, computed in new `base._load_dataset_dense`).
CLI: `--cand-topk` (32), `--policy-temp` (50). Value ckpt var return_scale ->
value_scale (old ar_pretrained_value ckpts incompatible; retrain).
base.py: added `_load_dataset_dense` + `_value_scale`; legacy `_load_dataset` /
surge_correction KEPT for the flat pretrainer (coexist until flat+gen_flat pivot).

MEM (RTX 3080 Ti 12GB, depth64/K32): batch 256 OOMs; 128 -> 5.8GB, 96 -> 4.3GB,
64 -> 2.9GB. Lowered CLI batch default 512 -> 128 (the batch*K decoder load is new;
flat has no multiplier, pass 256-512). Smoke (depth32/bs16, 40 steps): policy
6.4->5.0, value 1.5->0.78 both falling, XLA-compiled, no crash.
WATCH: on a 400-step set the per-pos max-score mean was ~2127 (high-b2b states push
scores into the thousands, not "tens-hundreds"). Intra-position candidate GAPS (not
absolute scale) set softmax sharpness, but temp 50 may be sharper than the earlier
25-step estimate suggested -- eyeball via the demo's full-candidate softmax at
temp 50 on real data, tune `--policy-temp` if needed.
NOTE: never run `pretrain ar` against default ckpt dirs to smoke -- it saves over
checkpoints/ar_pretrained_policy at epoch end. Use an inline _train_step loop.

### AR distillation v1 was BROKEN -> fixed (WRONG OBJECTIVE; temp secondary)
Symptom: many gen+dagger rounds, latest 20-epoch pretrain had flat loss ~2.2 +
flat Top1 ~0.65, demo extremely poor. NOTE the loss VALUE is uninformative: soft-
distillation bottoms out at the target ENTROPY (~2 here) for any objective, so
"loss ~2" just means converged-to-floor, not broken.
ROOT CAUSE = wrong objective. v1 used a candidate-set softmax over per-sequence
log-probs: softmax_k(seq_logp_k) KL'd to softmax(score/temp). That only constrains
RELATIVE ranking of the 32 seqs -> the per-token probs greedy decoding uses are
underdetermined (good rank, bad generation), and summing logp length-biases
AGAINST longer spin seqs (cand key-len 2..10). Replaced with per-token weighted CE
(joint seq KD): policy_loss = mean(-sum_k softmax(score/temp)_k * seq_logp_k) --
trains the per-token policy directly, no length bias.
CONTROLLED A/B (same init/data/steps, depth48, metric = `Gen` greedy per-token acc
along best cand = demo-relevant; `Rank1` = best cand ranked #1):
  softmax   T50:  Gen 0.255  Rank1 0.278
  softmax   T10:  Gen 0.247  Rank1 0.264
  weightCE  T10:  Gen 0.515  Rank1 0.031
=> objective DOUBLES Gen (0.25->0.52); softmax wins Rank1 but loses Gen (the exact
failure: ranks fine, generates badly -> why Top1~0.65 looked ok but demo broke).
TEMP was a red herring for generation: T50->T10 barely moved softmax Gen
(0.255->0.247). It only explained the flat LOSS (entropy floor). Kept default
temp 50->10 anyway (sharper target -> sharper marginal, secondary), CLI-tunable.
Diagnostics relabeled `Gen`/`Rank1`. NO data regen (dense schema unchanged); just
retrain from a clean policy ckpt (old weights trained on broken objective).
CAVEAT: A/B is directional (small model/data); real proof = train to convergence
+ run_eval_games / demo. Gen is teacher-forced (proxy for free-running).

pyproject: `tetrisenv` source set `editable = true` so source edits + in-place
`.so` rebuilds propagate to `uv run` entrypoints (datagen was importing a stale
installed copy). `uv sync` applied.

### NOT done yet (next step) — trainer is now incompatible
`pretraining/ar.py` still expects `actions`/`masks`/`returns` and will break on the
new dataset. To finish the pivot: policy loss -> KL/soft-CE over the candidate
sequences weighted by `cand_scores` (vs hard-CE to one action); value loss -> MSE
to `value` (the search score) instead of `returns`. gen_flat not yet pivoted.
Oracle is still b2b_search (per user "don't use as teacher" — revisit using a
checkpoint-guided search as the scoring oracle).

## Pre-existing deferred work

- I eventually want to move the AR runner from the environment to this repo.

## Placement scorer (new 3rd family) — in progress (2026-05-31)

New `placement` family: a fusion-style candidate-ranking policy. Scores up to 128
candidate placements (64 no-hold slots 0:64, 64 hold slots 64:128, packed by
score, illegal masked). Each candidate is an 18-dim placement vector
(onehot piece[7] C-order I,J,L,O,S,T,Z via `piece_value-1` + onehot rot[4] +
col/9 + landing_row/(board_height-1) + onehot spin[4] + hold flag). Distills the
search's full candidate dist (soft-CE to `softmax(cand_scores/temp)`) + reuses
the state-only `ValueModel` (target `max_legal_score/value_scale`). Head =
cross-attention: per-candidate length-1 query over `piece_dec (B,10,64)`,
candidates independent (reuses `DecoderLayer`).

- **C change (done):** `b2b_search_c` now takes `out_root_landing_rows`;
  `search_with_scores` returns a 6th array `cand_landing_rows`. Callers
  (`gen_ar`, `dagger`, `b2b_demo`) updated to unpack/ignore it. `.so` rebuilt
  in place (`python setup.py build_ext --inplace`). Smoke: 24-row board →
  landing_row in [0,23], ~34/34 candidate split.
- **ROW_NORM = board_height-1** (24-row board → 23). Normalize in datagen by the
  same board passed to the search.
- flat + ar + their datagen/pretrainers/dagger schema left behaviorally
  untouched (only the search-return unpack adapted — unavoidable ripple).

- **Demo (done):** `demo placement --checkpoint ...` ([demo/placement.py]). Per
  step it enumerates candidates from a wide/shallow `search_with_scores`
  (depth 2, beam 512), packs them via `build_placement_inference`
  (placements + bool mask + aligned key sequences), the model ranks, and the
  chosen slot's sequence executes. `placement_features` refactored to share the
  packing order (`_branch_order`/`_decode_vec`) between datagen + inference.

- **DAgger (done):** `datagen placement --dagger --policy-checkpoint ...`
  ([dagger.py] `collect_dagger_placement`). The per-step search serves both roles:
  its candidates feed the policy's ranking (which drives the env) and its scores
  form the label (`build_placement_target`). Output schema is identical to
  `gen_placement`, so BC + DAgger rounds accumulate into one dataset. main()
  branches collect/schema-check/save on `is_placement` via a `label_key`.
- **Policy+value merge (done, 2026-05-31):** `PlacementPolicyModel` →
  `PlacementPolicyValueNet` — one shared `process_obs` trunk, two heads
  (candidate-ranking policy + state-only value `value_trunk`/`value_top`).
  `ValueModel` dropped from the placement family (still used by ar/flat). Encoder
  + policy-head submodule names kept identical so old policy-only checkpoints
  warm-start the trunk+policy via `restore(...).expect_partial()` (verified:
  in-dist Top1 0.83 after warm-start, value head fresh). Pretrainer now trains one
  model/optimizer with `policy_loss + value_weight*value_loss` (single dir
  `placement_pretrained_policy`; old `placement_pretrained_value` orphaned). This
  is fusion-aligned: one encoder pass → policy+value per node for the future
  search-at-inference.

- **Neural-guided search (built, 2026-05-31):** `src/qtris/search/placement_search.py`
  — fusion-style beam: policy gates top-K candidates + adds a prior bonus, state-only
  value evaluates resulting boards, beam-prune, back leaf scores up to the root move.
  Purely inference (no new training). Simulator: `clone_sim_env` deep-copies the env
  while SHARING its two stateless ctypes handles (`_key_sequence_finder`/`_hole_finder`,
  which deepcopy can't pickle), pathfinding+garbage off, then real `_step` — verified
  bit-identical to `env._step` over 12 steps. `demo placement --search [--depth --beam
  --gate]` wired ([cli/demo.py], [demo/placement.py]). depth-1 ≈ value-rerank (value
  head needs no candidates: `score_value`). Timing ~150ms/move depth-1, ~0.5s depth-2.
  **Finding:** with the current value head it does NOT beat greedy — depth-1 is
  redundant with the policy (`value(child)≈cand_score`), depth-2+ leans on OOD value
  (optimistic max-over-subtree back-up picks bad roots). The value is trained on oracle
  states; it's unreliable on the search's own boards. **Unlock = DAgger on
  search/rollout states** (relabel with the oracle) so the value is accurate where the
  search explores — then deeper search pays off. NOTE: ckpt-1830 is policy-only (value
  head random); a merged retrain (joint policy+value) is needed before the search is
  even meaningfully testable.

- **Open-loop collapse diagnosis (2026-06-01): NO BUG.** Reconstructed 120 training
  states into a sim env and ranked them through the full rollout pipeline
  (`net_input_from_env`+live `enumerate_node`+argmax): Top1 **0.933, identical** to the
  in-dist stored-candidate Top1; net-input boards + candidate sets match stored 120/120.
  Stratified Top1 high on openings (0.928)/low-b2b (0.956), lowest on high-b2b (0.569).
  ⇒ model/encoding/enumeration/simulator/pipeline all correct. Root cause = open-loop
  distribution shift on the oracle's narrow, fragile b2b-hoarding manifold: 87–93% per-move
  on-manifold, but one off-move breaks the chain and there's zero recovery data → competent
  ~1 bag then collapse. BC can't fix it; fix = learn from the model's own states (PPO/DAgger).
- **Single-player PPO (built, 2026-06-01):** mirrors flat. Env: `PyTetrisEnv(placement_candidates=True)`
  exposes per-step dense-320 candidates in the obs (`cand_scores`/`cand_landing_rows`/`cand_sequences`,
  via the C search inside the env, parallelized across subprocesses) — applied directly in
  QTris's `tetrisenv/` (subtree too diverged from TFTetrisEnv main for a clean pull: base
  commit gone, ~900-line search rewrite upstream; per user, decoupled). Runner
  [runners/placement.py](src/qtris/runners/placement.py) (mirror FlatRunner, single merged
  net, candidates from obs via `build_placement_inference`); trainer
  [training/placement.py](src/qtris/training/placement.py) (GAE+clipped PPO+value, one net/
  optimizer/tape, KL early-stop, return-scale EMA, warm-start BC → save `checkpoints/placement_rl`);
  `train placement` wired. Integration smoke (2 envs) passes end-to-end. **Honest:** PPO may
  not re-discover elite hoarding from the fragile BC init; DAgger is the lower-risk first try.
- **Candidate source fix (2026-06-01): env candidates now come from pathfinding, not b2b_search.**
  Root cause of the recurring `is_spin` crash in PPO training: the env's
  `_enumerate_placement_candidates` ran `CB2BSearch.search_with_scores` (a *search*), which
  **death-prunes legal placements** at the root (`placement_is_dead`, mh≥board_height−4=20).
  At a non-terminal near-death board (env death line = row 24−max_height) all roots could be
  pruned → 0 candidates → runner samples an all-PAD slot → `env._play_action` leaves `is_spin`
  unbound. b2b_search's *scores* are only needed by datagen; the runner ranks with the net +
  GAE, so it needs only the legal set. **Fix:** new `find_placement_candidates_c` in
  [pathfinder.c](tetrisenv/TetrisEnv/pathfinder.c) — the same BFS that fills `obs["sequences"]`
  (AR), bucketed into datagen's dense spin4 layout (`rot*40+norm_col*4+spin_type`, deepest
  landing row per slot), **no death-pruning** (death = env termination, not a filter). Ported
  `detect_t_spin` (4-value, verbatim from b2b_search) + `compute_spin_type`; extracted
  `bfs_expand` shared by both enumerators; `is_t_spin` removed (binary-equivalent, proven).
  `CKeySequenceFinder.find_placement_candidates` binding added; env drops `CB2BSearch`.
  **Verified:** AR `find_all` unchanged; pathfinder vs b2b candidate sets **byte-identical**
  across 400 boards / 13,770 slots (0 landing-row mismatches, 0 asymmetric, spin types match)
  → warm-start encoding preserved; invariant **non-terminal ⇒ ≥1 candidate** holds over 12,800
  steps / 773 deaths at max_height=18 (the 9 zero-candidate states are all terminal → tf_agents
  auto-resets, no crash). PPO smoke + runner stress pass.

- **Oracle search aligned with fusion (2026-06-02): un-prune + futility + scale.** The
  label-generating oracle (`b2b_search.c`) pruned harder than fusion, narrowing the candidate
  label manifold. Three changes (all in `b2b_search.c`): (1) **removed the hard death-prune at
  root enumeration** (all 3 branches) — near-death placements are now emitted as very-low-scored
  candidates (never silently removed); kept the child-expansion death-prune for efficiency.
  (2) **seed `root_best[i] = s->score`** at enumeration + removed the `root_best <= -1e29`
  survival gate → **every legal root is emitted** (no more silently dropped legal moves; this was
  the `is_spin` crash mechanism). (3) **futility pruning** in `finalize_beam` (fusion-style: drop
  beam nodes `> FUTILITY_DELTA` below the tier-best, keep ≥1). (4) **global rescale** `SCORE_SCALE
  = 0.05` on `evaluate_state` (+ death return + `evaluate_state_decompose`) → final scores now
  fusion-magnitude **O(tens)** (verified: empty [-1.6,7.5], mid-game [-20,1]; near-death roots
  ~−275..−767, ~0 softmax prob). `FUTILITY_DELTA = 15` (fusion's value, in rescaled units).
  Monotonic rescale **preserves b2b-hoarding** (NOT re-tuned to fusion's balanced weights).
  **Verified:** near-death board → ≥1 candidate incl. the near-death move; healthy boards emit all
  ~68 legal roots; **best move identical** with futility off vs on (A/B on 5 boards). Value target
  is scale-invariant (`max/value_scale`). **policy_temp → 1.0** (fusion) in `pretraining/placement.py`,
  `cli/pretrain.py` default, `training/placement.py EXPERT_TEMP` (already 1.0) — sharper, fusion-aligned
  distillation target. **Global impact:** `SCORE_SCALE`/futility live in the shared search, so
  `ar`/`flat` datagen scores are rescaled too → their datasets are stale and the shared `--policy-temp`
  default=1.0 now applies to them as well (correct once regenerated). **MUST regenerate the placement
  dataset + re-pretrain** (`tetris_oracle_placement` cand_scores changed; checkpoints stale). RL/env
  runtime unaffected (pathfinder candidates use 0.0 validity sentinels, not oracle scores).
- **Futility regressed the oracle; disabled it (2026-06-02).** With futility=15 the b2b_demo
  (depth16/width200) died ~300 steps at ~0.8 APP vs the prior >1.0 / near-no-death. Cause: QTris's
  eval is **peaky** (a kept-b2b line outscores a cash-out/downstack line by ~`b2b` points;
  alive-root spread ~26 scaled by b2b=30), so tier-relative `delta=15` cuts the **strategic
  survival/surge lines** mid-beam → greedy hoard into top-out. Fusion tolerates futility only
  because its eval is smooth (board-shape+attack) **and** paired with quiescence; quiescence can't
  rescue lines futility already pruned mid-loop. **Fix:** `FUTILITY_DELTA = 0` (disabled — redundant
  for QTris: `placement_is_dead` + truncation already drop bad lines; only harmful at useful deltas).
  Switched the un-prune to **emit-but-don't-expand** (near-death roots added to `depth0_placements` +
  `root_best` seed, but `continue` before `next_beam_size++`) → search beam **provably identical** to
  the strong death-pruned version (no play regression), labels still include near-death candidates.
  Rescale verified monotonic (all score math inside `evaluate_state`; only relative compares outside).
  **Open:** quiescence depth-extension (the genuine fusion value-add) not implemented — would aim to
  *exceed* the restored baseline; needs eval to validate.
- **REVERTED the height-fade + futility (2026-06-02, user call).** The height-fade reached nowhere
  near the original b2b (salient even at ~100 steps) → much lower APP; futility hurt more than
  helped. Removed both (the `finalize_beam` futility block, `FUTILITY_DELTA`, `B2B_HEIGHT_FADE` —
  statics, table entries, and the faded `W_B2B_LINEAR` term). Oracle PLAY is back to the original
  hoarding (W_B2B_LINEAR=20, no fade, no futility). **Kept (non-play):** `SCORE_SCALE=0.05`
  (monotonic — only rescales evaluate_state magnitude for the pretraining target; no ranking
  effect), Option-B emit-all-roots (near-death candidates emitted but not expanded → search beam
  identical to the original; widens the dataset label set), and the run_eval_games **banked-surge
  metric fix** (credits held b2b on a survived game). The fade-saga entry below is historical
  (that approach was abandoned).
- **Dropped SCORE_SCALE too (2026-06-02, user call).** With futility gone, SCORE_SCALE had no
  consumer that reads absolute magnitude — the policy target `softmax(scores/temp)` absorbs any
  global scale into `policy_temp`, and the value target `max/value_scale` is std-standardized
  (scale cancels). So it was vestigial + caused a flatter-than-original distillation. Removed it
  entirely (statics, table entry, the two `evaluate_state` multiplies, the decompose scaling) →
  scores back to **raw magnitude** (O(hundreds)). Restored `policy_temp=10` (pretraining + CLI
  default + `EXPERT_TEMP`) so the distillation target is the original `softmax(raw/10)`. Net: the
  oracle + pretraining pipeline are back to the **original** behavior; the only retained deltas are
  Option-B emit-all-roots (wider labels, play-neutral) and the run_eval_games banked-surge metric fix.
- ~~**Resolution — height-faded hoard reward makes hoarding + survival + futility coexist (2026-06-02).**~~ (ABANDONED — see above)
  Two findings drove this: (1) **the eval metric was undercounting hoarding** — the C self-play
  (`b2b_run_eval_games`) never credited the banked surge of a b2b chain still held at the step cutoff.
  Fixed: on a *survived* game, `total_attack += b2b` if `b2b>=4`. With the fix, hoarding's true APP
  jumped 0.738→0.879 (vs structure-driven 0.736) — hoarding IS the stronger APP strategy (the
  "high b2b doesn't help APP" conclusion was a measurement artifact). The **same undercounting likely
  exists in the RL env reward** (`Scorer` per-step attack, no episode-end banked-b2b credit) → would
  bias the placement model against hoarding; worth fixing there too. (2) **Height-faded hold reward**:
  `W_B2B_LINEAR·(1 − B2B_HEIGHT_FADE·h_ratio)` (new tunable `B2B_HEIGHT_FADE`, in the weight table).
  Full hoard incentive when safe, fading as the stack rises → the bot cashes out before topping out,
  AND the hold-vs-break gap shrinks near death (where futility was cutting cash-outs). Sweep (4 games,
  0.2 garbage): hoard-base 0.862 APP / **2/4** survived → **fade=1+futility=15: 0.808 APP / 4/4
  survived**, b2b still reaches ~35 when safe. So futility is now **viable with no regression**
  (fusion-style match) while hoarding + survival both hold. **Committed defaults:** `W_B2B_LINEAR=20`,
  `B2B_HEIGHT_FADE=1`, `FUTILITY_DELTA=15`, `SCORE_SCALE=0.05` (all weight-table tunable). `chaining
  ruleset NOT used` (user: not the fusion ground-truth ruleset; oracle stays flat-+1, matching the env
  `Scorer`). Slight APP dip (0.862→0.808) is over full games (hoard-base's 0.862 included 2 deaths);
  survival-weighted the fade is stronger, and at the demo's 0.15 garbage it should hoard more (higher APP).
  Tune `B2B_HEIGHT_FADE` down toward 0 to hoard harder (more APP, more death risk). **Still required:**
  regenerate the placement dataset + re-pretrain (oracle scores changed); optionally port the banked-b2b
  credit + height-fade-equivalent into the RL env reward.

### Deferred (placement)
- **Run real PPO training** (`uv run train placement`, 64 envs) + measure open-loop survival
  vs BC on the seed harness; tune entropy/KL/reward if it collapses to survive-flat.
- **DAgger on search/rollout states** — the value-quality unlock for the search; lower-risk
  than PPO; `collect_dagger_placement` already built.
- Refactor the duplicated encoder-construction block into `QtrisModelBase.__init__`.

## Placement DAgger "agreement stuck at ~80%" — diagnosed (2026-06-01)

Symptom: `datagen→pretrain→dagger→pretrain→…` never moves `policy≠beam` off 80–83%
despite train Top1 0.88–0.92 / Top3 >0.95. Suspected labeling/normalization/seed/
checkpoint. Diagnosed with two cheap checks at the prod budget (d16/w200); scratch
scripts in `/tmp/check_oracle_gap.py` + `/tmp/check_dagger_decomp.py`.

**Root cause = genuine distribution-shift collapse, NOT a bug.** Decomposition on the
policy's own rollout (298 states, ckpt-2060, vars verified restored):
- (a) policy==best_seq = 0.168 (reproduces the 80–83% stat)
- (b) policy==argmax(cand_scores) slot = 0.171 (agreement with what it's TRAINED on)
- (c) argmax(cand_scores)==best_seq = 0.970 (oracle gap ~0 on policy states)
- by phase: opening ~0.9, collapses within ~1–2 moves (later≥7 = 0.150).
Model matches its target ~90% on oracle states but ~17% on its own states: one
off-move on the narrow b2b-hoard manifold jumps it far OOD; 0.9^k compounding +
manifold narrowness = opening-then-crater. The stat is brutal but faithful (since
(c)≈1 it ≈ (b)).

**Ruled out by measurement:** input norm (shared `encode_placement_features`/
`build_placement_*`, bit-identical; opening 0.9 proves inputs match), schema/seed/
max_len, checkpoint load (`assert_existing_objects_matched` passes). The oracle
mismatch I first hypothesized (`cand_scores`=`root_best` cross-depth/pre-prune/
speculative max vs `best_seq`=final-beam root) is REAL but minor: ~26% on oracle
states (median score-gap 0.0 → mostly ties), ~3% on policy states. Not the driver.

**Why DAgger doesn't help (actionable):**
1. Dilution — BC+DAgger share one dataset (`datasets/tetris_oracle_placement`, 63,382
   rows, median b2b −1 / fill 0.25 = oracle-dominated). "Standard data collection"
   re-floods oracle states each loop → on-policy states stay a minority under uniform
   shuffle. FIX: separate/oversample/cap DAgger vs BC; re-measure (b) on a fresh
   rollout (track (b)+top3 in dagger.py as the honest signal).
2. If (b) still won't rise after DAgger-dominant retrains → fragile target not
   BC/DAgger-realizable → use the built single-player PPO (`train placement`).
Independent: value target `max(cand_scores)` is an optimistic max → for the search,
use final-beam backed-up value of `best_action` (notes.md "Neural-guided search").

### Architecture fix: candidates attend to the board (2026-06-01)
Followed up the representation finding. Was: `score_candidates` had each candidate
cross-attend ONLY to `piece_dec` (~10 tokens: 7 piece + 3 bcg); the 60 board patches
from `make_patches` enriched `piece_dec` then were discarded → candidates never saw
board geometry (fusion gives every candidate the full board embedding by concat). This
matched the symptom: opening ~0.9, mid-game collapse, and weak in-distribution high-b2b
(0.569).
Fix (all in [model.py](src/qtris/models/placement/model.py), reused existing
primitives): overrode `process_obs` to ALSO return `board_dec`; `call`/`predict` build
`context = concat([board_dec, piece_dec])` (~73 tokens) and the candidate cross-attn now
attends to `context`. Value head still reads `piece_dec` (board-aware global summary;
keeps params identical). bcg appears in both streams (harmless 2 views).
Verified: trainable params **1,615,362 unchanged** (attention is length-independent;
`PosEncoding` non-trainable), warm-start `assert_existing_objects_matched()` passes from
`placement_pretrained_policy` (exact restore — resume, no scratch run), forward/predict
shapes + masking correct, ruff clean. VRAM +~1MB×batch activations only. NOTE: a live
pretrain run was using OLD arch (ckpt advanced 2060→2335 during the session) — restart
pretrain to pick up the new arch. Success metric to watch: in-distribution high-b2b
Top1 rises off 0.569; `/tmp/check_dagger_decomp.py` (b) rises off ~0.17. Deferred: give
the value head board context too (changes param count); fold the duplicated `process_obs`
into `QtrisModelBase`.

## tetrisenv subtree re-synced with TFTetrisEnv (2026-06-01)

The subtree was decoupled (changes applied directly in `tetrisenv/`); now re-synced via
upstream. The env-side changes were ported INTO TFTetrisEnv on a NEW branch
`experimental/placement-candidates` (based on `main` @ `79ece6d`), then the subtree was
freshly re-added from it. `tetrisenv/` is now byte-for-byte identical to that branch.

- **Ported up** (6 files, upstream was untouched at these since base `ebb5b2a`):
  `TetrisEnv/{b2b_search.c, pathfinder.c, CKeySequences.py, PyTetrisEnv.py, CB2BSearch.py}`
  + `b2b_demo.py`. TFTetrisEnv commits: `0524efd` (port) + `bc13115` (remove cmaes).
- **Upstream cleanups the subtree ADOPTED on re-pull** (it was behind base+3):
  `__init__.py` → `KeySequenceFinder` (was `BitboardKeySequenceFinder`),
  `KeySequencesBitboard.py` deleted, `.gitignore` adds `*.so`. Verified safe — nothing
  in `src/`/`tetrio/` imports `BitboardKeySequenceFinder`.
- **`cmaes_optimize.py` removed both sides** (was already gone from the subtree).
- **MCTS WIP left untracked upstream** (`b2b_mcts.c`, `b2b_common.c/h`, `CB2BMCTS.py`,
  not in `setup.py`) — did NOT flow into the subtree.
- The old `.so` gotcha is RESOLVED: upstream now gitignores `*.so`, so future pulls
  won't re-introduce committed binaries.

Re-add mechanics: the env branch was pushed to `origin/experimental/placement-candidates`,
then the subtree was re-added from the REMOTE: `git rm -r tetrisenv` → `rm -rf tetrisenv`
→ `git fetch tetris-env experimental/placement-candidates` → `git subtree add
--prefix=tetrisenv tetris-env experimental/placement-candidates --squash`. QTris squash
commit `e47c117` (merge `ed4f65b`) records `bc13115`; one merge only (the earlier
local-fetch re-add was reset away to avoid a duplicate merge). `uv sync` rebuilds the
editable extension.

Verification: TFTetrisEnv + QTris `b2b_test.py` both 5/5 survived / b2b 38.6 (identical);
imports OK; placement-candidate env obs (dense-320, 68–70 legal/step, non-terminal⇒≥1)
smoke OK; subtree byte-for-byte == `tetris-env/experimental/placement-candidates`.

### Future pulls
Subtree now tracks `tetris-env/experimental/placement-candidates` (pushed). Pull upstream
changes with `git subtree pull --prefix=tetrisenv tetris-env experimental/placement-candidates --squash`.

## Placement PPO wandb observability wired (2026-06-01)

Wired the placement single-player PPO pipeline to `qtris.observability` (reusing the AR
trainer's pattern, not flat's — AR is the most recently used). Reuses `SingleAgentTrainConfig`
+ `SingleAgentPPOLog` + `init_run`/`log_step`/`finish` (no new schema needed).

- **`runners/placement.py`**: trajectory buffer now also collects the reward channels from
  `time_step.reward` (`attacks`, `clears`, `attack_reward`, `total_reward`, `garbage_pushed`)
  so the same gameplay/reward metrics AR logs are available. (Previously only `total_reward`.)
- **`training/placement.py` `train_step`**: now returns an AR-shaped dict
  (`ppo_loss/entropy/approx_kl/clipped_frac/value_loss/explained_var/board/scores/expert_loss/
  expert_accuracy`) instead of a 5-tuple. Calls the merged net with `return_scores=True` to get
  `piece_scores` (the attention heatmap image). `expert_accuracy` = Top1 agreement between the
  policy's argmax candidate and the oracle's argmax `cand_scores` (the soft BC target has no
  hard label). clipped_frac/explained_var computed as in AR.
- **`training/placement.py` `main`**: builds `SingleAgentTrainConfig`, `init_run(project="Tetris")`,
  aggregates the buffer metrics each gen (mirrors ar.py), `log_step(SingleAgentPPOLog(...))` every
  `gen % 4`, `finish` + `runner.env.close()` at the end. Logging cadence kept at the file's existing
  gen%4 (print) / gen%5 (save).

### GOTCHA: pydantic image fields reject tf tensors (affects ar/flat/1v1 too)
`PPOLogBase.board`/`scores` are typed `np.ndarray` (pydantic v1, arbitrary_types_allowed), which
**rejects `tf.Tensor`** (`type_error.arbitrary_type`). placement converts at the log boundary
(`.numpy()` on board + norm_c_scores). **ar.py/flat.py/_1v1.py still pass raw tf tensors to these
fields** → they would raise `ValidationError` on the first `log_step`. Latent bug, not yet hit/fixed.
Fix options when picking up: add `.numpy()` at each callsite, OR a central `@validator(pre=True)` in
`observability/models.py` coercing image fields via `np.asarray` (fixes all trainers at once).

Verified: 2-env smoke (WANDB_MODE=disabled) — runner channels correct shapes, train_step dict,
scores→(12,5,1), full `SingleAgentPPOLog` + `log_step`/`finish` path green. ruff clean.

## b2b_search datagen speed — TT enlargement is a DEAD END (2026-06-01)

Datagen throughput is ~3.4–3.6 search/s at d16/w200 (≈ the user's 4 steps/s). Profiled the
oracle search (`tetrisenv/TetrisEnv/b2b_search.c`). The leaf eval (`evaluate_state` →
`compute_board_stats`) is the hot path: ~700k evals/move.

**Enlarging the transposition cache does NOT help** (tested, reverted). `TT_SIZE` 2^16→2^21
(64k→2M entries) left the hit rate essentially flat: **54.3% → 54.9%**, timing within noise
(3.59 vs 3.49 search/s). Behavior-preserving (40/40 byte-identical search outputs on a fixed-seed
40-move replay — the TT key encodes all score-relevant state). Root cause: the ~45% misses are
**first-time states**, not evictions — the 64k table was never thrashing, so a bigger cache has
nothing extra to reuse. Caching is already tapped out at ~54%.

Real levers for datagen speed (per-eval cost, since the eval count is fixed unless depth/beam or
pruning change — which the user ruled out): **(a) `-march=native -funroll-loops`** in setup.py for
b2b_search (enables popcnt + SIMD; no FP-semantics change; NOT `-ffast-math`); **(b)
`__builtin_popcount`** at the 3 manual popcount loops (b2b_search.c ~2015/2064/2102; one already
uses it at ~1830); **(c)** micro-opt `compute_board_stats`/`tt_hash` (every eval pays `tt_hash`,
even the 54% hits). These target all 700k evals' hash + the misses' compute. Not yet benchmarked.
Bench harness pattern that worked: record fixed-seed inputs + sha1 digests of
(best_action,best_seq,cand_actions,cand_scores,cand_rows) with the old .so, then replay the same
inputs through the new .so and compare digests (proves zero behavior change) + time only the search
calls. Force the in-place rebuild with `cd tetrisenv && uv run python setup.py build_ext --inplace`
— `uv sync` does NOT recompile a changed .c in the editable install.

### Per-eval micro-opts: popcount KEPT (+~5%), -march=native REJECTED (2026-06-01)
Multi-thread timing is too noisy (~15%: OMP scheduling + atomic contention + Zen5 boost) to resolve
a micro-opt — measure SINGLE-THREAD (OMP_NUM_THREADS=1) for clean per-eval code-speed signal.
Single-thread best-of-N on a fixed 10-state replay (parity 0 mismatches throughout):
- pristine:                       0.691 search/s
- `__builtin_popcount` (3 loops): 0.726 search/s  (**+5%**, KEPT)
- popcount + `-march=native -funroll-loops`: 0.687 search/s (march/funroll **ERASED** the popcount
  win → REJECTED; setup.py left unchanged). CPU = AMD Ryzen 7 9800X3D (Zen5, has AVX-512 but no
  Intel-style downclock; -march still didn't help this branchy board code, and -funroll bloats).
KEPT change: `b2b_search.c` ~2015/2064/2102 `while(v){bits++;v&=v-1;}` → `__builtin_popcount(row)`
(matches the existing idiom at ~1830). Byte-identical output (single-thread parity 0/40 every run;
popcount can't change the bit count). Modest (~5% per-eval ≈ ~1h off a 24h run) but free + safe.
- Aside: search is effectively MT-deterministic (same binary/inputs, 0 diffs over 200 MT
  comparisons); one stray 1/80 mismatch vs an independently-recorded golden = an extremely rare
  residual TT/beam race, NOT the popcount change (which is single-thread byte-identical).
Net for datagen speed across #1/#2/#3: only popcount helps, and only ~5%. Real throughput needs
either cheaper per-eval `compute_board_stats` (bigger effort) or fewer evals (depth/beam/pruning —
ruled out). The 24h is fundamentally eval-count × eval-cost bound.

### Profile: where the search time actually goes (2026-06-01, py-spy --native, single-thread)
Profiled d16/w200 search with py-spy `--native` (15.6k samples; the .so ships `-g`, so DWARF
unwinding works — but `-O3` inlining corrupts py-spy's INCLUSIVE numbers, so trust SELF-time only).
SELF-time by function:
- **b2b_check_collision  26.2%** — the inner collision kernel (already bitmask-tight: precomputed
  piece row-masks `board_row & (mask<<c)`, 4-row loop; its cost is pure CALL VOLUME, not waste).
- count_immobile_placements 14.7% · compute_reachability 11.9% · find_placements 7.1% ·
  compute_board_stats inline scans 7.1% · cell_filled 3.2% · state_hash_rows 2.6% ·
  count_deep_holes 2.5% · compute_hole_ceiling_weight 2.4%.
- The actual heuristic SCORING (`evaluate_state` body) is only ~2%.
**Conclusion: ~60% of search time is board GEOMETRY** (collision/reachability/placement-enumeration),
funneling through `b2b_check_collision`; the eval's scoring math is negligible. So the lever is
reducing the NUMBER of geometry/collision checks in the three callers (immobile/reachability/
movegen), NOT optimizing the leaf score or the kernel. NOTE: `b2b_hard_drop_row` linear-scans rows
via collision checks — a col_heights-based landing would be O(4) but is NOT exactly equivalent under
overhangs (breaks tuck/spin placements), so it would change output — rejected. Any geometry speedup
must preserve overhang-aware results exactly (parity-gate it). Also: `__builtin_popcount` currently
lowers to a libgcc `__popcountdi2` CALL (1.1% self), not the `popcnt` instruction — `-mpopcnt` alone
(not full `-march=native`) would emit the instruction; marginal.

### count_immobile_placements: removable for ~24% speed, but LOAD-BEARING — keep it (2026-06-01)
A/B at d16/w200 via `b2b_run_eval_games` (8 garbage seeds / 6 no-garbage, 100 steps), full heuristic
vs `count_immobile_placements` no-op'd (source early-return; zeros its result + masks → all immobile
terms vanish, wasted-hole loses its spin-setup exemption):
- garbage 0.15: b2b **28.6→16.2 (−43%)**, APP 0.748→0.661 (−12%), combo 0.31→0.37, surv 100% both;
  **298→228 ms/move (~24% faster)**.
- no-garbage:   b2b 33.3→28.3 (−15%), APP 0.515→0.508, surv 100%; 296→235 ms/move (~20% faster).
Verdict: it IS the biggest single removable cost (~24% of runtime — its 15% self + a big share of the
26% b2b_check_collision via the 4 immobility collision checks per slot), BUT removing it craters b2b
maintenance (the exact thing it rewards), violating the b2b/APP priority. NOT removable. Reusability
inside it is marginal (already reuses `reachable[]`, bounds scan to a ~8-row surface window,
early-exits FITS/REACHABLE before the 4 collision checks); only micro-reuse left = precompute the
shifted piece masks once per (rot,c) vs re-shifting in each of the 4 immobile collision checks.
Reverted the no-op (confirmed b2b restored). Net: no safe big win in the search; only the ~5% popcount.

## Standalone fusion_search C engine (2026-06-02)

Added a SECOND search oracle alongside b2b_search, faithfully porting fusion's heuristic beam
search (`fusion/src/{eval,analysis,search,search_expand,search_config}.rs`) into
`tetrisenv/TetrisEnv/fusion_search.c` + a `CFusionSearch` ctypes wrapper. Same API surface as
`CB2BSearch` (drop-in for `search_best_move`), so it slots into the placement neural-guided search.

What it is: a board-SHAPE composite oracle (the opposite philosophy to b2b's hoarding). Node score =
`board*W_BOARD + (path_attack/df)*W_ATTACK + (path_chain/df)*W_CHAIN + context*W_CONTEXT`, where
`df = min(sqrt(pieces_placed), MAX_DEPTH_FACTOR=2.45)`, `path_chain` accumulates `shape_chain(combo) =
1-exp(-0.25c)` clamped, board eval = fusion's holes/coveredness/height-tiers/bumpiness/row-transitions/
well/tsd-overhang/four-wide (port of `eval.rs::evaluate`). Defaults W_BOARD=1, W_ATTACK=0.5,
W_CHAIN=0.15, W_CONTEXT=0.10. Futility (drop nodes > FUTILITY_DELTA=15 below the best survivor) lives
in `finalize_beam`; quiescence (extend "loud" = combo>0||b2b>0||cleared nodes up to QUIESCENCE_MAX=3
extra depths, fold resolved-quiet ones back into the beam) runs after the main depth loop.

How it was built: `cp b2b_search.c`, renamed the 9 exported `b2b_*`→`fusion_*`, then REPLACED the
eval (`evaluate_state` body → fusion board eval + composite) and weight table, ADDED
`path_chain`/`fusion_context` SearchState fields (accumulated in `expand_and_insert` + the 3 root
branches), futility, an `expand_one_parent` helper (shared by the main loop + quiescence), and the
quiescence block. DELETED the now-dead b2b eval cluster (compute_board_stats + reachability + immobile
+ decompose), 3931→2464 lines. Reuses verbatim: all board geometry, find_placements/BFS,
compute_attack (FLAT b2b attack — matches the env Scorer + fusion's no-chaining ground truth),
garbage sim, the OpenMP beam loop, the C game loop (`fusion_run_eval_games`). The flat-attack +
omitting fusion's NN-policy-guided expansion + transposition table (use_tt=false) was the agreed scope.

Verified: builds clean; `weight_names()` lists fusion's 18 weights; `search_with_scores` returns the
full legal root set with finite scores + valid landing rows + HARD_DROP-terminated best sequence;
diverges from b2b on 7/12 random boards (same root COUNT, different judgment); `run_eval_games` 3×200
under garbage all survive, avg height ~6.8, max_b2b ~2 (clean low-stack downstacker, NOT a b2b hoarder
— expected for a board-shape eval). Demos: `b2b_demo.py --engine {b2b,fusion}`; placement
`demo --search {b2b,fusion}` (was `store_true`) selects which C oracle enumerates the candidate set
the net ranks (greedy/no --search defaults to b2b).

### Follow-ups / deviations (actionable)
- **Height-tier thresholds kept verbatim at 10/15** (fusion's `eval.rs` values for a ~20-row
  playfield) on QTris's 24-row board. The plan flagged scaling them to 24 rows as a TUNING follow-up,
  not part of the faithful port — survival is independently enforced by `placement_is_dead` (death
  zone = top 4 rows + garbage), so the un-scaled tiers only shape height preference, they don't risk
  topping out. Revisit if fusion plays too tall/short.
- **`context` implemented faithfully as `clamp(combo - prev_combo, -1, 1)`** (×W_CONTEXT=0.10), a
  deliberate refinement of the plan's "context=0 (no coaching state)": the combo-delta is intrinsic,
  not coaching state (fusion's `coaching_context_bias` IS zero here since we have no CoachingState),
  so this is the genuine offline-oracle value and keeps W_CONTEXT meaningful. Magnitude is tiny (±0.1);
  set W_CONTEXT=0 via `set_weight` to recover the strict plan behavior.
- **Eval cache keyed on a board-only FNV hash** (not the inherited `tt_hash`, which mixes path-state
  fields — wrong key for a pure board-shape eval and would gut the cross-call hit rate). `tt_hash` is
  now unused (`static inline`, no warning); could be deleted in a later tidy. The composite (attack/
  chain/context) is recomputed every node (path-dependent, not cached).
- Subtree note still applies: `tetrisenv/` diverged from upstream, so these edits live only here; it's
  ruff-`extend-exclude`d (CFusionSearch.py + b2b_demo.py are not linted).

## b2b_search eval simplification + attack-credit reframe (2026-06-02)

Stripped `evaluate_state` (`tetrisenv/TetrisEnv/b2b_search.c`) down to the terms that actually move the
score (per the 2026-05-29 decompose), and reframed the attack term. Goal was to explore attack/combo-driven
play with the b2b hoarding store removed.

**Changes (behavior-affecting):**
- `evaluate_state` now scores ONLY: survival (death floor, near-death cliff, height quartic, avg-height,
  bumpiness), holes + hole_ceiling, the b2b store (flat/sqrt/linear), the **new attack credit**
  `W_ATTACK_TOTAL·(total_attack + max(0, leaf_b2b))` (banked b2b counted as pending surge attack),
  b2b_attack, APP, garbage_prevent, T-slot, immobile_clear/lines. All `<1`-contribution terms removed from
  scoring: combo/combo_chain/b2b_combo, cascade, surge_pot, break_ready, max_single, downstack,
  covered_clear, well_aligned, well-shaping, hole_forgive, wasted_hole, tspin_multiline, spin_channel,
  future_b2b.
- **Fixed a pre-existing compile break**: an incomplete `SCORE_SCALE` revert left a dangling ref in the
  decompose mirror (the WIP b2b_search.c with the user's coefficient bumps did NOT compile). The user's
  earlier `W_ATTACK_TOTAL/W_B2B_ATTACK=15`, `W_B2B_COMBO=100` bumps had also been reverted to HEAD defaults
  (1.2/1.5/0) before this work — so they were never built/run.

**Changes (behavior-neutral cleanup):** decompose mirror rewritten to match (22→13 components; D_* renumbered;
`b2b_get_num_decompose` + `CB2BSearch.NUM_DECOMPOSE`/`COMPONENT_NAMES` synced — NUM_DECOMPOSE now READ from C).
16 dead `<1` weights removed from decls + `W_TABLE` (19 live knobs remain). Build green, no warnings;
play verified identical on fixed seeds (hoarding-off APP unchanged). The aspiration prune / `cheap_prescore`
no longer exists (removed earlier), so there was nothing to keep consistent there.

**Finding — hoarding-OFF + attack-credit is viable; attack weight is inert; combo is construction-gated.**
run_eval_games d16/w200, g0.15(1–4), 100 steps, 10 seeds, b2b store zeroed (linear+sqrt+flat=0), sweeping
`W_ATTACK_TOTAL`:
- 1.2 → surv 100%, maxb2b 7.6, APP 0.732, avg-combo 0.68, avgH 4.3
- 5   → surv 100%, maxb2b 6.5, APP 0.707, avg-combo 0.56, avgH 4.5
- 15  → surv 100%, maxb2b 7.9, APP 0.787, avg-combo 0.54, avgH 4.7
Removing the b2b store entirely keeps 100% survival + APP on par with the hoarding eval (~0.73), with a
CLEANER board and MORE comboing than hoarding (avg-combo 0.34–0.53 in the hoarding notes). But scaling the
attack credit 12.5× barely moves APP/b2b and actually REDUCES comboing (greedier 1-by-1 spin-clearing) — so
reward magnitude isn't the combo lever; comboing stays gated by the construction gradient (consistent with
the spin-channel findings above).

**Deferred (behavior-neutral, flagged):** 3 weights still gated inside `compute_board_stats` and kept
declared (`W_FUTURE_B2B`, `W_SPIN_CHANNEL`, `W_CHANNEL_HEIGHT_RELIEF`); the dead `BoardStats` fields +
their `compute_board_stats` scans + dead helpers (cascade/surge/downstack/well/wasted/spin_channel/
future_immobile/t_multiline) still computed-but-unread (pure per-leaf perf cost); dead `SearchState` fields
(`max_single_attack`, `max_b2b_combo`, `b2b_combo_len`) + their `tt_hash` mix still tracked. NOTE: the SOURCE
default still has hoarding ON (b2b store at 20/8/5) — the hoarding-off test was a runtime `set_weight`
override, not a source change. Deciding whether to bake hoarding-off in is open.

## b2b_search board-stats trim + b2b_demo weight-reload & score-breakdown (2026-06-02)

Finished the deferred dead-code trim and added two demo features.

**Part 1 — `compute_board_stats`/dead-field trim (`b2b_search.c`), behavior-neutral.** `BoardStats`
slimmed from ~30 fields to the 11 the eval reads (col_heights, max_height, avg_height, holes,
hole_ceiling_weight, immobile_clearing_placements, immobile_clearable_lines, t_spin_setups,
t_slot_quality, t_queue_count, bumpiness_exempted). Deleted the dead `compute_board_stats` scans
(near-horizon future_b2b, spin-channel + height relief, hole_columns, deep_holes, clearable/almost_full
rows, accessible/blocked/well_aligned/non_well 9-rows, cascade, wasted_holes; well loop trimmed to just
find `well_col` for the kept bumpiness exemption), the dead helpers `detect_spin_channel` +
`count_deep_holes` (+ SPIN_ROOF_MAX/SPIN_CHANNEL_CAP), the 3 still-gated weights (W_FUTURE_B2B,
W_SPIN_CHANNEL, W_CHANNEL_HEIGHT_RELIEF → **16 live weights**), and the dead `SearchState` fields
(max_single_attack, max_b2b_attack, max_b2b_combo, b2b_combo_len) incl. their tracking + the
`tt_hash` max_single mix. `b2b_search.c` 3931→3177 lines. Hole funcs now get an empty exemption mask
(identical to the prior W_SPIN_CHANNEL=0 behavior — `detect_spin_channel` already memset it to 0 before
its early-return). **Gate:** an OMP_NUM_THREADS=1 fingerprint (best_action + root-score hashes over 12
fixed positions, default weights) is byte-identical pre/post (md5 cb0bf94f20f9d01e) — provably zero play
change. `decompose` still returns `(n, 13)`; `CB2BSearch` unchanged.

**Part 2 — live weight config (`b2b_demo.py`).** Per-engine `{engine}_weights.json` (weight name-sets
differ b2b vs fusion). Startup seeds it from current weights if missing, else loads it (so edits persist
+ apply); both headless + GUI. GUI "Reload Wts" button re-reads the file via `set_weight`. Reuses the
already-bound `weight_names`/`get_weight`/`set_weight`. The search+decompose was factored into a
`refresh_search()` (evaluates the current board WITHOUT advancing the game); reload (and startup +
reset/new-game) call it so the breakdown reflects the new weights IMMEDIATELY — without this, reload
mutated the weights but the strip kept the last step's stale values until you stepped again (the
reported bug). NOTE: the config path is cwd-relative (`{engine}_weights.json`), so run the demo from the
dir where you edit the file.

**Part 3 — score-breakdown bottom strip (`b2b_demo.py`).** Replaced the cramped right-panel MOVE-DIST box
with a full-width strip below the board (window grew to fit). Shows the top-5 candidates by beam score
(headline = deep `search_with_scores` total, chosen move accented) with each one's **depth-0**
`decompose` component breakdown as a labeled, color-signed row (green +, red −). Components align to
candidates by enumeration index (both enumerate active-then-hold via `find_placements`), guarded by an
equal-length check. **fusion degrades gracefully** to totals-only (CFusionSearch has no `decompose`
binding). Verified: py_compile, headless config seed/reload roundtrip (applied 16), and SDL-dummy GUI
smokes for both b2b (populated strip) and fusion (totals-only) run + draw + quit without error.

Minor leftover (not addressed): the GUI "Dist Temp" spinner is now dead (it fed the old MOVE-DIST
softmax, removed); `--dist-temp` still drives the headless move-dist print. Source default still hoarding
ON. Decompose action-index output (to harden the index-alignment) is a possible future tweak.

## b2b_search offense de-dup + sublinear shape (2026-06-02)

Reshaped the offensive eval to remove duplicate crediting (each quantity credited exactly once) and
de-peak it. Trigger: to make attack beat the board terms the user had to crank `W_ATTACK_TOTAL` to ~100,
and a linear term × 100 spikes hard — because the SAME scalar was credited in multiple compounding terms.

Duplicate map (before): **b2b** was credited 4× — `W_B2B_SQRT·√b2b` + `W_B2B_LINEAR·b2b` (store), inside
`W_ATTACK_TOTAL·(total_attack + b2b)`, and inside `W_APP·(total_attack + b2b)/pieces` (+ a flat
`W_B2B_FLAT` flag). **total_attack** was credited 2× (attack term + APP), and `W_B2B_ATTACK·b2b_attack`
credited the b2b-maintaining *subset* of total_attack a 3rd time.

Now (two independent sublinear knobs, each quantity once):
- `W_ATTACK_TOTAL · √(total_attack)` — realized attack only (was linear `(total_attack + b2b)`).
- `W_B2B_SQRT · √(b2b)` — banked surge potential (the sole b2b credit).
- **Removed**: `W_B2B_FLAT`, `W_B2B_LINEAR`, `W_B2B_ATTACK`, `W_APP` (decls + W_TABLE) → **12 live weights**.
  The now-dead `SearchState.b2b_attack` field + its tracking (expand/roots/decompose-c) + its `tt_hash` mix
  removed too. Decompose mirror: 13→**10** components (dropped b2b_flat/b2b_linear/app, `D_B2B_SQRT`→`D_B2B`,
  `D_ATTACK`→√total_attack); `CB2BSearch.COMPONENT_NAMES` updated (NUM_DECOMPOSE auto-reads from C).
- KEPT (distinct scalars, not strict double-credits): height quartic + near-death cliff (same max_height
  but different regimes — smooth ramp vs survival floor), holes count vs hole-ceiling depth, immobile
  count vs lines. The user can revisit those if desired.

Verified: builds clean; weight_names=12; decompose (n,10); b2b=9 → 8·√9 = 24.00; attack col sublinear;
search returns a HARD_DROP-terminated best move; default-weights survival sanity 100% (low b2b/APP at
defaults is expected — offense is now lightly weighted by default).

**Re-tuning needed (the economy changed):** the user's `b2b_weights.json` still has the 4 removed weights
(reload silently skips unknown names) and `W_ATTACK_TOTAL` tuned for the OLD linear form. With the new √
shape, `W_ATTACK_TOTAL` should be re-tuned (likely different scale), and `W_B2B_SQRT` is now the ONLY b2b
lever (the linear store is gone). Delete `b2b_weights.json` to re-seed a clean 12-weight file, or edit the
live ones + Reload.

**Follow-up — encourage STARTING b2b.** The de-dup left the √b2b term guarded `b2b > 0`, so b2b==0 (the
first difficult clear from -1; `compute_attack` does `new_b2b = b2b + 1`) earned 0 — same as no b2b — so
nothing rewarded starting a chain. Changed to `if (b2b >= 0) W_B2B_SQRT·√(b2b + 1)` (eval + decompose
mirror): b2b=-1→0, b2b=0→W (the biggest marginal jump, +8 at default), then sublinear. So starting b2b
beats staying at -1; the search then prefers b2b lines over wasteful non-b2b clears. Default-weight sanity:
max_b2b 3.0→4.0, APP 0.42→0.55, survival 100%. (Reconciled a concurrent hand-edit of b2b_search.c that had
left two dead `s->b2b_attack` assignments in the hold/queue root branches — removed; the field is gone.)
Open: making "no clear" strictly beat a "useless (non-b2b) clear" is opportunity-cost, not per-state — a
non-b2b clear lowers the board (survival-useful), so an explicit penalty would fight garbage downstacking;
leaning on the b2b-start reward + lookahead instead.

**REVERTED the offense de-dup (2026-06-02).** The single √-term offense lost a lot of play strength —
the linear b2b store (`W_B2B_LINEAR·b2b`, the hoarding driver) + APP were carrying the b2b economy.
Default-weight sanity quantified it: de-duped max_b2b ~4 / APP 0.55 → restored **max_b2b 18.3 / APP 0.86**
(survival 100% both). So brought the offense back: b2b store `W_B2B_FLAT` (b2b>=0, also rewards starting) +
`W_B2B_SQRT·√b2b` + `W_B2B_LINEAR·b2b`; attack `W_ATTACK_TOTAL·(total_attack + max(0,b2b))`; `W_APP`. →
**15 weights, 13 decompose components** (b2b_flat/b2b_sqrt/b2b_linear/app restored; `CB2BSearch.COMPONENT_NAMES`
updated). LEFT OUT the minor `W_B2B_ATTACK` tiebreaker (and its `SearchState.b2b_attack` field stays
removed) — negligible play value, and re-adding the field is the fiddly part that broke the build earlier;
re-add on request. Net offense delta vs the original 16-weight version: just the missing b2b_attack
tiebreaker; the √-shape de-peak experiment is fully unwound. NOTE: restored `W_B2B_FLAT`/`W_B2B_LINEAR`/
`W_APP` use compiled defaults (5/20/10) unless present in `b2b_weights.json` — delete the file to re-seed
all 15, or re-add the tuned values.

**Baked b2b_weights.json into the C defaults (2026-06-02)** + removed all the saturation caps. Removed
`HOLES_CAP` (uncapped hole penalty), the `W_IMMOBILE_LINES` cap (`fminf(...,8)`), and the `W_TSLOT`
multi-setup multiplier cap (`fminf(usable_setups-1,2)`) — all in eval + decompose; kept the *semantic*
queue caps (`min(t_spin_setups, t_queue_count)`, count_immobile per-piece weighting). Then set the C
static initializers **and** W_TABLE default_values to the tuned `b2b_weights.json` values so datagen (which
builds `CB2BSearch()` with no override) uses them — verified the compiled defaults == the json (all 15).
Only two actually changed: `W_ATTACK_TOTAL` 1.2→1.0, `W_APP` 10→100 (the per-piece efficiency term — the
"100" attack knob; the rest already matched). Demo + datagen are now consistent at this snapshot; future
json edits still won't auto-propagate to datagen (re-bake, or wire `sync_weights_file` into the datagen
path). NOTE: this is a SNAPSHOT — the eval still differs from the pre-session "original" by the Stage-A
removals (hole_forgive, well-shaping, b2b_attack, combo/burst terms) flagged above as the likely "a bit
below original" contributors, none of which were restored.

**Added a b2b weight to fusion_search (2026-06-02).** Fusion's eval had no b2b term (board-shape +
attack/chain/context only — it tracked `b2b` but never rewarded it). Added `W_B2B` (fusion → 19 weights):
`+ W_B2B * max(0, leaf_b2b)` in `evaluate_state` (+ W_TABLE), default 0.5. It's a **leaf-state** term (NOT
df-normalized) — b2b is a current-state property like the board eval, unlike the path-cumulative
attack/chain which ARE df-normalized. Verified live (W_B2B 0→5 lifts the best root score on a b2b=12 board
6.4→80). Notes: fusion is demo-only (datagen uses b2b_search, not fusion); the demo loads
`fusion_weights.json` over the C default, and the existing json predates W_B2B so it'll use the 0.5 default
until reseeded/added; CFusionSearch has no decompose binding so W_B2B won't appear in the breakdown strip.

## AlphaZero self-play pipeline for the placement model (2026-06-03)

Added a SECOND placement RL pipeline alongside PPO: an AlphaZero-style self-improvement loop
(PUCT MCTS self-play → train net to imitate the search). Coexists with PPO behind a new
`train placement --algo {ppo,az}` flag (default `ppo`; `az` is placement+single only). The
`fusion/` tree was explicitly out of scope and untouched.

**New files.** `src/qtris/search/placement_mcts.py` (`MCTSConfig`, `PlacementMCTS`) and
`src/qtris/training/placement_az.py` (self-play + MC-return targets + AZ `train_step`).
**Edited.** `cli/train.py` (--algo + all AZ knobs as CLI flags), `observability/models.py`
(extracted `WandbPayloadModel` base from `PPOLogBase`; added `AlphaZeroTrainConfig` +
`SingleAgentAZLog`), `observability/wandb_backend.py` (type hints broadened to `BaseModel` /
`WandbPayloadModel`).

**Design (confirmed with user).** True PUCT MCTS: net policy = priors, value head = leaf eval.
Reuses `clone_sim_env` / `net_input_from_env` / `_policy_value_batch` from `placement_search.py`.
Candidates come from the env's own pathfinder (`_enumerate_placement_candidates` → the full legal
set, no death-pruning), NOT the b2b beam searcher — see the perf note. Search runs across N games at
once — each simulation
batches all games' leaf evals into one net call (candidate enumeration stays per-leaf C-search CPU).
Reward-bearing backup `G = r + γ·V` (dense per-step reward, not a pure terminal outcome) with
per-tree min-max Q-normalization so PUCT's exploration term stays calibrated. Value target = MC
discounted return over the env's `total_reward`, bootstrapped on the live horizon tail; rewards
scaled by the same `return_scale` EMA as PPO. No expert/BC anchor (pure self-play). Warm-starts from
`checkpoints/placement_pretrained_policy`; own checkpoint dir `checkpoints/placement_az`.

**Verified.** ruff format+check clean (no noqa); 2-gen smoke
(`train placement --algo az --num-generations 2 --num-games 4 --horizon 6 --num-simulations 4
--mini-batch-size 8`) warm-starts from BC, runs MCTS self-play, trains, saves, logs (the smoke
checkpoint was deleted so a real run warm-starts cleanly). Value loss starts high (~10) as expected:
return_scale=1 initially, calibrates via the EMA over gens.

**BLOCKER FOUND (pre-existing, NOT from this work).** The uncommitted `model.py` edit added a
`process_obs` override returning **3** values (`piece_dec, board_dec, piece_scores`; at HEAD the
placement model had no override → base returned 2). But `runners/placement.py:169` and
`search/placement_search.py:116` still do `piece_dec, _ = net.process_obs(...)` (2-unpack) → **the
PPO trainer and the neural-guided search demo currently crash** ("too many values to unpack"). The
AZ pipeline handles the 3-return correctly. **FIXED (user go-ahead):** both callers now do
`piece_dec = net.process_obs(...)[0]` (arity-robust). Verified — `train placement` (ppo)
`--num-generations 1` reaches `Gen 0 ... Updates: 4` and closes cleanly.

**Perf diagnosis (2026-06-03).** Profiled because self-play felt way too slow. Dominant cost was
`enumerate_node` (the C beam search) called once per MCTS node (~1040 nodes/move at 16 games × 64
sims). At the demo's `enum_depth=2/beam=512` it was 1.55 ms/call → ~1.6 s/move just enumerating
(measured 4.7 s/move total, **~150 s/gen** for only 512 samples). Key fact (verified across 6 board
states): the candidate SET is depth-invariant — depth controls only the C lookahead SCORES, which
MCTS discards (priors come from the net). So **defaulted `enum_depth=1` (beam 320)**: identical action
space, enum 1.55→0.31 ms, measured **2.1× faster end-to-end (150→70 s/gen)**. depth=0 is NOT faster
end-to-end (2.6 s/move vs 2.2 for depth=1), so depth=1 is the floor. Remaining ~70 s/gen is inherent:
~32k serial clone(0.39ms)+step(0.69ms)+enum(0.31ms) per gen in one Python process (vs PPO's 64
subprocess-parallel envs + 1 net call/step). Visit dist is healthily concentrated (peaked BC prior →
PUCT explores only ~2-10 of 68 candidates, max_pi 0.36-0.58), so 64 sims is fine and gating is
unnecessary. Further speed only via fewer sims/games/horizon (linear, costs samples), a fast custom
clone (the env's `_execute_action` already deep-copies internally, so a shallow clone is feasible but
risky), or process parallelism (loses the cross-game net batching).

**Then switched enumeration to the env pathfinder (2026-06-03, user call).** The b2b `search_with_scores`
did two jobs — enumerate placements AND score them via a `search_depth`-ply lookahead. MCTS only used
the enumeration (priors come from the net), so the scoring was wasted. The env ALREADY enumerates the
legal set via its pathfinder (`CKeySequenceFinder.find_placement_candidates`, what the PPO runner uses).
Measured: pathfinder 0.235 ms vs searcher-depth1 0.378 ms (1.6×), and — key difference — the b2b search
**death-prunes** (68 vs 74 candidates as the board fills) while the pathfinder returns the FULL legal
set. Chose the pathfinder: more AlphaZero-faithful (no external death heuristic filtering the action
space; MCTS discovers death itself and the dead moves become useful negative signal), no risk of the
b2b death-def mismatching the env's, and it lets us **delete `enum_depth`/`enum_beam` + the entire
`CB2BSearch` dependency** from the AZ pipeline (`PlacementMCTS(net, cfg)` now). New `_enumerate()` in
`placement_mcts.py`. End-to-end: **150 → 70 (enum_depth) → 58.5 s/gen (pathfinder), ~2.6× total.**
Smoke re-verified (Gen 0/1, exit 0); ruff clean.

**Demo wired (2026-06-03).** `demo placement --checkpoint checkpoints/placement_az --mcts-sims N`
plays the AlphaZero way: PUCT MCTS (net policy priors + value leaves) picks each move greedily by
visit count (temperature 0, no Dirichlet noise). Added `--mcts-sims` (0=off) + `--mcts-cpuct` to
`cli/demo.py` (mutually exclusive with `--search`); a `PlacementMCTS` branch in `demo/placement.py`
overrides the greedy `predict` move (the predict call stays for the attention panel). Reuses the live
`py_env` (single env, pathfinding=True) — MCTS clones it; the greedy `predict` and the b2b/fusion
`--search` paths are untouched. Verified headless: 8 moves, valid hard-drop sequences, survives.

**Open items.**
- Decide how far to push self-play throughput (fast clone / fewer sims / parallelism) vs accept ~58 s/gen.
- Tune AZ knobs (num_simulations, c_puct, horizon, lr) — defaults are conservative placeholders.
- MCTS is in-process Python (env clones); throughput will be far below PPO's parallel rollouts.
  If too slow, raise `--num-games` (batch) before `--num-simulations`, or revisit later.

## MCTS throughput: Tier 0 + Tier 1 (2026-06-03)

Profiled `PlacementMCTS.search` per-move and bucketed the cost (scratch harnesses in
`/tmp/mcts_prof/`: `profile_mcts.py` full split, `profile_step.py` `_step` internals).
Key finding: the net forward is **overhead-bound, not compute-bound** — ~4.2 ms/call flat
from batch 1→16, so the GPU is far from saturated and `num_simulations` is just a count of
serial GPU round-trips. At the training regime (16 games × 64 sims) the cost was CPU/Python,
not GPU: clone (deepcopy) 26%, env_step 38%, glue 18%, enumerate 13%, net_fwd only 16%.

**Tier 0 — state-only clone.** `clone_sim_env` was `copy.deepcopy(whole env)`. A sim `_step`
only mutates in place: `_scorer`, `_garbage_queue`, `_next_bag`, `_tetrio_rng`, `_random`
(board / active piece / queue are rebuilt, never mutated, so safe to share). Replaced with
`copy.copy` + selective copy of those 5. clone bucket 455→109 ms (4.2×); **~1.2× total.**
Verified: determinism / parent-isolation / garbage-off fidelity all pass.

**Tier 1 — C placement-step (hybrid).** Sub-profiling `_step` showed `_execute_action` is
83% of it and garbage is 0.2%. So only the deterministic lock+clear+spin-attack core moved
to C; **all RNG / queue / bag / hold / garbage / shaping-reward stays in Python** (avoids
bit-replicating TetrioRNG's 7-bag PRNG in C — the scary parity surface that turned out to be
performance-irrelevant). New C entry `b2b_lock_score_c` in `b2b_search.c` reuses the existing
`lock_piece_on_board` + `clear_lines` + `compute_attack` (board passed as plain occupancy,
no GARB markers, so `clear_lines` clears filled garbage rows like Python). ctypes wrapper
`CB2BSearch.lock_score`. New `placement_step` in `placement_search.py` mirrors `_step` but
locks via C and steps by placement descriptor `(is_hold, rot, norm_col, landing_row, spin)`
decoded from the dense action index + `cand_rows` (`build_placement_descriptors` in
`placement_features.py`, same 128-slot order as the net input). MCTS rewired: `_Node.desc`,
`_enumerate` returns a 4-tuple, `_simulate_batch` uses `placement_step` instead of
clone+`_step`. **Note:** this re-introduces a `CB2BSearch()` in `PlacementMCTS.__init__`
(for `lock_score` only — enumeration still uses the env pathfinder), partly walking back the
earlier "deleted CB2BSearch dep" note above; it's now the loader for the lock-score C fn.

Result (16×64): env_step 38% → place_step 6% (~9.4×); **total 1498→979 ms, 1.81× vs original
deepcopy/keystep** (111→61 ms/game-move). 16×256: 1.70×. Behaviour unchanged (proven, below).

**Parity gates (scratch, `/tmp/mcts_prof/`).** `parity_gate.py`: C `lock_score` ==
Python `_lock_piece`+`Scorer.judge` on (board, clears, attack, b2b, combo) — **27,708
candidates, 0 mismatch.** `parity_step.py`: `placement_step` == `_step(key_sequence)` on
total_reward + full next state (board, b2b, combo, garbage, queue, hold, active, terminal) —
**33,227 checks, 0 mismatch.** The float-`logf` combo path never diverges from numpy.

**New bottleneck (16×64):** net_fwd 34% (65 serial batch-16 calls → tf.function / virtual
loss), glue 26% (`_simulate_batch`/`_descend`/`_backup` + 5 arrays/`_Node`), enumerate 23%
(C, but 1040 ctypes crossings + Python packing/move), clone 10%, place_step 6%.

**Tier 2 (intended, not done) — single source of truth.** Goal: extract one `tetris_core`
both `PyTetrisEnv` and `b2b_search` use, deleting the duplicate game logic. Once the C path
is trusted as canonical, **delete the Python `_lock_piece`/`Scorer.judge` duplication and the
parity gates** (per "match perfectly, then remove the gate"). Until then keep `_step`
(real-env move commit still goes through it) and the gates as the safety net — but they live
in `/tmp` (ephemeral): persist or commit before relying on them long-term. Subtree caveat:
`b2b_search.c` edits diverge from upstream `m-sher/TFTetrisEnv` (re-apply after `subtree pull`).

## MCTS Tier 2: fully-C OpenMP sim engine — Phase 1 (enumerate de-risk) (2026-06-03)

Plan file: `~/.claude/plans/giggly-prancing-knuth.md` (approved). Goal: move the entire MCTS sim
loop into C (compact bitboard node, C step+enumerate+reward), persistent tree in C across a move,
ping-pong to Python once per sim round for the batched TF net eval (the only thing that can't be C;
no C→Python callback exists). OpenMP across games/trees. Reward locked to **attack + b2b only**:
per-edge `r = w_attack·attack` (surge+combo already in `compute_attack`'s attack), leaf bootstrap
`v + w_b2b·max(0, b2b_leaf)` (unrealized-hoard credit; mirrors the `W_ATTACK_TOTAL·(total_attack +
max(0,leaf_b2b))` heuristic). **No board-stats / potential / shaping / death-penalty port** — that
whole chunk is dropped. Dirichlet noise + final action sampling stay in Python.

**Phase 1 DECISION (enumerate parity, the flagged top risk): reuse pathfinder.c's
`find_placement_candidates_c` verbatim.** The AZ MCTS enumerates via the env pathfinder
(`CKeySequenceFinder.find_placement_candidates`, full legal set, no death-pruning), NOT b2b
`find_placements` (which death-prunes 68 vs 74). So the C engine must call the SAME function:
- All pieces spawn at `loc=[0,3], r=0` (`_spawn_piece`), so enumerate = `find_placement_candidates_c(
  board_mask, bh, piece_type, 0,3,0, max_len, is_hold, seqs, lr)` for active branch + hold branch
  (hold piece = `_hold_piece` or `queue[0]` if hold empty). landing_rows[160] keyed by
  `rot*40 + norm_col*4 + spin_type`; lr>=0 == legal. Descriptor = decode(slot)+lr (Tier-1's
  `build_placement_descriptors` already does this). Enumeration parity is then **by construction**.
- **Build:** add `TetrisEnv/pathfinder.c` to the b2b_search Extension `sources` in setup.py.
  Function names don't collide (pathfinder unprefixed `init_pieces/check_collision/...` vs b2b's
  `b2b_*`; all overlapping macros identical; pathfinder globals all static). Duplicate symbols across
  b2b_search.so + pathfinder.so are harmless (ctypes default RTLD_LOCAL). Forward-declare
  `find_placement_candidates_c` in b2b_search.c and call it.
- **Thread-safety fix REQUIRED:** `find_placement_candidates_c` uses function-`static` scratch
  (`meta[1600]`, `visited[1600]`, `queue[8192]`) → not reentrant; OpenMP-parallel leaf enumerate
  would race. Fix = drop `static` (≈52KB stack/call, safe; function already re-zeros them, so no
  behavior change for the single-thread env use). pathfinder.c now also subtree-divergent (re-apply
  after subtree pull, same as b2b_search.c).

**Phase 2 incremental build order (each C primitive gated before assembling the tree):**
1. enumerate primitive + setup.py + reentrancy → gate vs `env._enumerate_placement_candidates()`.
2. sim-step primitive (game-loop body + attack/b2b; garbage queue carried) → gate vs `placement_step`.
3. feature-encode (placements[128,18]/mask/desc) → gate vs `build_placement_inference/_descriptors`.
4. PUCT tree + arena + 7 `mcts_*` protocol entries + `CMCTS` ctypes wrapper.
5. rewrite `PlacementMCTS` onto the protocol; callers commit via `placement_step(desc)`; behavioral
   validation (APP/survival/visit dist vs Python MCTS); then delete Python sim path + gates.

## MCTS Tier 2: fully-C OpenMP sim engine — DONE (2026-06-04)

Shipped. The whole PUCT sim loop runs in C on a compact bitboard+scalars node; only the TF net
stays in Python (ping-pong: collect_leaves -> 1 batched net call -> apply_leaves per round).

**Files.** `tetrisenv/TetrisEnv/b2b_search.c` (+~550 lines: MState/MNode/MEngine, per-tree bump
arena, `mcts_apply_step` [mirrors placement_step + the b2b game-loop garbage path], `mcts_enumerate`
[calls pathfinder.c `find_placement_candidates_c`], `mcts_count_holes` [replica], PUCT
select/descend/backup + min-max, 8 exported `mcts_*` entries, 2 `mcts_debug_*` parity hooks).
`pathfinder.c` made reentrant (dropped `static` scratch in find_placement_candidates_c). `setup.py`
links pathfinder.c into the b2b_search extension. New `src/qtris/search/cmcts.py` (ctypes `CMCTS`).
`placement_mcts.py` rewritten (drives CMCTS; `search()` returns `descriptor` not `key_sequence`;
+w_attack/w_b2b in MCTSConfig). `placement_search.placement_step` now returns
`(total_reward, attack, clears, died)`. `placement_az.py` + `demo/placement.py` commit by descriptor
(demo reconstructs the key seq from the env pathfinder for its TF-env render). Removed orphaned
`build_placement_descriptors`.

**Reward (locked).** per-edge `clip(w_attack*attack / return_scale, ±10)`; leaf bootstrap
`v + w_b2b*max(0,b2b)/return_scale`. attack already includes surge+combo (compute_attack). No
board-stats/potential/shaping/death-penalty. w_attack=w_b2b=1.0 default (cfg + `getattr(args,...)`).

**Parity gates (scratch /tmp/mcts_prof; driven by the mcts_debug_* hooks): all PASS.**
enumerate 480 checks (0 extra/0 missing vs env), step 800 checks (board/b2b/combo/garbage/queue/
hold/terminal/attack, 0 mismatch), feature-encode (board/pieces/bcg exact + feature-row multiset).
Enumerate is parity-by-construction (same `find_placement_candidates_c` the env uses). End-to-end:
`train placement --algo az` 2-gen smoke resumes ckpt + learns; search smoke survives 40 moves.

**Throughput (16 games, trained net, GPU): ~3.3x over Tier 0+1.** 64 sims 979 -> ~295 ms/move;
256 sims 4350 -> ~1180-1325. **The win is the C port, NOT threading** — the search is now
*net-bound* (GPU round-trip latency dominates). Two OpenMP footguns found + fixed:
(1) malloc contention -> per-tree bump arena (pre-allocated, no malloc in the parallel region);
(2) libgomp's default pool (all 16 logical cores) idle-spins, stealing CPU from TF -> cap threads
to `min(num_trees, num_procs/4)` via `omp_set_num_threads` (saved/restored around the engine so it
doesn't shrink the b2b beam-search's threads). Default all-cores was *slower than serial* before the
cap. ~10% threading gain on top of the C port; ~456ms outliers are thread-placement variance (no
affinity pinning). Tunable lower via OMP_NUM_THREADS.

**Exploration finding (tuning, not a bug).** With the sparse attack reward + peaked trained prior +
min-max, MCTS concentrates hard at c_puct=1.5 (distinct-visited ~1.7, max_pi ~0.9) vs the old dense
shaping reward (notes: 2-10, 0.36-0.58). Responds correctly to c_puct (sweep: 1.4 -> 5.3 distinct as
c_puct 0.5 -> 20). The min-max `normalize` returning raw Q when max==min (faithful port of the old
`_MinMaxStats`) over-exploits when Q values cluster (sparse reward). If MCTS should add more over the
raw policy, raise c_puct and/or num_simulations, or revisit the Q-normalization.

**Subtree caveat.** b2b_search.c AND pathfinder.c now diverge from upstream `m-sher/TFTetrisEnv`
(re-apply both after a `subtree pull`; re-run the /tmp gates via the mcts_debug_* hooks to confirm).

**Not done (future).** Behavioral A/B vs the old shaping-reward MCTS is moot (reward changed by
design). Thread-affinity pinning (OMP_PROC_BIND) would cut the ~456ms variance. The Tier-2-original
"single tetris_core, delete Python _lock_piece/Scorer.judge" remains out of scope (Python `_step`
still used for dead-root forced-drop + env reset).

## MCTS AZ reward refinements (2026-06-04, follow-up)

Reworked the AZ reward for coherence + AlphaZero-fidelity (search and value now share one objective):
- **Value target = attack-only realized return** (was env `total_reward` with shaping/death/safety).
  `rewards[t,i] = w_attack*attack` in the self-play loop, MC-discounted + tail-bootstrapped as before.
  The b2b term stays a *search-time leaf bootstrap* (`v + w_b2b*max(0,b2b)`), NOT a realized reward,
  so it's deliberately absent from the value target (the value head learns hoard value via realized
  cash-out attacks). This removes the old search(attack+b2b)/value(shaped) mismatch.
- **`return_scale` seeded** via `_estimate_return_var` (one warm-start rollout, attack-only pure-MC
  variance) before the gen loop, skipped when resuming a calibrated AZ ckpt. The EMA weight is 0.01
  (very slow) so a cold start at 1.0 would sit mis-scaled (huge value loss) for ~100s of gens.
- **`w_death` added** (default 5.0, *raw attack units* so it normalizes by return_scale alongside
  attack). Applied to the **terminal edge** in the C search (`mcts_descend_expand`: death = top-out/
  holes OR a resulting no-legal position) AND to the value-target reward on fatal moves (+ dead-root
  branch). Before this, NOTHING penalized death — attack-only rewards are ≥0 so death was just
  truncation-to-0 (weak survival pressure, bad given survival is the stated top priority). This is
  AZ-faithful: death becomes the negative terminal outcome (like AZ's z=-1 for a loss). 5.0 is "same
  scale as a strong clear"; **survival-first may want it larger** (toward return_scale → normalized
  ~ -1). Threaded through MConfig/mcts_create (now 6 floats), CMCTS, MCTSConfig, and `getattr(args,...)`.

**Conceptual anchors (from the design discussion):**
- The AZ value target (MC return) is the AlphaZero-faithful one; the *pretrain* value target
  (`max(cand_scores)/value_scale`, a b2b-heuristic regression) is the non-AZ one. Warm-starting the
  value head from BC still gives useful *relative* leaf rankings (PUCT min-max-normalizes Q), even
  though the absolute value loss spikes until return_scale calibrates.
- The running return-std normalization is **PPO-style**, not AZ (AZ uses a bounded outcome z∈[-1,1],
  tanh head) and not MuZero (value transform + min-max). The min-max Q-norm we have IS MuZero. A more
  AZ-pure option: freeze return_scale at the seed (drop the EMA) for a stable target — offered, not done.
- **Exploration**: at c_puct=1.5 + sparse attack reward + min-max over clustered Q, root visits
  concentrate (~1.7 distinct, max_pi~0.9 -> pi_target ≈ prior argmax -> little policy-improvement
  signal). Responds correctly to c_puct (sweep 1.4->5.3 as 0.5->20). Raise num_sims / c_puct so the
  search actually deviates from the prior, else AZ can't surpass BC.

Gates still green after these changes (enum 480, step 800, features 12; w_death doesn't touch those
paths). AlphaZeroTrainConfig does NOT yet log w_attack/w_b2b/w_death, and cli/train.py doesn't expose
them as flags (only via getattr defaults) - wire if needed.


## MCTS intra-tree leaf batching (virtual loss) — Phase A prototype (2026-06-04)

**Why.** The C MCTS engine is net-bound: the driver does `num_sims` rounds x one-leaf-per-tree =
`num_sims` *sequential* batched net calls/move (each round depends on the prior backup). Net forward
is overhead-bound (~flat batch 1->16), so ~65 fixed-cost calls/move at 16x64 is the wall regardless of
how fast C is. Fix: collect L leaves/tree/round with *virtual loss* so the L descents diverge -> each
net call batches num_trees x L, move needs only ceil(num_sims/L) calls -> ~Lx fewer calls.

**Prototype** (/tmp/mcts_prof/py_mcts_proto.py, throwaway): minimal Python tree mirroring the C engine
(attack-only edge reward + w_b2b leaf bootstrap + w_death terminal, min-max PUCT, env pathfinder enum),
with `_simulate_round(L, vloss)` = up to L descents, `_apply_vloss` per path (N+=1, W-=vloss),
stop-on-collision (reach an awaiting-eval leaf), one batched net call, then `_revert_vloss` + real backup.
Measured vs the L=1 sequential reference (Dirichlet off => deterministic), ckpt placement_az/ckpt-67,
16 games, after 5 warm moves.

**Results.** Net calls drop exactly Lx (65->17 at L=4, ->9 at L=8). **Zero collisions** at 16 games
(vloss spreads descents). vloss=1.0 beats 3.0 (less distortion). Calibration: the L=1 visit dist
naturally drifts KL=0.13 (argmax 15/16) just from sims 64->256. Against that yardstick:
- L=4:  KL 0.26-0.37 (~2-3x drift), argmax 15/16  <- same move-agreement as natural drift. 4x fewer calls.
- L=8:  KL 0.56-0.75 (~5x drift),  argmax 12-14/16. ~7x fewer calls.
- L>=16: distortion grows sharply (argmax -> 8/16). Avoid.
Distortion tracks L, NOT L/num_sims (more sims sharpens the reference, so KL does not recover at 256).

**Decision: GO.** Implement L configurable in C, **default L=4** (4x fewer net calls; move agreement
indistinguishable from sim-budget noise), L=8 available. Virtual-loss distortion is the standard AZ
parallel-MCTS tradeoff; the real quality gate is the Phase-B downstream self-play A/B (APP/survival),
not visit-dist KL. Note: vloss broadening may *help* the over-concentration flagged earlier
("AZ can't surpass BC" at c_puct=1.5) — verify in the A/B.

**Phase B (next).** MNode.awaiting_eval; MTree pending[L]/path[L]; mtree_apply/revert_vloss;
mcts_descend_expand -> mcts_collect_round (L descents, vloss, stop-on-collision); collect_leaves emits
up to num_trees x L rows; apply_leaves reverts vloss + backs up each. cmcts.py: buffers xL, pass
leaves_per_round+vloss, budget-driven loop. MCTSConfig + cli/train.py `--leaves-per-round`. Re-tune
n_threads (per-round C work now ~Lx larger). Rebuild .so; throughput + behavioral A/B.

## MCTS intra-tree leaf batching — Phase B (C engine) done (2026-06-04)

Implemented in b2b_search.c + wired through cmcts.py / placement_mcts.py / cli/train.py. Default L=4.
- MNode.awaiting_eval; MTree holds up to L pending leaves (path[MAX_LPR][MAX_PATH], pending[MAX_LPR],
  n_pending; MAX_LPR=16, clamped). mtree_backup takes (path,len). New mtree_apply_vloss/revert_vloss
  (N+=1,W-=vloss / inverse; do NOT touch minq/maxq). mcts_descend_expand -> mcts_collect_round: up to
  L descents/round, vloss on each pended path, terminal/dead backed up in-place, collision (descend
  into an awaiting_eval child) breaks that descent and retries (C retries vs the prototype's break-
  round; benign, collects closer to L). collect_leaves emits up to num_trees*L rows (tree-major);
  apply_leaves sets prior+value, reverts vloss, backs up each. mcts_create takes leaves_per_round+vloss
  (now 8 int + 6 float + 2 int + (int,float)); arena = num_sims + L + 1. Driver loops ceil(num_sims/L).

**Decisive throughput finding.** Net forward on the RTX 3080 Ti is *flat* ~1.4ms batch 16..256
(ms/sample 0.091 -> 0.005) BUT every *new* batch size pays a ~2s XLA recompile (jit_compile=True).
The per-round leaf count varies with L>1 -> recompile storm that made the first naive L=4 run *slower*
(364ms) despite 4x fewer calls. **Fix: pad every net call to a fixed batch num_trees*L** in
_net_eval (flat net => padding is free; also fixes the latent L=1 recompile when games die). Real
self-play: fixed num_games/L => one compile at startup.

**Warm results** (16 games, 64 sims, ckpt-67):
  L=1: 275 ms/move, 65 calls, argmax 16/16     L=8:  101 ms, 9 calls, 11/16
  L=4: 120 ms/move, 17 calls, argmax 14/16     L=16:  95 ms, 5 calls,  5/16
2.3x at L=4; plateaus past L=4 (a ~90ms non-net floor = per-move CMCTS construct + C sim + per-call
tf.constant/.numpy marshalling now dominates, NOT the net). **Behavioral A/B (greedy, 640 pieces):**
L=1 APP 0.42 / 1 death; L=4 APP 0.55 / 0 deaths -> equivalent-or-better, no degradation (vloss
broadening may even help the over-concentration). End-to-end `train placement --algo az` smoke (1 gen,
isolated ckpt dir) green: collect + train_step + save all run.

**Follow-ups (not done).** (1) The ~90ms non-net floor: reuse one CMCTS engine across a game's moves
instead of construct/destroy per move (callocs 16 pools + omp_set_num_threads churn every move); and
persistent net input tensors to cut per-call tf.constant/.numpy overhead. (2) n_threads cap unchanged
(procs/4); per-round C work grew ~Lx but net/python still dominate - revisit if the floor is attacked.
(3) Larger behavioral A/B (640 pieces is one greedy run) + try num_sims>64 now that it's affordable
(notes flagged over-concentration: "AZ can't surpass BC" at c_puct=1.5).

Demo wired too: demo/placement.py passes leaves_per_round (cli/demo.py `--mcts-leaves`, default 4)
into MCTSConfig; it already commits by descriptor. Single-env demo benefits most (old path = batch 1).

## AZ: evaluate_state potential shaping (phi value-prior) in C MCTS (2026-06-04)

**Why.** AZ self-play was *regressing* the pretrained/DAgger checkpoint: the value head distills the
b2b oracle's evaluate_state (rich board eval = what makes pretrained+MCTS strong), but AZ's attack-only
MC-return value target overwrites that evaluator. DAgger keeps it oracle-grounded; AZ alone abandoned it.

**Fix (PBRS, AZ-faithful).** Inject evaluate_state into the search as potential-based shaping with
phi = the board-state part of evaluate_state. Per-edge reward = w_attack*attack - w_death*dead +
w_phi*(gamma*phi(s') - phi(s)); leaf bootstrap stays the *learned* net value (AZ core intact); phi is a
policy-invariant value-prior, annealable toward pure terminal-reward AZ via w_phi as exploration
catches up. (Chose this over evaluate_state-as-leaf-value, which would be non-AZ heuristic-MCTS; both
cost the same per-node evaluate_state. The user handles exploration / num_sims to actually surpass.)

**phi = sum of evaluate_state_decompose with the path fields (total_attack/pieces_placed/
garbage_prevented) zeroed** -> a pure state fn (board health + b2b store + spin setups). Uses the
TT-free decompose (not evaluate_state) => thread-safe in the OpenMP region. Implemented once in C
(phi_eval); mcts_phi caches MNode.phi at expansion + root; b2b_eval_phi exports it for the trainer.
**Replaced w_b2b end-to-end with w_phi** (the b2b-store term is now inside phi; old leaf bootstrap
w_b2b*max(0,b2b) dropped to avoid double-counting). Trainer value target gets the same shaping
(_shaped_reward, phi via eval_phi on the env before/after each move, phi(terminal)=0); return_scale
seed + per-gen EMA self-calibrate (they use the shaped rewards). cli/train.py --w-phi; demo inherits
the MCTSConfig default. Search and value now optimize ONE objective.

**Verification (gate PASS).** /tmp/mcts_prof/phi_gate.py: b2b_debug_decompose locks each placement and
returns the resulting board + the 13 evaluate_state_decompose components (trusted oracle path); feed
that board back through b2b_eval_phi and compare. **3489 candidates, 0 board-state mismatches
(max err 0.00e+00), 0 path-term leakage, D_ATTACK == W_ATTACK_TOTAL*max(0,b2b) exact.** phi IS the
oracle evaluate_state board-state terms. Cost: ~0.6 ms/node (compute_board_stats); negligible at L=4
(121 vs ~120 ms/move) since the per-call net overhead still dominates there.

**Caveats / tuning (user's training).** phi in raw heuristic units (near-death floor -1e6); watch that
w_phi*(gamma*phi'-phi) stays inside the +/-MCLIP(10) reward clip on non-death moves after return_scale
calibrates - lower --w-phi if it clips. Value head migrates to V_attack - phi (expected PBRS); search
stays phi-guided throughout. Subtree caveat unchanged: b2b_search.c diverges from upstream
m-sher/TFTetrisEnv (re-apply after a subtree pull; re-run the /tmp gates).

### Redesign (same day): phi is a VALUE-TARGET anchor, NOT search shaping

The PBRS-in-the-search version above muddied the edge reward (attack + phi both) and, worse, had the
wrong sign for the actual goal: PBRS with Phi=+phi drives the learned value to V_attack - phi, i.e.
*away* from the pretrained oracle-eval. The real problem is narrower: the AZ value target (attack-only
return) represents a different function (future-attack) than the pretrained value head (board-quality /
oracle eval), so the head must relearn from scratch -> huge initial loss -> MCTS on garbage value ->
collapse. Fixing that is purely a value-target concern; the search reward shouldn't change.

**Final design (user pick: "value target only, clean search"):**
- **Search reward reverted to attack + death** (no phi in the C edge reward; bare net value as the leaf
  bootstrap). Removed MConfig.w_phi, MNode.phi, mcts_phi from C; mcts_create back to 5 reward floats
  (cmcts argpack + CMCTS updated). phi reaches the search *only* through the learned value head.
- **AZ value target gets the phi anchor** (placement_az `_shaped_reward`): per-step reward =
  w_attack*attack - w_death*dead + w_phi*(phi(s) - gamma*phi(s')). Sign is Phi=-phi, so the discounted
  return telescopes to (attack-return + w_phi*phi(s_t)) -> the value head stays ~= the oracle eval it
  was pretrained on (board-quality units), with realized attack as the learned improvement on top. No
  initial misalignment -> no collapse. phi(terminal)=0. Kept b2b_eval_phi + eval_phi/phi_components +
  the gate; return_scale seed/EMA self-calibrate on the anchored returns (phi dominates the variance,
  so return_scale ~ std(evaluate_state) ~ the pretrain value_scale -> scales line up "at least somewhat").
- MCTSConfig.w_phi kept (trainer-only now, not passed to CMCTS); cli/train.py --w-phi help updated;
  demo unaffected.

**Verify (rebuilt):** phi gate still PASS (3489 cand, 0 mismatch - b2b_eval_phi unchanged); engine runs
with the 5-float signature; ruff clean. Tuning: w_phi balances oracle-anchor vs attack-improvement
(higher = more anchored/less collapse-risk but weaker attack signal; 0 = attack-only AZ). The phi value
anchor is also annealable toward pure AZ as the value learns.

## Cleanup audit (2026-06-04)

Full read-through of `src/qtris/` against the standing conventions (no
back-compat by end of refactor, minimal comments, no stale phase-refs). User
triaged the findings; an approved subset was fixed as 5 one-by-one commits, the
rest recorded here for a later pass.

### Fixed (5 commits)
1. **Bugfix** — `training/_1v1.py` `compute_raw_returns(...)` was missing the
   `num_collection_steps`/`num_envs` args required since the Phase 3 GAE
   extraction (would raise if hit). Now mirrors the ar.py call.
2. **Dead demo code** — removed the unreachable `run_eval` + `--eval-steps`
   path and `num_sequences` (+ now-unused numpy import) in `demo/ar_1v1.py`;
   removed the dead `if not pathfinding` branches + `Convert` import in
   `demo/ar.py` and `demo/vs.py` (kept `Keys`, inlined `pathfinding=True`).
3. **Unused constants/classes** — deleted `Paths` dataclass (+ its dead
   dataclass/Path imports) from `config.py`, and `generations = 1_000_000`
   from `training/flat.py` and `_1v1.py`.
4. **Wired `PretrainConfig`** — `RETURN_CLIP_LOW/HIGH` now from `PretrainConfig`
   in `pretraining/base.py`; the four `3e-4` Adam/AdamW LR literals replaced
   with `PretrainConfig().learning_rate` (ar/flat/placement). Stub `batch_size`
   corrected `512 -> 128` to match the CLI default (batch_size/epochs stay
   CLI-driven).
5. **Wired `DataGenConfig`** — single canonical budget `search_depth=16,
   beam_width=200` for ALL families (flat changed `7/96 -> 16/200`, a deliberate
   behavior change). Dropped the unused `gamma` field. gen_ar/gen_placement/
   gen_flat mains now source search_depth/beam_width (+ flat death_trim_count)
   from `DataGenConfig`.

### Kept by explicit user decision (NOT dead-code to remove)
- `EncoderLayer` (`nn/transformer.py`) — unused but intentionally retained.
- `data/convert.py` — orphaned standalone (own argparse + __main__, not wired
  into datagen), kept for now.
- `placement_search.py` beam path — still powers the demo `--search` flag.
- `EnvConfig.pathfinding` field (`config.py`) — unwired; left in place.

### Deferred findings (not fixed — for a later pass)

Back-compat / scaffolding (no-back-compat-by-end-of-refactor candidates):
- `USE_FLAT` global-mutation pattern in `training/_1v1.py` (long-flagged above).
- SimpleNamespace arg-backfill shims: `data/dagger.py` ~382-414, `demo/ar_1v1.py`.
- Old-dataset-schema fallback: `data/gen_placement.py` ~154-165 (pre-128-slot).
- `sample_weights` backfill: `pretraining/base.py` ~269-277.
- Pervasive `getattr(args, "x", default)` fallbacks across placement modules.
- Commented-out wandb resume kwargs: `training/ar.py` ~444-446.
- Stale ctor default `policy_temp=10.0` vs CLI `1.0`: `pretraining/ar.py` ~16.
- Pydantic v1 `.dict()`: `observability/wandb_backend.py` (tied to deferred dep
  modernization above).

Placement subsystem:
- Duplicated constants in `search/cmcts.py` (~20-21) — import
  `CANDIDATE_CAPACITY`/`PLACEMENT_FEATURE_DIM` from `placement_features.py`
  (the single source of truth) instead of redefining 128/18.
- Dead-`predict` dual path in `demo/placement.py` ~210-250 (discards
  `key_sequence` in the MCTS/search branches).
- `if gen % 1 == 0:` always-true gate at `training/placement_az.py` ~397.
- `# noqa: E731` lambdas (`placement_mcts.py` ~57, `cmcts.py` ~160) — violate
  the no-`noqa` rule.

Stale / misleading comments:
- `AsymmetricValueModel` docstring (`models/ar/model.py` ~258-265) describes a
  self-attention-encoder + mean-pool opponent path that no longer matches the
  Flatten+Dense `trunk_a_bcg`/`trunk_b_bcg` impl.
- "X now uses Y / previously" history comments: `pretraining/base.py` (~5-6,
  19-33 surge-correction old-env history), `_1v1.py` ~878 ("Legacy
  definition"), `cli/datagen.py` ~46 ("old default"), old-checkpoint-warmstart
  narration in `models/placement/model.py` (~14-20, 99-100) and
  `pretraining/placement.py` (~190-191), placement-rationale in
  `models/value.py` (~1-5).

Verbose comments vs. the minimal-comment convention:
- Banner dividers in `training/_1v1.py` (~407, 470, 527, 570, 628).
- Long module docstrings: placement_mcts.py, cmcts.py (overlapping),
  placement_az.py, runners/placement.py, models/base.py, models/encoders.py.
- Redundant inline comments restating code throughout `demo/ar.py`, `demo/vs.py`.

Untracked artifacts (not source): `fusion/` (nested Rust reference repo),
`b2b_weights.json`, `Demo.mp4` — consider `.gitignore` or removal.
`tetrisenv/*.c` local edits left out of scope (vendored subtree, kept out of
ruff to avoid diverging from upstream).

## Placement net: dropout support via inference/training jit split (2026-06-04)

`PlacementPolicyValueNet` forward fns were all `@tf.function(jit_compile=True)`. dropout_rate>0 +
training=True puts a stateful dropout RNG op inside the XLA cluster -> compile error (at rate=0 Keras
Dropout is a no-op so it never showed). All four train_steps are already plain @tf.function (non-jit);
the only reason it broke is they call `net(...)`=`call`, which was jit.

Fix (split, single forward source):
- **process_obs / score_candidates / score_value -> plain methods** (dropped their @tf.function). Composed
  by the non-jit training forward (graph-mode dropout) and inlined into the jit inference wrappers
  (training=False -> no dropout op in XLA).
- **call -> @tf.function (non-jit)**: the training forward. pretraining + PPO + AZ train_step keep calling
  `net(..., training=True)`; dropout now works.
- **New jit inference wrappers (training=False hardcoded):** `policy_value(inputs)->(logits,value)` for MCTS
  + PPO rollout + beam `_policy_value_batch`; `state_value(board,pieces,bcg)->value` for the AZ/PPO tail
  bootstrap + beam `_value_batch`. `predict` unchanged (already a jit wrapper; demo + DAgger).
- Repointed all placement-model inference consumers to the wrappers (placement_mcts `_net_eval`,
  placement_az bootstrap, runners/placement step+bootstrap, placement_search both helpers). Training still
  on `call`. So: every inference path is jit; training is graph-mode and supports dropout.

Verified: net built with dropout_rate=0.1 -> call(training=True) no longer raises; policy_value/state_value/
predict compile; inference deterministic; MCTS end-to-end via policy_value works; ruff clean. Pure TF/model
layer change - no .so rebuild.

## AZ value target = max(cand) oracle eval (align with pretraining, fix warm-start collapse) (2026-06-05)

**Problem.** Warm-starting AZ from the pretrained checkpoint collapsed: pretrain value =
max(cand_scores)/value_scale (oracle eval over candidates), but AZ regressed discounted attack-return/
return_scale - a different function AND magnitude. Value head must relearn -> garbage leaf-eval -> collapse.

**Findings (measured, not assumed):**
- value_scale = 764.99 (calibrated at pretrain, saved in the ckpt), NOT 1.0. Datagen beam = depth-16/
  width-200 (config.py). So pretrained V is a deep-beam oracle value, tight band ~0.19 (std 0.10).
- A cheap phi(current-state) target aligns poorly with V (corr 0.21) and has evaluate_state's -1e6
  death floor -> wildly negative targets on near-death states. WRONG target. (Rejected option.)
- max over candidates of full evaluate_state (depth-0 decompose) aligns better (corr 0.43) and dodges
  the death floor (max picks a surviving candidate). Exact depth-16 beam per state = too expensive for AZ.

**Chosen (option 2): AZ value target = max over the oracle's depth-0 candidate evals / value_scale**
(`_max_cand_value` in placement_az -> existing `CB2BSearch.decompose(...).sum(axis=1).max()`). Matches
the pretraining target's FORM (max over candidates), depth-0 vs the datagen beam. Considered "let DAgger
own value, AZ own policy" (option 1) but rejected: shared trunk means policy gradients move the value's
representation regardless, so the value can't be held fixed.

**Implementation (placement_az.py only; no C / cmcts changes needed - uses existing decompose):**
- Restore `value_scale` from the pretrain ckpt (saved there); add to the AZ Checkpoint.
- Per stored (pre-move) state: `value_tgt = _max_cand_value(searcher, env) / value_scale`.
- train_step value loss regresses `batch["value_target"]` (was discounted "returns").
- Removed `_discounted_returns` + the state_value tail bootstrap (value is no longer a return).
- Search untouched: `return_scale`/EMA/`_estimate_return_var` stay (they scale only the SEARCH edge
  rewards now); demo + C search path unchanged.

**Verified.** Rebuilt .so, ruff clean. End-to-end smoke (warm-start from pretrained, MCTS num_sims=16):
gen-0 value_loss = 0.067 (small), value_target mean -0.05 / range [-0.32,0.37] (bounded, no death-floor
blowup), pretrained value_mean 0.17 ~ target -> nothing to relearn -> no collapse. (Greedy-policy play
gave a looser 0.245 MSE; MCTS play is tighter since the death penalty keeps it off near-death states.)

**To test the clean handoff:** start AZ FRESH (move/clear checkpoints/placement_az) so it warm-starts
from the pretrained ckpt. Resuming the existing AZ ckpt starts from the OLD attack-return value head
(units mismatch). Tradeoff: value pinned to the (depth-0) oracle critic -> AZ improves the policy via
search, value won't surpass the oracle. /tmp scratch: value_target_compare.py, az_value_smoke.py.

## Placement PPO expert-loss NaN investigation (2026-06-05)

Dataset `datasets/tetris_oracle_placement` (252,947 rows) inspected for the
PPO expert-anchor NaN (appears ~gen 20-40, term = `expert_loss`).

Findings (dataset is NOT directly poisoned):
- No inf/NaN anywhere (boards, bcg, cand_placements, cand_scores all finite).
- Softmax expert TARGET is finite for every row (0 NaN targets).
- Replayed the EXACT expert loss + gradient over all 252k rows with the BC
  warm-start checkpoint (ckpt-30): 0 NaN batches, max per-row expert_loss=11.4,
  max raw grad-norm=9.7, max|e_logit|=1e9 (the -1e9 mask constant, real logits
  are small). => data + trained model cannot NaN on their own; the NaN requires
  weight drift during PPO and surfaces first in the expert term.

Real dataset pathology to FIX (drives the drift):
- Searcher assigns a hard **-1e6 death penalty** to losing moves. It is > -1e29
  so the expert path (`cand_scores > -1e29`) treats it as a VALID candidate:
  1.39% of all valid slots carry -1e6; 101 rows have best-minus-2nd gap = 1e6.
- Score dynamic range is huge (std 1.17e5, min -1e6, max +1e4). At EXPERT_TEMP=1
  the softmax anchor is a magnitude-driven near-one-hot: 91% of rows put >0.99
  mass on a single candidate. Anchor behaves like hard CE, not a soft target.

TODO:
- Mask death moves (-1e6) to SENTINEL in build_placement_target OR add a
  death threshold to the expert mask (e.g. cand_scores in (-1e5, +inf)).
- Standardize/rank scores per-row (or raise EXPERT_TEMP) so the anchor is a
  genuine soft distribution, not magnitude-driven.
- Add tf.debugging.check_numerics to localize the first NaN; consider a
  separate clipnorm/coef for the expert gradient on the shared trunk.

### Resolution (2026-06-05)
Implemented: expert mask now `cand_scores > MIN_EXPERT_SCORE` (-1e5) in
placement.py, excluding both the -1e6 death penalty and the -1e30 sentinel in
one comparison. Verified on the full dataset: 0 rows go empty (min valid/row=1),
0 NaN batches, max per-row expert_loss=11.4, target mass on death slots = 0.
Still-open (optional): standardize/temper the anchor (91% near-one-hot) and add
check_numerics / separate expert clipnorm if drift recurs.

### Anchor change to hard argmax (2026-06-05)
T=100 soft-anchor still NaN'd at ~same gen count => anchor sharpness is NOT the
driver; expert_loss is a symptom (most sensitive readout), real NaN likely from
the PPO/value path poisoning the shared trunk. Per user direction, switched the
PPO expert anchor from soft-CE over the full softmax to HARD CE to the oracle's
argmax candidate (sparse_softmax_cross_entropy_with_logits); removed EXPERT_TEMP.
NOTE: both pretrainers (pretraining/placement.py, pretraining/ar.py) actually use
soft-CE, so this is stricter than "matching AR". Verified finite over the full
dataset (0 NaN, max CE 11.4, grad-norm 11.5). If this still NaNs at ~same gen,
the PPO ratio exp()-overflow + non-finite KL-break guard are the next suspects.

## BCG normalization swap (2026-06-06, experimental/score-placements)

Replaced log1p BCG compression with fusion-style squash/clip
`min(value / 20.0, 1.0)` in the SHARED `tokenize_bcg`
(`src/qtris/models/encoders.py`). User asked to scope to placement only,
but the fn is shared, so this now affects AR/Flat/value too.

- Divisor 20 for all three (b2b, combo, garbage); saturates at 1.0.
- Old checkpoints' `_bcg_proj_*` Dense weights were fit on log1p inputs —
  distribution shift on resume; intended for fresh training.
- TODO if it underperforms: either (a) re-split per-model so placement-only,
  or (b) revisit caps for high-b2b hoarding regime (>20 collapses to 1.0).

## APP reporting in datagen (2026-06-06)

Added APP (attack per piece) reporting to all four datagen pipelines:
gen_placement, gen_ar, gen_flat, and dagger (both collect_dagger +
collect_dagger_placement). APP = total `reward["attack"]` summed over
env steps / pieces placed (one piece per env._step). Shown in the tqdm/
headless progress postfix (`app=`) and the final summary line (`APP:`).
Each `collect`/`collect_dagger*` now returns app as the last tuple element;
no dataset schema change (reporting only).

## MCTS death-penalty fix (2026-06-06, experimental/score-placements)

Symptom: placement MCTS builds high b2b but never cashes out / breaks it,
dying with the hoard intact; raising w_death did nothing.

Root causes (b2b_search.c): (1) terminal edge `scale(w_attack*attack -
w_death)` was clipped to -MCLIP(=10), so large w_death saturated away;
(2) per-tree min-max Q normalization pinned death to 0 and survival to 1,
capping death's PUCT pull at one normalized unit regardless of w_death;
(3) the leaf bootstrap `+w_b2b*max(0,b2b)` rewards HOLDING b2b and is lost
when b2b breaks, so spending to survive is pure value loss.

Fix applied (chose "make death penalty bite" only):
- Terminal edge now applies `-w_death/return_scale` UNCLIPPED (attack part
  still clipped). 
- Removed min-max Q norm; PUCT uses Q in raw return_scale units. Deleted
  mtree_normalize + MTree.minq/maxq; dropped unused params on mcts_select /
  mtree_backup. Rebuilt .so inplace (smoke-tested).
- Did NOT touch the bootstrap (root cause 3 left intact per b2b-hoard goal).

TODO / watch: with Q now in raw units (not [0,1]), c_puct(=1.5) and
vloss(=1.0) may need retuning - exploration/Q balance changed. If it still
won't cash out to survive, revisit root cause 3 (spendable/decaying hoard
credit).

## Placement AZ: search-bootstrapped value + replay buffer (2026-06-07)

Problem: AZ training (`train placement --algo az`) RESUMED from the BC pretrained
checkpoint made the model worse over generations, even though pretraining hits
~40% top1 / ~65% top3 and AZ *play* (128-1024 sims) plays well.

Diagnosis (three compounding causes):
1. **Value target capped at 1-ply.** The old `_max_cand_value` regressed the value
   head to the oracle's depth-0 `decompose` max, while the pretrained policy came
   from the depth-16 datagen beam. So MCTS bootstrapped a static, myopic critic; the
   visit-count policy target wasn't reliably better than the current policy, and
   distilling it eroded the deep-beam knowledge. (Canonical AZ improves because the
   value learns the true search return - that engine was missing.)
2. **No replay buffer.** ~512 correlated on-policy samples/gen, 2 epochs, then
   discarded -> chases its own shifting self-play distribution, forgets the pretrained one.
3. **Dual-normalizer mismatch.** MCTS leaf bootstrap added net value (value_scale
   units) to edge rewards + b2b (return_scale units) - non-stationary as return_scale
   EMA-drifts.

Fix applied (user chose: full search-bootstrap value + replay buffer; NO BC anchor,
NO new eval harness):
- **Value target = bootstrapped self-play return** (the attack-death return the search
  optimizes), `/return_scale`. Reuses `compute_gae_and_returns` (gae.py) with the net's
  own root value as the horizon bootstrap; `--gae-lambda` default 1.0 (MC return + one
  boundary bootstrap). Deleted `_max_cand_value` and the `searcher.decompose` per-move call.
- **Removed `value_scale` entirely** (Variable, checkpoint entry, warm-start restore).
  Value now lives in return_scale units = same units as the MCTS leaf bootstrap/edge
  rewards, so cause 3 is fixed for free. Warm-start now restores the **policy only**;
  the value head retargets over the first few gens (accepted transient).
- **Root value surfaced** from `PlacementMCTS.search()` (new `"value"` in the result
  dict) + new `PlacementMCTS.root_values(envs)` (collect_roots + one net eval, no sims)
  for the horizon bootstrap. NO C/ctypes changes.
- **Replay buffer**: `collections.deque` of per-gen storable positions, FIFO-evicted past
  `--replay-capacity` (default 25k). Each gen trains `num_epochs * (n_new // batch)` steps
  sampled from the WHOLE buffer (old+new mixed) - per-gen compute unchanged, diversity up.
- Plumbed `--replay-capacity` / `--gae-lambda` through cli/train.py + AlphaZeroTrainConfig;
  added `buffer_size` to SingleAgentAZLog.

Files: training/placement_az.py, search/placement_mcts.py, cli/train.py,
observability/models.py. ruff clean. Smoke (3 gen, sims16/games4/horizon8): warm-start
fires, return_scale seeds, value_loss small, entropy moves 1.98->1.61, no crash. Smoke's
throwaway checkpoints/placement_az was deleted (would otherwise be resumed instead of
warm-starting).

TODO / watch (NOT yet verified): the real test is a 50-100 gen resume run at sims=128 -
confirm value_loss falls / explained_var rises and deaths/attacks/b2b trend UP instead of
the prior decline. If the early-gen value-head transient (oracle-units -> return-units)
hurts the search too much, consider a brief value-only warmup or ramping gae_lambda.
Tune knobs if needed: c_puct/vloss (raw-Q interaction), value_coef, replay_capacity, LR.

### 1k-gen result + w_b2b finding (2026-06-08)

Ran 1k gens (sims=128) resuming from BC. **Value fix WORKED** (explained_var 0.2->0.4,
value_loss flat ~0.5) but the **policy regressed into hoard-to-death**: max_b2b 40->60,
deaths 0.05->0.15, reward 20->12, attack 27->25, clears 17->15, value_mean 0.4->0.2,
return_var 450->600, entropy 2->2.2, policy_loss 2->2.2.

Root cause: the MCTS leaf bootstrap `leaf_value + w_b2b*max(0,b2b)/return_scale` with
w_b2b=1.0 now FIGHTS the (newly meaningful) learned value. At b2b=60, return_scale~24.5:
hoard credit = 60/24.5 ~= **2.45** at every leaf vs value_mean ~**0.2** -> ~92% of the leaf
value is the raw b2b term; the net value is noise next to it. Before the value-target swap
this was harmless (old depth-0 oracle target was itself b2b-dominated, so crutch+target
agreed). Now the value learns the REALIZED return (attack-death, NO b2b term - b2b is only
this bootstrap, never a realized reward), correctly says "high-b2b-about-to-die is worth
little", and w_b2b overrides it. It's also linear+unbounded and cashing out RESETS b2b, so
the search never wants to spend -> b2b climbs to 60 and tops out. This is exactly the
notes.md "root cause 3" (hoard credit lost on break -> spending is pure value loss) biting
in the AZ value context.

Fix: exposed `--w-attack` / `--w-b2b` / `--w-death` (were hardcoded in MCTSConfig); wired
to cfg + AlphaZeroTrainConfig. **Recommended next run: resume with `--w-b2b 0`** - let the
working value head carry b2b's worth (it values b2b because it leads to realized surge
attack). Expect deaths down, max_b2b to stop climbing, attack/clears/reward to recover. If
b2b/APP collapse too far (bot stops building surges), dial w_b2b up to a small nudge
(~0.1-0.25), NOT back to 1.0. Alternative if a small w_b2b still hoards: cap it
(min(b2b,cap)) or make it decaying/spendable (the notes.md root-cause-3 idea), but try 0
first. w_death=100 makes return_var death-spike-dominated (a death = 4x an episode's
attack); fine for survival-first, not the main lever here.

### Demo BCG panel relayout (2026-06-09)

Issue #4 (demo UI rework), first slice: BCG attention heatmaps in the placement/flat/ar
demos moved from 3-across in the top-right corner (labels overlapped each other) to a
vertical column along the right edge (x=720, 90x168 each, column ends ~y=592 - clear of
the y=610 text area). ar_1v1 untouched (own per-player bottom layout).

Remaining for #4: surface more info (max b2b, max combo, largest spike = consecutive
attacks, value estimate, maybe full dist bar chart for placement+flat). The BCG draw
block is now byte-identical x3 across placement/flat/ar - extract into a shared helper
(per issue comment) when doing that info pass; needs a pygame-using module, so demo/utils
or a new demo/panels.py, NOT demo/rendering.py (numpy-only by design).

### Demo info display (2026-06-09)

Issue #4, second slice: new demo/panels.py (pygame-coupled helpers, kept out of
numpy-only rendering.py) with MaxStatTracker + draw_max_stats. placement/flat/ar bottom
panel is now 3 columns (rewards | current state | max stats at x=600, dividers at
335/590 ending y=765) + full-width Action row at y=770 (was overflow-prone in the state
column). Max stats show "episode / run" per user choice; spike = summed attack over
consecutive attacking placements, episode maxes reset at is_last like the running
counters. placement.py also shows Value Est via p_model.state_value (raw net units, the
same units as value_mean in training logs; NOT scaled by return_scale).

Remaining for #4: dist bar chart for placement+flat (user deferred this pass); ar_1v1
max stats (user excluded); BCG draw block still byte-identical x3 - extract into
demo/panels.py alongside the new helpers when next touching the demos.

### Demo dedup extraction (2026-06-09)

Issue #4, third slice (the issue comment's dedup ask). Extracted everything duplicated
x3 across placement/flat/ar:
- rendering.py: colorize_attention_scores (dominant-piece colorization, vectorized;
  verified numerically identical to the old per-cell loop on random inputs).
- panels.py: draw_board_area (surfaces+border+blits, incl. the vis_board branch),
  draw_bcg_panel (+ BCG_* layout constants moved here), draw_step_counter,
  draw_text_column, draw_info_panel (bottom 3-column panel + action row), run_replay
  (full slider/buttons/speed replay UI loop).
- Each demo now defines one draw_bottom_panel(ind) closure used by BOTH the live loop
  (ind=-1; safe because all per-step lists are appended before rendering) and replay.
Line counts: placement 587->348, flat 462->227, ar 560->251.

ar_1v1.py deliberately NOT unified: bespoke two-player layout, screen_w-relative replay
widget positions, return-vs-exit() quit semantics; forcing it through run_replay/
draw_info_panel would mean parameterizing nearly every coordinate.

Pre-existing quirk preserved in run_replay: the "Speed: N FPS" text is blitted BEFORE
the recorded frame (which covers the whole screen), so it's never actually visible.
Was this way in all three demos. FIXED in a follow-up commit same day: speed text now
drawn after the frame blit, on a black backing rect (same pattern as the step counter).

### AZ collapse diagnosis + bootstrap units fix (2026-06-09)

The h=512/w_b2b=0/sims=256 run (resumed from the 1k-gen ckpt-95, NOT a fresh BC start)
collapsed in <10 gens (reward 444 -> -1152, deaths 0.5 -> 12.4, entropy 2.24 -> 3.34).
Compounding causes: (1) UNITS BUG - in placement_az.py the GAE call mixed net values
(return_scale units) with raw rewards, shrinking the horizon bootstrap ~scale x (~25x)
toward zero. At the old h=32 every target was missing 0.21-0.29 normalized units
(value_mean-sized!) -> the value head was trained on ~32-step truncated returns all
along. THIS is why short horizons "didn't see deaths"; h=512 only worked around it.
(2) With w_b2b=0 the Q-signal is value-head-sized; mid-retarget that's ~nothing, so
PUCT visits ~ Dirichlet-noised priors (eps .25, alpha .3 over ~100 branches, 256 sims)
and the policy distills noise -> entropy climbs -> death spiral. (3) updates/gen and
replay span scale with horizon: h=512 -> 64 updates/gen from a ~3-gen buffer (vs 4
from ~49 gens at h=32). Verified numerically: fixed form == closed-form MC + scaled
bootstrap (dones cut tail); buggy form == bootstrap/scale; v_root cancels at lam=1.

FIXED: last_v and v_root now multiplied by scale at the compute_gae_and_returns call
site (one-line semantics; v_root scaling matters only for lam<1 / advantages).

Next run (user launches; descoped by user: no value warmup - fused trunk, no
update/replay decoupling, no reseed flag):
  train placement --algo az --horizon 128 --num-simulations 256 --leaves-per-round 8 \
    --w-b2b 0 --dirichlet-alpha 0.1
- EMPTY checkpoints/placement_az first (holds collapsed ckpt-96/97; trainer resumes
  them otherwise). Fresh BC start auto-seeds return_scale via _estimate_return_var.
- h=128: 16 updates/gen, ~12-gen replay span, gamma^128=0.28 (72% of return mass
  in-window). Fallback h=64 if unstable (8 updates, ~24-gen span, leans on cold value).
- alpha 0.1 ~= 10/branching for ~100 legal placements.
- Watch ~50 gens: entropy must stop climbing (uniform ~4.6), explained_var up, deaths
  flat-to-down. If b2b/APP collapse after value converges, nudge w_b2b to 0.1-0.25.
- If cold-value transient bites again: a HEAD-ONLY value warmup (update value_trunk/
  value_top only, stop-grad into shared trunk) avoids the fused-trunk objection.

### h=128 run (toasty-surf-1710) first 100 gens: stable but b2b-starved (2026-06-09)

Bootstrap fix verified in the wild: value_mean 0.76 -> 2.6 (heading to ~3.7 full-horizon
estimate), explained_var 0.14 -> ~0.4, deaths ~0.2 flat, entropy plateaued ~2.6 (no
collapse). BUT w_b2b=0 + cold value killed surge play at gen 0-2 (avg_b2b 4.6 -> 0.85,
surge_rate 0.44 -> 0.09) -> greedy cash-out local optimum at reward ~100/128 moves. The
predicted chicken-and-egg: value can't learn hoard payoff from data with no hoards.
Organic recovery IS happening (surge_rate back to 0.25, avg_b2b 1.84, attacks ~110+ by
gen 103) but slowly. Action: resume same checkpoint with --w-b2b 0.25 (notes fallback).
Now safe: crutch at b2b=10 is ~0.09 leaf-Q vs value_mean 2.6 (~3%), vs the old 10x-value
domination at w_b2b=1.0. Watch surge_rate/avg_b2b recover toward gen-0 levels (0.4/4.6)
with deaths flat; if b2b re-runs away (max_b2b >> 30 with deaths rising), drop to 0.1.

### TensorBoard observability backend (2026-06-09, issue #11)

tf.summary is now the single observability write path (branch
feature/tensorboard-observability). observability/wandb_backend.py deleted;
observability/backend.py exposes the same init_run/log_step/finish interface:
file writer at tb_logs/<project>/<timestamp>, config logged once as markdown text,
scalars per numeric field, images per _image_fields (board/scores). models.py is
backend-neutral (WandbPayloadModel -> LogPayloadModel, to_payload(), no wandb import).
Default is TB-only (per issue + user choice); `train ... --wandb` opts into mirroring
via wandb.init(sync_tensorboard=True), called BEFORE writer creation so the event
writer gets patched. tb_logs/ gitignored. No new deps (tensorboard 2.15.2 ships w/ TF).
Verified: EventAccumulator round-trip of all scalar tags + board image + config text;
all 5 trainers import clean. NOT yet verified: the --wandb mirror path needs one real
run (sync_tensorboard patch ordering is the fragile spot) - check the first mirrored
run shows scalars AND board images in wandb. View runs: `uv run tensorboard --logdir
tb_logs`. ar.py's commented-out wandb resume kwargs (id=/resume=) were dropped; if
resume-into-same-run is ever needed for TB, point run_name at an existing logdir.
Dev dep added: setuptools<81 (TB 2.15's CLI imports pkg_resources, removed in
setuptools 82; uv venvs don't ship setuptools at all). Drop the pin when issue #6
moves TF past 2.15 (tensorboard >=2.16 no longer imports pkg_resources).

### max_steps truncations were logged as deaths (FIXED 2026-06-09)

The run's "Deaths" had a clockwork spike every 8th gen (0.38-0.69 at gens 7,15,23,...):
8 gens x 128 horizon = 1024 = the env max_steps hardcoded in placement_az. Truncation
sets _episode_ended without death; the collection loop counted it in dones -> (a) logged
as a death, (b) treated as TERMINAL in GAE (bootstrap cut to zero future for a healthy
state) - a periodic downward value bias. TRUE death rate was ~0.05-0.07/game/128 moves,
likely concentrated in the high-garbage envs (chance swept 0->0.2; some of those topouts
may be unavoidable, so 0.00 deaths is not necessarily attainable in the hardest envs).
FIX: placement_az now builds envs with max_steps=None (PyTetrisEnv already supported it:
`False if not self._max_steps else ...`); deaths are the only episode boundary. Side
effect to remember: low-garbage envs may now run very long episodes -> temp-1 opening
moves (move_count resets on death only) become rare there, slightly less opening
diversity. Entropy context from same session: policy entropy ~2.55 = perplexity ~13 of
~100 legal; the Dirichlet-noised pi targets impose a floor ~2.4-2.7 (Dir(0.1) over ~100
legal has E[H]~2.8, mixed at eps 0.25), so the net sits AT its target floor - to sharpen
it, reduce eps or store noise-free pi targets, not value-side knobs.

### AZ b2b: leaf-credit -> potential shaping (death-forfeit) (2026-06-09)

Diagnosed the w_b2b>0 hoard-to-death failure (run `tb_logs/Tetris/20260609_163111`,
w_b2b=0.25): from gen ~66 max_b2b ran to 55 while deaths spiked 0.06->0.88, reward
114->-7, value_mean 2.6->0.44. The value head correctly learned the hoarded dying states
are worthless, but the search's b2b credit overrode it.

Root cause (the long-standing "root cause 3"): the MCTS leaf bootstrap was
`v + w_b2b*max(0,b2b)/scale` (b2b_search.c:3705) - a STANDING bonus on the b2b LEVEL added
at every non-terminal leaf, while the value head was trained on the UNSHAPED realized return
(attack-death, no b2b). That inconsistency = a permanent, un-reconciled hoarding bias:
death is a one-time terminal hit that the recurring frontier +Phi outweighs, and cashing
out (b2b->0) forfeits the credit at all future leaves so the search never wants to spend.

Key finding while designing the fix: NAIVE potential shaping (just add `gamma*Phi(s')-Phi(s)`
to the search edges) is mathematically EQUIVALENT to the current leaf-credit scheme (differs
by a per-root constant) and fixes NOTHING - because the value head is still trained on the
unshaped return. Proper, policy-invariant shaping requires the VALUE TARGET to also regress
the shaped return, so the net learns `V_shaped = V_true - Phi` and the frontier Phi is
reconciled. This is a coordinated 2-part change.

Implemented (chosen over height-gate per user; height-gate is the fallback if this underbuilds):
- **b2b_search.c**: `mphi(cfg,b2b)=w_b2b*max(0,b2b)`. Non-terminal edge +=
  `(gamma*mphi(child)-mphi(parent))/scale` (unclipped). Death edge (terminal, Phi=0) gets
  `-(w_death+mphi(parent))/scale` - dying with a hoard now forfeits the held potential. Leaf
  bootstrap is the net value DIRECTLY (removed the +Phi); the net is V_shaped.
- **training/placement_az.py**: track per-step phi_pre=Phi(s_t) / phi_post=Phi(s_t+1) (0 at a
  terminal boundary, read from `env._scorer._b2b` post-move before reset); add
  `gamma*phi_post - phi_pre` to the realized reward -> `shaped_rewards` feeds both the GAE
  value target AND the return_var EMA. Same shaping added to `_estimate_return_var` (fresh-start
  return_scale seed). `_phi` helper mirrors C `mphi`. Logged avg_total_reward stays RAW.

Properties: policy-invariant (front-loads b2b credit for DISCOVERY, does NOT add a permanent
hoarding preference like the height-gate would); building b2b pays immediately, holding is
~neutral, cashing is neutral (Phi loss cancels the surge gain), dying-with-hoard is PENALIZED.
HONEST caveat: if the true attack-death optimum doesn't favor building b2b enough (greedy
cash-out genuinely higher EV), shaping won't force hoarding - then switch to the height-gate
(deliberate survival-clamped thumb on the scale). Surge economics (surge=b2b on break, linear;
spin/tetris clears out-attack singles) suggest building IS worth it and the problem was only
discovery/credit-assignment, which this targets.

Verified: .so rebuilt clean (setup.py build_ext --inplace), ruff clean, 3-gen smoke (fresh
warm-start from placement_pretrained_policy, num_games4/horizon12/sims16/w_b2b0.25): warm-start
+ return_scale seed fire, shaped MCTS + GAE + train_step run, value_loss finite (0.25->2.2 as
the value retargets to shaped units), deaths 0, no crash. Smoke ckpt + smoke TB run deleted;
`checkpoints/placement_az` left EMPTY (user had cleared it) so the real run warm-starts fresh.

Next run (user launches; FRESH start - placement_az is empty): real horizon/sims, --w-b2b 0.25
to start. Watch: value_mean should NOT collapse, deaths flat, max_b2b should stabilize (not run
to 55) while avg_b2b/surge_rate hold or build. If b2b underbuilds (reverts toward greedy
cash-out like the old w_b2b=0 run), that's the signal to try the height-gate instead. Tunable:
w_b2b sets the discovery-signal magnitude (can go higher now that it's survival-consistent).

### AZ potential-shaping run (20260609_192227): shaping works, noise-distillation decline - WAITING (2026-06-09)

First run on the potential shaping (h=128, sims=256, lpr=8, w_b2b=0.25, alpha=0.1), 233 gens in.

**The shaping fix WORKED for its goal** - no hoard-to-death: max_b2b stable 16-21 (vs the old
runaway to 55), avg_b2b ~3, surge_rate ~0.36, value_mean healthy 2.7-3.0, explained_var 0.3-0.4,
APP ~0.95, no value collapse. Do NOT touch the b2b mechanism for what follows.

**Remaining gradual decline is a NEW failure mode: noise distillation at a converged
equilibrium.** Reward peaked ~117 (rolling-20, ~gen 91) -> ~102; deaths 0.05 -> 0.13; attacks
122 -> 115. Last ~30 gens look like a noisy degraded plateau, not an accelerating collapse.
Evidence chain:
- policy_loss == entropy to within 0.01 the WHOLE run. CE(pi_tgt,p) = H(pi_tgt) + KL(pi_tgt||p),
  so KL ~= 0: the net has fully distilled the search output; no policy-improvement signal left.
- pi targets are the Dirichlet-NOISED visit dists (placement_mcts.py:175 -> placement_az.py:346).
  alpha=0.1 is correctly scaled (~10/n_arms); eps=0.25 is the mixing weight. At 256 sims over
  ~100 arms, noise-boosted arms soak ~5-15 visits each before Q refutes them -> ~10-20% of every
  stored target is noise mass on random placements.
- Entropy CLIMBED from BC's sharp 2.32 to the predicted noise floor ~2.6 and pinned - the policy
  was smeared up to exactly its targets' noise level. (Gens 0-40 still improved, reward 88->115:
  real improvement signal outweighed the smearing until distillation converged ~gen 40-90.)
- Post-convergence each gen's gradient is freshly-random noise (16 updates/gen continue) -> slow
  random walk around the floor; survival is the sharpest-edged behavior (one smeared placement in
  a garbage env = death), so deaths erode first. Value-side wobble (value_loss 0.42->1.0,
  value_mean rolling over late) is DOWNSTREAM of deaths (a death = -3.5 vs typical +2.9 target),
  not the cause - value was at its best exactly when deaths started climbing.
- Why real AZ tolerates noised targets: 800-1600 sims refute noise; value target is the grounded
  game outcome; ~100x more replay averaging. KataGo prunes exploration playouts from its policy
  targets for exactly this reason.

Gen-91 peak ckpt is UNRECOVERABLE (max_to_keep=3 at 5-gen cadence keeps only ~gen 225+). User
declined keep-best checkpointing.

**Decision: WAIT / keep watching** (user choice; run left alive). Re-check TB at gen ~350-400
(~2.5h at ~52s/gen): rolling-20 reward, deaths, entropy, policy_loss - entropy.
- KEEP waiting if reward holds >= ~100 and deaths <= ~0.15 (noisy plateau; equilibrium confirmed,
  decide at leisure).
- STOP waiting (kill + implement fallback below + BC restart) if rolling reward < ~95, deaths
  trend past ~0.2, or entropy climbs > ~2.8 (the old death-spiral signature).
- Recovery above ~110 without intervention would falsify the random-walk story - re-diagnose
  before changing anything.

**Fallback fix, specced (pi-target noise pruning, KataGo-style)** - ready to implement on trigger:
- Keep eps=0.25 Dirichlet in the SEARCH (discovery preserved; a noise arm that pans out earns
  visits via Q and survives pruning).
- In PlacementMCTS.search() result-building (placement_mcts.py:165-181): store res["pi"] as the
  visit dist with arms below pi_prune_frac*total_visits zeroed + renormalized (fallback to
  unpruned if all pruned). Move selection (_select_action) stays on raw counts - pruning affects
  ONLY the stored target.
- MCTSConfig.pi_prune_frac (default 0.03 ~= 8/256 visits), --pi-prune-frac in cli/train.py, field
  on AlphaZeroTrainConfig; new SingleAgentAZLog metrics pi_pruned_mass + pi_target_entropy
  (validates the diagnosis live: expect ~0.1-0.2 pruned mass initially).
- Success criteria for the BC-restart run: entropy settles BELOW the 2.5-2.6 floor; policy_loss
  separates from entropy (KL > 0 = improvement flowing again); deaths <= 0.06; reward exceeds
  117. Escalations if it stalls: sims 256->512, or eps 0.25->0.15.

### AZ wait-window verdict (gens 233-355) + pi-target pruning IMPLEMENTED (2026-06-10)

Wait window resolved to the "noisy plateau - equilibrium confirmed" branch; none of the
stop-triggers fired: rolling-20 reward min 99.5 / mean 104 (trigger <95), deaths max-roll20
0.144 (trigger >0.2), entropy max 2.62 (trigger >2.8), no recovery above 110 either. Reward
slope went -11.2/100 gens (decline phase) -> +0.2 (dead flat); KL still ~=0
(policy_loss - entropy = -0.006); entropy pinned 2.55. Reading: the gens-90-230 decline was
the policy walking DOWN from its BC-boosted peak to the noise floor; it now sits AT the floor.
120 gens bought zero progress -> waiting falsified as a path to improvement. Healthy sides:
value_mean recovered 2.65->2.85, b2b creeping up benignly (max_b2b 17.5->21, surge 0.35->0.40,
deaths flat). DECISION (user): proceed with the pruning fix.

Implemented pi-target noise pruning per the spec above (4 files):
- search/placement_mcts.py: MCTSConfig.pi_prune_frac (default 0.03) + _prune_target() - zero
  arms with counts < frac*total_visits, renormalize; fallback to unpruned if all pruned or
  total==0; frac<=0 disables. Applied in search() result-building: res["pi"] is now the PRUNED
  dist, res["pi_pruned_mass"] the removed visit fraction. Move selection (_select_action) stays
  on RAW counts - exploration unchanged, only the distilled target is cleaned.
- training/placement_az.py: cfg + AlphaZeroTrainConfig plumbing; collects pruned_mass; logs
  pi_pruned_mass + pi_target_entropy (entropy of stored targets - the floor we expect to break).
- observability/models.py: the 2 new SingleAgentAZLog fields ("search" tag group) +
  AlphaZeroTrainConfig.pi_prune_frac.
- cli/train.py: --pi-prune-frac.

Verified: ruff format+check clean; unit check of _prune_target (256 sims, threshold 7.68:
noise tail pruned, mass 0.1445, renorm sum 1.0, disable/zero-visit passthroughs); CPU-only
2-gen e2e smoke from isolated cwd (live GPU run untouched) - new TB tags round-trip
(search/pi_pruned_mass, search/pi_target_entropy, config row pi_prune_frac). Smoke artifacts
deleted. NOTE: at tiny sim budgets the frac threshold is <1 visit -> pruning is a no-op
(pruned_mass 0.0 in the smoke); it needs real sim counts to bite (256 -> >=8 visits to keep).

Next run (user launches, BC restart - EMPTY checkpoints/placement_az first; live run holds
ckpt-73+ there): same flags as 20260609_192227 (--pi-prune-frac 0.03 is now the default).
Watch per success criteria above, plus the new metrics: pi_pruned_mass should log ~0.05-0.2
(validates the noise-mass diagnosis); pi_target_entropy should sit well below the old ~2.5-2.6
net-entropy floor. If pi_pruned_mass < ~0.05 AND entropy doesn't drop below the old floor,
the threshold is under-pruning the strong noise arms (they soak 8-15 visits > 7.68 cutoff) ->
raise --pi-prune-frac to 0.05 (threshold 12.8).

### Pruned run (20260610_091850, BC ckpt-105) gen-198 check-in: WORKING, mechanism confirmed (2026-06-10)

The fix changed the dynamics exactly as predicted, with one corrected estimate:
- pi_target_entropy ~1.8 vs net entropy ~2.5: a sustained ~0.7-nat KL signal (old run had
  KL~=0). Net entropy now DECLINES post-peak (2.65 @ gen 104 -> 2.47) - the old run never
  sharpened. Gap narrowing slowly (0.75 -> 0.68).
- pi_pruned_mass ~0.013, NOT the predicted 0.05-0.2: visits are more concentrated than modeled
  (noise arms either refute to ~0 or earn >=8 visits and survive). The noise damage is mostly
  TARGET INCONSISTENCY across searches (which arms get boosted varies), seen as the persistent
  net-vs-target entropy gap (net fits the mixture). Pruning still helps via sharper targets.
- Performance: peak rolling-20 117.0 @ gen 72 (old 116.8 @ 91); current 108.5 vs old floor 102;
  attacks 118 vs 115. Mid-run sag (gens ~115-165, deaths to 0.15, vloss ~1.0, b2b pushing up)
  PARTIALLY RECOVERED - last 30 gens flat-to-rising. b2b building organically (avg 1.0->2.6,
  max 9->16, surge 0.12->0.36) with deaths contained.
- Caveat: warm-started from BC ckpt-105 (old run used an earlier BC ckpt) - mild confounder.

Escalation rule NOT triggered (pruned_mass < 0.05 but entropy DID break below the old 2.55
floor and is falling) -> keep running. Re-check ~gen 300:
- GOOD: rolling reward >= 110 climbing, entropy < 2.4 falling, deaths <= 0.1 -> fix confirmed.
- STALLED (reward ~108, entropy stuck ~2.45): kill + relaunch with --pi-prune-frac 0.05
  (resumes from latest ckpt - cheap mid-course tweak); sims 256->512 is the heavier lever
  against target inconsistency.

### AZ RESET: pruning + shaping + fade REVERTED; per-gen KL divergence instrumented (2026-06-10)

User direction after the gen-348 diagnosis debate: no ablation matrix, no encoder change
(min(b2b/20,1) squash STAYS - the model only needs "large"), revert the failed attempts, and
test the hypothesis that **the policy diverges too far per generation** (16 updates/gen on
2048 correlated samples outrunning the search's improvement region).

Analytical state of the diagnosis debate (recorded, unresolved): the bootstrap-lag ratchet
hypothesis (uncancelled gamma^(T-t)*Phi(s_T) at the b2b frontier when the bootstrap lags ->
value over-credits frontier building; encoder saturation at b2b=20 makes it permanent above
20) retrodicts: run2's max_b2b slope +3.3 -> +6.1/100g breaking exactly at the first
sustained 20-crossing (gen ~268); run1 (no pruning) never sustained >20 until ~gen 309 and
stayed bounded; run3 lived entirely >20 and ratcheted fastest; run2 corr(avg_b2b, value_mean)
-0.04 pre-knee -> -0.49 post-knee. MISS: the vloss-on-frontier-gens fingerprint is null
(0.98/1.11) - per-gen logged vloss (one minibatch over a 12-gen replay mix) cannot resolve
it. Measured facts that survive regardless: value-head dV/db2b = +0.067/level below 20 (vs
<= ~0.03 physically attainable) and exactly 0 above 20; the edge-credit's pi-target footprint
is ~zero (TV <= 0.024) at both BC and entrenched ckpts; w_b2b=0 rollout of the entrenched
ckpt behaves identically.

REVERTED (no back-compat residue; .so rebuilt):
- pi-target pruning: _prune_target, MCTSConfig.pi_prune_frac, --pi-prune-frac,
  pi_pruned_mass/pi_target_entropy log fields. pi target = raw noised visit dist again.
- b2b potential shaping + height fade: NO b2b term anywhere now (not a revert to the leaf
  credit, also failed). C edge = clip(w_attack*attack/scale) - w_death/scale on death
  (unclipped); leaf bootstrap = net value. Removed mphi/mstate_stack_height/mcts_debug_phi,
  w_b2b from MConfig + mcts_create (6 -> 5 floats; cmcts argtypes updated). Trainer: _phi,
  phi_pre/phi_post, shaped_rewards gone; GAE + return_var on raw attack-death rewards;
  --w-b2b removed.
KEPT (load-bearing): search-bootstrapped value + replay (06-07), GAE bootstrap units fix,
unclipped death + no min-max Q, max_steps=None, leaf batching, BCG squash encoder.

NEW INSTRUMENT (the divergence lens):
- optimization/policy_kl: exact KL(pi_tgt || p_net) on the train batch (= CE - H(tgt)).
- optimization/update_kl: mean KL(p_before || p_after) over the gen's storable states,
  computed via one fixed-shape forward pass before/after the gen's update loop (one trace;
  _gen_log_probs). Printed per gen in the console line.
Verified: ruff clean, .so rebuilt, grep residue-clean, 2-gen CPU smoke green (update_kl
~0.006 at random init - sane), TB round-trip of both tags, config clean, demo imports.

Next run (BLOCKED on user re-pretraining BC - checkpoints/placement_pretrained_policy is
EMPTY, deleted for the tabula-rasa test; quality bar ~40% top-1 / 65% top-3). Also EMPTY
checkpoints/placement_az first (holds the parked tabula-rasa ckpt-1/2). Flags: previous
command minus --w-b2b/--pi-prune-frac. Pre-registered reading of the KL lens:
- Decline recurs with elevated/rising update_kl -> divergence hypothesis supported. Remedy
  ladder (one at a time): --num-epochs 2->1, LR 1e-4 -> 3e-5, replay 25k -> 50k, or a
  PPO-style update_kl early-stop (precedent: target_kl in the PPO trainers).
- Decline recurs with small flat update_kl -> divergence hypothesis disfavored; problem is
  in targets/data, not step size.
- No decline -> the reverted baseline + BC sufficed; the removed interventions were the
  problem.
Parked: tabula-rasa run 20260610_175359 (random init, 9 gens: gens 0-2 dip, 2-8 improving
on deaths/expl_var/entropy/vloss; ckpt-1/2 preserved in checkpoints/placement_az until
emptied).

### Fresh diagnosis: survival-signal starvation + scale-loosening crash loop (2026-06-10)

Restarted the analysis from all runs. The improve-then-crash shape is intervention-
independent because every variant shared the same gradient asymmetry:
1. IMPROVEMENT consumes the survival signal: at peak, deaths = the ONLY survival teacher
   (w_death=100) appear in ~1.1-1.3 of 2048 samples/gen (measured, both BC runs); replay
   holds ~a dozen death events in 25k. Episodes lengthen (max_steps=None), data narrows to
   the policy's groove.
2. SILENT RISK DRIFT: higher-attack play is intrinsically riskier, so the dense attack
   gradient points into risk while the sparse death counter-gradient starves; off-groove
   risk pricing decays. The b2b credit variants were ACCELERANTS of this drift (slope
   changes: no-credit/no-prune -> plateau; leaf credit -> spiral), never the source.
3. FAST CRASH amplifiers once deaths begin:
   (a) scale loop: return_var EMA rises -> return_scale rises -> w_death/scale SHRINKS and
       all Q shrinks vs the fixed-c_puct U term (search more prior-dominated, less able to
       correct). MEASURED sign flip: return_var slope/100g = -126/-169 during improvement,
       +19/+54/+504 during the three decline phases.
   (b) value poisoning: deaths inject -100 targets into a ~+2.9*scale-calibrated head;
       corr(deaths_t, vloss_t) = +0.28/+0.41.
   (c) noise distillation: degraded targets -> entropy climbs (2.24->3.34 in the h=512
       collapse) -> noisier play -> more deaths.
Why WARM starts: BC hands the policy competence the value hasn't earned - deaths are rare
from day one, so risk pricing must be learned from a signal the policy's own competence
suppresses. Tabula rasa (5+ deaths/game) was improving monotonically - densest survival
signal of any run.

Fingerprints to watch in the next run (all already logged): death-events/gen (= avg_deaths
x 16) dropping below ~1-2 at peak; the return_var sign flip preceding the reward fall;
update_kl (elevated -> the divergence hypothesis is an additional amplifier; small + flat
while decline starts -> starvation mechanism primary).
Cure class IF confirmed (not implemented; measure first): freeze return_scale after the
calibration seed (kills amplifier (a); flagged as the AZ-pure option in the 2026-06-04
notes); preferential replay retention of death episodes OR hotter garbage envs so death
data never starves; terminal-state value anchoring.

### Garbage realism: bot attack-distribution measurement (2026-06-11)

Premise (user): training garbage (chance 0.15, rows 1-4) = 0.375 incoming APP - far below the
bot's own output, so the simulated opponent is unrealistically weak. Measured the bot's
OUTGOING attack distribution (ckpt-114, sims=128 greedy, 16 envs x 256 pieces per level,
harness /tmp/measure_attack_dist.py):
- Bot APP ~0.66-0.72, nearly FLAT across incoming gch 0-0.2 (calibration loop converges
  immediately). Event rate ~0.25-0.29/placement; mean chunk ~2.5 (coincidentally = uniform
  1-4 mean, so rows 1-4 is fine at Tier 1).
- Chunk SHAPE is bimodal, not uniform: ~50-60% singles (spin economy) but 16-19% >= 5 rows
  (quads/surges; max single burst 54 rows at gch 0, where the bot hoards to avg b2b 8.9).
  Uniform(1,4) cannot express the lump tail - and lumps are what kill.
- b2b falls with pressure (8.9 -> 0.49 across the sweep); deaths 0-4 per 4096.
Tier-1 mirror match for the CURRENT bot: garbage_chance ~0.28, rows 1-4 (incoming APP ~0.70).
For the 256-sim peak bot (APP ~0.95): ~0.36. Re-measure per milestone.
Tier 2 (env change): empirical chunk table {1:.53, 2:.15, 3:.06, 4:.06, 5-6:.15, 7-9:.03,
10+:.02} in PyTetrisEnv garbage gen. Tier 3 (eventual): mirror-match 1v1 self-play garbage.

### Trace-replay garbage implemented (2026-06-11)

Realistic incoming pressure: envs replay RECORDED attack streams from the bot's own games
instead of Bernoulli(chance) x Uniform(1,4). Difficulty = source strength (same ckpt at
different sims), not event-thinning; variance = library volume x random trace/offset per
episode + traces being unobservable until events hit the queue.

- **PyTetrisEnv** (subtree, Python-only): additive params `garbage_traces` (list of 1-D
  np arrays) + `garbage_trace_cap` (=10; uncapped 54-row recorded surges would insta-kill).
  Trace branch in _add_to_garbage_queue (same queue/delay/cancel mechanics); fresh trace +
  offset drawn per _reset via env RNG (deterministic per seed); stale-trace guard when a
  pool is removed. Cumulative telemetry counters in BOTH modes: _garbage_spawned_rows/
  _spawned_events/_pushed_rows + _garbage_max_event (trainer reads-and-resets). Chance
  mode verified byte-identical (200-step parity check). Subtree diverges further from
  upstream (existing caveat).
- **Library**: dir of tier subdirs of .npy streams; sorted name = difficulty
  (00_sims16/01_sims64/02_sims128/03_sims256/99_recent). Generated from ckpt-114:
  **APP ladder 0.163 / 0.447 / 0.719 / 0.849** - real coherent play at every level.
  `garbage_traces/` gitignored (generated data).
- **Trainer** (placement_az): `--garbage-traces DIR` (None = legacy chance sweep,
  unchanged), `--trace-free-envs` (2), `--trace-harvest-cap` (256). First K envs
  garbage-free, rest split evenly across tiers (weak->strong). **Rolling harvest**: each
  gen's non-zero attack columns -> 99_recent/<epoch>_g<gen>_e<i>.npy (timestamp prefix =
  chronological eviction across resumes); pools re-scanned + reassigned every 25 gens
  (envs re-read pools at reset), so the top tier tracks the current bot automatically.
- **Telemetry** (SingleAgentAZLog): gameplay/garbage_in_app, _in_rate, _in_chunk, _in_max,
  _cancel_frac (approx per gen: queue carryover + reset-dropped count as cancelled);
  progress/trace_pool_size. Config logs garbage_traces + trace_free_envs + the env->tier
  map printed at startup.
- **scripts/measure_attack_dist.py** (promoted from /tmp): measurement mode (per-level
  APP/event-rate/chunk-hist/burst report; tautology line removed) + generation mode
  (--save-traces DIR --tiers 16,64,128,256 --gen-level 0.15).
Verified: env unit checks (event timing, 54->10 clamp, modulo wrap, reset determinism,
chance parity, empty-trace filtering); ruff clean; CPU 2-gen smoke (tier assignment);
GPU 2-gen smoke on real ckpt + dense synthetic tier (harvest fires, eviction keeps newest
N, all 6 metrics round-trip with in_max pinned at cap, cancel_frac in [0,1]).
NOT explicitly exercised: the gen%25 rescan branch (same loader/map code as startup).
Next run: `train placement --algo az ... --garbage-traces garbage_traces` (other flags as
before). Watch garbage_in_app per tier-weighted mean vs the bot's own APP; regenerate the
static tiers via the script when the bot moves a milestone (harvest keeps 99_recent fresh
automatically).
2026-06-11 follow-up: demo wired too - `demo placement --garbage-traces garbage_traces
[--trace-tier 02_sims128]`; default tier = last sorted (99_recent = freshest harvested
mirror once training has run). Demo reuses _load_trace_pools; chance model (0.15) remains
the no-flag fallback.

### Baseline run (20260610_205304) declined on schedule -> return_scale FROZEN (2026-06-10)

The clean baseline (no b2b credit, no pruning, KL instrumented, fresh re-pretrained BC)
peaked at rolling-20 ~108.5 (gens 90-119) and turned at the usual age: by gen 164,
rolling-30 deaths 0.129 (trigger >0.12 FIRED), reward windows 110 -> 101 -> 100 -> 95.6.
DECISIVE negative control: the decline happened with NO b2b machinery and NO hoard runaway
(avg_b2b 2.4 healthy/organic, max_b2b ~16) - the b2b credit variants were accelerants at
most. Instrument readings at the turn: update_kl rising all run (0.026 -> 0.050, spikes
0.083 SIMULTANEOUS with high-death gens - symptom-of-target-noise reading, divergence not
acquitted); return_var tightening exhausted (-719 -> -38/100g, bottoming ~823, last gens
ticking up - amplifier (a) about to engage); death events still starved ~1.5-2/2048 per gen
throughout; entropy at the no-pruning noise floor ~2.65 (no spiral); explained_var fine.

ACTION (user-directed after web check): original AZ has NO return normalization (bounded
z in [-1,1], tanh head); MuZero handles unbounded rewards via IN-TREE min-max Q norm +
value transform - NEITHER runs a return-variance EMA; ours was PPO-style (2026-06-04 notes
said so) and we'd already removed the MuZero min-max (death-penalty fix), leaving the EMA
as the only - and harmful - normalizer. REMOVED the running EMA (placement_az.py):
return_scale is seeded once from _estimate_return_var on fresh start (or restored from
ckpt on resume) and FROZEN; per-gen measured return variance still logged as `return_var`
(diagnostic only - the rise-before-fall fingerprint stays observable). Smoke green (3-gen
CPU, scale stays 1.0 across gens/ckpt round-trip); ruff clean; no C changes.

Next: RESUME the baseline run (latest ckpt, ~gen 160; restored scale ~28.8 freezes at its
tightest/best-calibrated point). Pre-registered reading: SUCCESS = rolling-30 deaths back
<= ~0.1 and reward recovering >= ~105 within ~50 gens (amplifier severed, value catches up
on the now-richer death data). FAILURE = deaths keep climbing with frozen scale -> next
rung of the ladder: --num-epochs 2 -> 1 (the update_kl/divergence remedy). return_var may
now rise freely in the log - that is information, not feedback.

### Pruned run gen-348: b2b hoard-overshoot returns (slow) -> height-faded Phi IMPLEMENTED (2026-06-10)

Gen-348 re-check superseded the stalled/good criteria with a clearer signature: NOT the noise
mode (entropy 2.45 below old floor, KL ~0.73 flowing, targets sharp) but the hoard dynamic
back in slow motion. Peak window (gens 60-120) vs last 30: max_b2b 13.2 -> 25.3 (slope
accelerating +2.9 -> +7.4/100g), avg_b2b 2.2 -> 4.6, deaths 0.069 -> 0.173 (roll-20 hit the
0.2 trigger), reward 114.9 -> 94.5 (-13.2/100g, steeper than the unpruned decline). Value head
responding correctly (explained_var ~0.5 best-ever, value_mean falling) but the hand-set credit
keeps arguing against it at the frontier. Root cause: Phi = w_b2b*max(0,b2b) is UNBOUNDED and
its forfeit arrives only at the terminal cliff - marginal build credit at every b2b level,
death cost discovered too late. Same signature as the leaf-credit spiral, damped ~5x by the
potential machinery (which therefore WORKS - the defect was the Phi shape).

Options weighed: fixed cap min(b2b,12) (blunt - kills safe deep hoards, conflicts with surge
economy); full revert w_b2b=0 (viable NOW with a hoard-experienced value, but relearns the
death boundary from deaths repeatedly and fresh BC starts lose the discovery signal that
already failed once without it); height-gate as standing leaf bonus (recreates the
search/value inconsistency that caused the fast spiral - the original objection applied to
THAT form, not to a potential). USER PICK: height-faded Phi as a potential (still PBRS).

**Phi(s) = w_b2b * max(0, b2b) * max(0, 1 - stack_height/max_height)** - full credit on a low
board, credit drains as the stack approaches the death line (the forfeit is PRE-PAID while
there is still time to cash out, teaching cash-out timing instead of a terminal cliff the
search must discover), already ~0 at death so nothing abrupt is forfeited.

Implemented:
- b2b_search.c: mstate_stack_height (row 0 = top; height = board_height - first occupied row),
  mphi(cfg, MState*) applies the fade; both edge call sites updated; new mcts_debug_phi export
  (parity-gate use only). No mcts_create/cmcts/MCTSConfig signature changes - demo inherits.
- placement_az.py: _phi(w_b2b, env) mirrors exactly (reads env._scorer._b2b + env._board +
  env._max_height); all 4 call sites (collect loop x2, _estimate_return_var x2) updated;
  phi ordering preserved (phi_pre read pre-step; phi_post=0 short-circuit on terminal).
Verified: .so rebuilt; parity gate 675 cases (exhaustive height 0-24 x b2b grid + 500 random
boards): max |C - py - closed-form| err 9.5e-7; semantics spot-checks (empty=full, h=9=half,
h>=18=zero); CPU 2-gen e2e smoke green (the gen-1 value_loss 2444 is the scale=1.0 smoke
artifact: death rows target -100, 2/8 in batch -> ~2450; real runs scale~30); ruff clean.

Next run: resume from the latest pruned-run ckpt (keeps the hoard-experienced value head;
retarget transient Phi_old-Phi_new ~ 0.01 normalized - negligible) or fresh BC, user's call.
Same flags. Watch: max_b2b should STABILIZE (fade caps the incentive in danger) with avg_b2b/
surge_rate holding (low-stack credit intact); deaths back <= 0.1; reward recover toward 117+.
If b2b now UNDER-builds, raise --w-b2b (safer than before - the credit is survival-faded).
Deferred: fold pending garbage into effective height for the fade (deaths concentrate in
high-garbage envs); doubles the C/py consistency surface, do as its own pass with the gate.

### Faded run (20260610_144541) FAILED + full diagnosis: credit is no longer the driver (2026-06-10)

Faded run (resumed from the pruned run's final ckpt) underperformed even the no-intervention
extrapolation at +150 gens: reward 76 (extrap ~95), max_b2b 38 (extrap ~33), deaths 0.24
(extrap ~0.11). NOTE the pruned run had kept running to gen 376 and PARTIALLY SELF-CORRECTED
before the kill (deaths 0.197 -> 0.109, reward 92 -> 101 over its last ~30 gens) - the hoard
ratcheted (max_b2b 24.7) but the system was oscillating, not monotonically diverging.

Diagnostic rollouts (CPU, damaged ckpt-109, 8 envs sweeping garbage 0 -> 0.2, greedy,
sims=128) produced the real diagnosis - three earlier beliefs corrected:

1. **The hand-set b2b credit no longer steers the policy AT ALL.** Falsification test: the
   same ckpt rolled out with w_b2b=0 vs 0.25 behaves identically (avg_b2b 12.2 vs 12.5 at
   gch=0, same heights, same deaths). The hoard is self-sustained by prior+value: sharp prior
   (entropy 2.35) -> prior-dominated PUCT visits at 256 sims -> pruned targets distill the
   prior -> value trained on on-policy hoard data ratifies it. AZ's loop at these exploration
   scales has no mechanism to unlearn an entrenched behavior. COROLLARY: no Phi tweak (cap,
   fade shape, forfeit term, w_b2b knob) can fix the current checkpoint, and resuming it with
   w_b2b=0 would also change nothing quickly.
2. **Hoarding is concentrated where it's safe; deaths are hoard-LESS.** gch<=0.057 envs:
   avg_b2b 8.8-12.5, no deaths. gch>=0.086: avg_b2b ~0.5. The observed deaths happened at
   b2b=-1 (broken), h=17, in garbage envs. The "dying with a hoard / dodged forfeit" story is
   NOT what kills; the reward decline is mostly ATTACK STARVATION from over-holding (APP
   0.5-0.73 in hoard envs vs 0.95 at peak) plus garbage-env survival degradation.
3. **b2b-hoarding mechanically rides a tall stack** (corr(b2b, height) = +0.41; spin-single
   economy nets height) - hoards live at h~11-13 where fade ~ 0.3-0.4. So the fade is in
   effect a ~60% cut of ALL hoard credit (a soft cap), not a targeted death-deterrent. Per #1
   that cut is currently irrelevant anyway.

Why the faded run got WORSE than extrapolation: not the forfeit-dodging (see #2) - most
likely the Phi-redefinition value retarget churn (explained_var 0.74 -> ~0.3-0.5, vloss
rising through the early gens) degraded search-Q quality exactly when the old run had been
self-correcting, plus ratchet momentum. The fade's implementation is verified correct (gate);
its DESIGN was aimed at a gradient that had already finished doing its damage.

Implications for the next attempt (no action taken yet - user explicitly deferred):
- The entrenched ckpt is not fixable by reward knobs; fresh BC is the clean path.
- Both long runs peaked at gens ~60-120 then drifted into hoard-entrenchment. The credit
  helps DISCOVERY early, then over-trains the prior while the realized cash-out signal is too
  sparse to push back and the improvement operator (prior-dominated visits, 256 sims) is too
  weak to re-rank against a sharp prior. Candidate levers for the fresh run, in order of
  theory-fit: (a) anneal w_b2b -> 0 over ~gens 0-100 (credit exactly for the discovery
  window, scheduled handoff to the value); (b) strengthen the improvement operator so the
  (consistently correct) value head can actually overrule the prior: sims 256 -> 512+ and/or
  c_puct tuning; (c) keep-best checkpointing so peaks are never lost again (declined once;
  bit us twice now - both the gen-91 and the pre-fade rollback points are gone).
- The fade-vs-cliff choice for the fresh run is secondary to (a)/(b) per #1/#3; either works
  for discovery. Keep the fade (already implemented + gated) unless it underbuilds early.

### Collapse campaign: instrumentation + hygiene + E0 seed probes (2026-06-12)

Autonomous campaign started on experimental/fix-collapse (living plan: collapse_plan.md).
Run archaeology reframed the problem: NO chance-sweep run ever collapsed (123325: 521 gens,
reward 124.7 ~ 1.05 atk/move); EVERY trace-garbage run collapsed except right-censored
233025. Recovered frozen return_scale: collapsed 132438 = 48.97 (ckpt-66), healthy sweep
lineage = 32.38 ("placement_az copy"/ckpt-112), 233025 inferred ~32.9 via vloss ratio.

Step 1 (f3204ab): --return-scale / --checkpoint-dir / --run-name / --no-harvest /
--trace-tiers / --np-seed; realized scale now in TB config + per-gen scalar (was console-only).
Step 2: immutable snapshots (ladder_v1, weak_v1), poisoned 99_recent (256 deep-collapse
files) quarantined to garbage_traces_archive/. Retrodiction: policy_kl g0-14 does NOT
separate improvers from collapsers (prediction miss, recorded); update_kl g0-14 still splits
(>=.032 vs <=.027, exception 225557).

E0 (8x num-generations 0): seed is near-deterministic per garbage mode - trace ladder
48.43 +/- 0.90, chance sweep 32.00 +/- 4.78. Trace mode itself inflates the seed +50%.
Phase 1 (2x2 scale x difficulty, forced 32.4 vs 49.0, ladder vs weak tiers) running.
