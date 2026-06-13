# AZ warm-start collapse — living experiment plan

Mission: determine the cause of the post-warm-start training collapse, fix it, then meet and
exceed ~1.0 APP on long runs under trace garbage. Rules: trials ≤ 20 min; one trial at a time;
per-trial `--checkpoint-dir trials/<name>`; immutable library snapshots + `--no-harvest` for all
controlled comparisons; results recorded here (numbers + run dirs, no interpretation beyond
hypothesis status) before the next trial.

## Evidence ledger

### Run table (TB archaeology 2026-06-12; all warm-start from BC ckpt-100; 16 games, horizon 128, c_puct 1.5, γ .99, dir_α .1, lr 1e-4, epochs 1 unless noted)

| Run | Sims | vc | Traces | update_kl g0-14 | policy_kl g0-14 | vloss g0-14 | Peak rew @gen | Verdict |
|---|---|---|---|---|---|---|---|---|
| 0611_123325 | 256 | 1.0 | no | .0337 | .804 | — | 124.7 @263 (521 gens) | improving, never collapsed |
| 0611_204333 | 256 | 1.0 | no | .0323 | .626 | — | 121.7 @19 | improving (short) |
| 0611_225557 | 256 | 1.0 | yes | .0697 | .679 | — | 25 @4 | collapsed ~gen 4 (Mode 1) |
| 0611_233025 | 256 | 0.1 | yes | .0344 | .547 | 2.41 | 95.3 @60 | improving (censored @90) |
| 0612_093015 | 128 | 0.1 | yes | .0271 | .576 | — | −40 @8 | collapsed ~gen 8 |
| 0612_102532 | 256 | 0.1 | yes | .0207 | .609 | — | 71.3 @108 | collapsed @108 |
| 0612_132438 | 256 | 0.1 | yes | .0209 | .585 | 1.09 | 43.8 @17 | collapsed ~gen 17 |

Collapse fingerprint: entropy +30-50% (≈2.1→3.5 nats), deaths 0.1→4+, reward → −400s,
update_kl decays to .005-.01. avg_attacks ≈ APP proxy: rew 124.7 w/ 0.10 deaths ≈ 1.05 atk/move.

### Recovered frozen return_scale values

- Collapsed 0612_132438 → **48.97** (read from `checkpoints/placement_az/ckpt-66`).
- Healthy chance-sweep lineage (123325→204333) → **32.38** (`checkpoints/placement_az copy/ckpt-112`).
- Improving trace run 233025 → **≈32.9 inferred** from vloss ratio (vloss ∝ 1/scale²: 48.97·√(1.09/2.41)).
- All known improvers ≈32–33; the one recoverable collapser 49. Scaled death repulsion:
  100/32.4 = 3.09 vs 100/49.0 = 2.04; same 1.5× shrink on every Q gap vs the c_puct·prior term.

### Mechanism facts (code-verified 2026-06-12)

- PUCT score = Q + c_puct·prior·√N/(1+n); Q = mean backed-up discounted return; **no Q
  normalization**; MCLIP=±10 never binds at scale ≥30 (max attack ~20 → ≤0.6 scaled).
- return_scale seeded once per fresh run = √var(MC returns) from ONE stochastic rollout
  (16 envs × horizon=128 steps, MCTS at scale=1.0, temp 1, Dirichlet on), envs already
  trace-loaded during seeding; each death contributes −100 → variance dominated by a small
  random death count; then frozen. Was console-only; now logged (commit f3204ab).
- π target = raw root visit proportions. Loss = CE + value_coef·MSE. No entropy bonus, no
  trust region, no gating, no keep-best (max_to_keep=3, save every 5 gens).
- Trace mode: 2 envs free, 14 split across sorted tiers (00≈0.16 → 03≈0.85 APP, + 99_recent
  self-harvest). Mean incoming ≈0.45 APP vs chance-sweep mean ≈0.19/max 0.375. Harvest mutates
  the shared library dir; tier count (4 vs 5 with 99_recent) shifts the env→tier map between runs.
- Resume restores model+optimizer+return_scale but NOT replay buffer; non-empty ckpt dir
  silently resumes (skips warm start AND seeding).

## Hypotheses

- **H-B** (scale lottery gates improvement): larger frozen scale → smaller scaled Q vs
  c_puct·prior → search ≈ prior → tiny updates while pressure mounts. Status: strong
  correlational support (scale 49 collapsed / ≈32-33 improved; update_kl g0-14 splits
  improvers ≥.032 from collapsers ≤.027, exception 225557). **Retrodiction MISS**: policy_kl
  g0-14 does NOT separate groups (improvers .80/.63/.55 vs collapsers .68/.58/.61/.58) —
  recorded 2026-06-12; the prior-domination story is NOT visible in the π-target distance at
  gen 0-14. Phase 1 is the causal test.
- **H-A** (trace difficulty acts directly): all collapses are trace runs; no chance-sweep run
  ever collapsed. Composes with H-B: difficulty → deaths in seed rollout → inflated scale.
- **H-C** (harvest ratchet): weakened — 225557 collapsed by gen 4, before meaningful harvest.
- **H-D** (no stabilization → any bad update self-amplifies): untested directly.
- Mode 1 vs Mode 2: 225557 (vc 1.0 + traces) collapsed fastest WITH high update_kl .07 —
  treated as a distinct value-flood mode; Phase 1 holds vc = 0.1.

## Experiment queue (decision tree)

- **E0 seed-lottery probe**: 8 × `--num-generations 0` (4 × ladder_v1 traces, 4 × chance
  sweep), fresh ckpt dirs, record `Seeded return_scale=`. Measures lottery spread +
  trace-vs-sweep shift; calibrates the deterministic constant.
- **Phase 1 — 2×2 scale × difficulty** (256 sims, vc 0.1, np-seed 7, no-harvest, ~24 gens):
  T1 ladder_v1@32.4, T2 ladder_v1@49.0, T3 weak_v1@32.4, T4 weak_v1@49.0.
  Pre-registered discriminators: D1 update_kl g1-14 ≥.030 healthy / ≤.025 sick; D3 deaths
  g15-23 ≤0.5 vs ≥1.5; D4 reward g18-23 ≥+60 (ladder) / ≥+90 (weak) vs <0; D5 entropy slope
  g5-23 ≥+0.3 nats = spiral. (D2 policy_kl retained as observational only after retrodiction miss.)
  Branches: scale main effect → 2A; difficulty main effect → 2B; interaction only → 2A+guard;
  all healthy → 2C harvest test + chained segments; all sick → T1@128 sims 46 gens + guards.
- **Phase 2A — deterministic scale** (primary fix): default = fixed constant (~32.4, refined
  by E0). F1 falsification: 093015 config (128 sims) at forced 32.4, 46 gens — healthy
  through gen 40 kills the fastest collapse mode. F2: T2 replicate, np-seed 11 (noise est).
  Fallback: MuZero min-max Q norm in b2b_search.c (w_death/c_puct retune — contingency only).
- **Phase 2B — difficulty path**: trace-free-envs 8; value-target clamp; perf-coupled tiers.
- **Phase 2C — harvest test**: T1 + harvest ON w/ poisoned 99_recent restored, vs T1.
- **Phase 3 — guards**: G1 value-target clamp [−4,+10] scaled, tested on 225557's killer
  config (vc 1.0 + traces @32.4); G2 update_kl circuit-breaker + keep-best second
  CheckpointManager (experiment-scoped; flag for user review before merging — declined
  previously for the main flow).
- **Phase 4 — long-run validation**: R-check (46 straight vs 23+23 resumed; accept if deltas
  < 2× F2 noise). Production: chained ~23-gen segments + APP eval gate per segment
  (`scripts/measure_attack_dist.py`; APP@0.1 ≥ prev−0.03; deaths ≤1.0; entropy not rising 2
  segments; copy passing ckpts aside). Success: avg_attacks/128 ≥ 1.0 sustained 15 gens AND
  measured APP ≥ 1.0.
- **Phase 5 — exceed 1.0**: perf-coupled curriculum + versioned garbage_traces_v2 from new
  best ckpt; A/B vs fixed map; chained long run.

## Results log

### 2026-06-12 Step 1 — instrumentation (commit f3204ab)

Added `--return-scale` (force frozen scale, skip seed rollout; ignored+warned on resume),
`--checkpoint-dir`, `--run-name`, `--no-harvest` (also disables the 25-gen rescan),
`--trace-tiers`, `--np-seed`. Config construction moved after seeding so the realized scale
lands in the TB config text; `return_scale` logged per gen (optimization group). Smoke
(2-gen tiny run, all flags): TB config shows return_scale 32.4 / harvest False / tiers
00_sims16 / np_seed 7; per-gen scalar present; ckpt in /tmp/smoke_az; 99_recent untouched;
`--num-generations 0` seeds and exits 0. ruff clean.

### 2026-06-12 Step 2 — hygiene

- `garbage_traces_snapshots/ladder_v1` (tiers 00-03, 16 traces each) and `weak_v1` (00-01)
  created; poisoned 99_recent (256 files, gens 308-326 of 0612_132438) quarantined to
  `garbage_traces_archive/99_recent_collapse_20260612/`; `garbage_traces/99_recent` now empty.
- policy_kl retrodiction: see run table column — H-B's D2 prediction FAILED retrodictively
  (no group separation); update_kl column unchanged (splits, exception 225557).

## Change log

- f3204ab AZ: trial isolation flags + return_scale logging/override
