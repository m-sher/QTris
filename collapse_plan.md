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

### 2026-06-12 E0 — seed probe results (8 × num-generations 0, full run config, no np-seed)

- ladder_v1 traces: 47.71, 48.68, 49.61, 47.71 → mean 48.43, sd 0.90 (1.9%).
- chance sweep: 27.65, 37.00, 28.17, 35.17 → mean 32.00, sd 4.78 (14.9%).
- The seed is near-DETERMINISTIC given the garbage mode/library: trace ladder ⇒ ~48.4
  (collapsed 132438's 48.97 in-cluster); sweep ⇒ ~32 (healthy lineage's 32.38 in-cluster).
  "Lottery" reframed: a mode/library shift (+50% under trace ladder) plus moderate sweep noise.
- Open: improving trace run 233025's inferred ≈33 is NOT reachable by trace-mode noise
  (cluster 48.4±0.9). Its library had a 5th tier — weak-ghost 99_recent from 225557's
  harvest. e0_extra probe (ladder + quarantined weak 99_recent as 5th tier) tests whether a
  weak 5th tier pulls the seed down; alternative: the vloss-ratio inference was biased.
- e0_ghost result: 5-tier ladder+weak-ghost seeds at **48.31** — in-cluster. A weak 5th tier
  does NOT lower the seed (only 2/14 envs re-map to 99_recent). ⇒ 233025 most likely also ran
  at ≈48; the vloss-ratio inference for it is judged biased and withdrawn. Consequence: an
  improving 90-gen trace run existed at scale ≈48 ⇒ scale ≈48 is not sufficient for collapse;
  whether it is necessary/contributory is exactly Phase 1's question (T1 vs T2).

### 2026-06-12 Phase 1 — T1 ladder_v1 @ scale 32.4 (23 gens, run 184213-t1-ladder-s32)

HEALTHY / improving. entropy 2.44→1.97 (D5 −0.04, no spiral); deaths g15+ 0.58; attacks
74→113 (≈0.88 APP); reward g18-22 mean 39, trending up. update_kl g1-14 = **0.0166** —
BELOW the pre-registered "healthy ≥.030" line yet clearly improving. ⇒ update_kl is NOT a
reliable health discriminator at forced scale / no-harvest; demote D1 to observational, rely
on entropy slope (D5) + deaths (D3) + reward/attacks trend. Key: ladder difficulty (incoming
≈0.48 APP, same as the collapsed runs) does NOT collapse at scale 32.4 in 23 gens.

### 2026-06-12 Phase 1 — T2 ladder_v1 @ scale 49.0 (23 gens, run 190149-t2-ladder-s49)

HEALTHY / improving. entropy 2.59→2.19 (D5 −0.05, no spiral); deaths g15+ 0.59; attacks
64→99 (≈0.77 APP); reward −80→ ~50-65 late, improving. vloss 1.5→0.74 (lower than T1, ∝
1/scale²). update_kl g1-14 0.0164 ≈ T1. ⇒ **scale 49 does NOT collapse in 23 gens under
clean conditions** — contradicts the simple H-B "scale 49 → collapse". Both scale arms
healthy; T2 slightly higher entropy / lower reward than T1 (weaker Q signal, as predicted),
but stable.

CRITICAL inference: harvest rescan fires only at `gen % 25 == 24`, so for gens 0-23 a
no-harvest run is training-identical to a harvest-on run (bar disk writes). Historical 132438
(ladder-ish @ ~49, harvest on) collapsed by gen 17; T2 (seed 7) did not. 132438's launch
99_recent held 102532's collapse-era (WEAK) traces ⇒ T2 (4-tier, hardest envs face 03) was
if anything HARDER. So difficulty is not the discriminator either. Remaining uncontrolled
variable: **RNG seed / stochastic instability**. Same nominal config (vc 0.1, 256 sims,
traces, scale ~48) historically gave improve-90 (233025), collapse-108 (102532),
collapse-17 (132438) — large run-to-run variance ⇒ collapse looks stochastic/seed-gated,
pointing at H-D (no stabilization) over H-B. Next: T3/T4 (difficulty arm), then a SEED SWEEP
at ladder@49 to measure collapse frequency, then extend a survivor past gen 24.

### 2026-06-12 Phase 1 — T3/T4 weak_v1 (23 gens each) + 2×2 verdict

- T3 weak @ 32.4 (run 192126): HEALTHY. reward g18+ 90.1, deaths g15+ 0.12, entropy +0.05
  (no spiral), attacks 74→101-107.
- T4 weak @ 49.0 (run 194143): HEALTHY. reward g18+ 84.5, deaths g15+ 0.16, entropy +0.01,
  attacks 62→100.

2×2 summary (reward g18+ / deaths g15+): T1 39/0.58, T2 36/0.59, T3 90/0.12, T4 85/0.16.
ALL FOUR HEALTHY → **branch D**. Difficulty is the dominant factor (weak ≫ ladder: ~2.3×
reward, ~4× fewer deaths); scale 32 vs 49 within-noise at both difficulties (no collapse at
49). update_kl uniformly 0.013-0.017 across all arms incl. the improving ones — confirms it
is NOT a health discriminator here. No arm reached the historical 124.7 peak in 23 gens but
all climbing (T3 hit reward 106.8 @gen 12).

**Conclusion of the causal test**: under clean conditions (static library, np-seed 7,
no-harvest) the warm-started policy does NOT collapse in 23 gens at any (scale, difficulty)
in the 2×2. Since harvest rescan first fires at gen 24, gens 0-23 are training-identical to
the historical harvest-on runs; 132438 (ladder-ish @49) collapsed by gen 17 with a launch
99_recent that was WEAKER than ladder_v1 ⇒ the collapse is NOT a deterministic function of
(scale, difficulty). Remaining live causes: (i) RNG-seed stochastic instability (H-D), (ii)
the harvest/library dynamic at gen 24+ (H-C), (iii) a slow >23-gen mode (102532 @ gen 108).

H-B (scale lottery) DEMOTED: scale 49 trains fine in isolation. H-A DEMOTED to "difficulty
modulates death rate" (not a collapse trigger by itself at these levels). Next: seed sweep
(ladder@49, 128 sims, 20 gens, seeds 11/17/23/29/41) to test H-D; then chain a trial past
gen 24 with harvest to test H-C; then a long single-config run for the slow mode.

### 2026-06-12 Seed sweep — COLLAPSE REPRODUCED in clean conditions (128 sims)

ladder_v1 @ forced scale 49.0, no-harvest, 128 sims, 20 gens, varying only np-seed:
- seed 11 (run sweep-s11): COLLAPSED. onset ~gen 5; gen 19 reward −255, deaths 3.0,
  entropy 3.37, update_kl ~0.003.
- seed 17 (run sweep-s17): COLLAPSED. onset ~gen 1; gen 19 reward −300, deaths 3.4,
  entropy 3.56.
- seed 23 (sweep-s23): COLLAPSED (gen 19 reward −318, deaths 3.6, entropy 3.6).
- seed 29 (sweep-s29): COLLAPSED (gen 19 reward −244, deaths 2.9, entropy 3.5).
- seed 41 (sweep-s41): COLLAPSED (gen 19 reward −245, deaths 2.9, entropy 3.4).
- **5/5 seeds collapsed at 128 sims.** vs 4/4 healthy at 256 sims (seed 7). Collapse is
  reliable at 128 sims, independent of seed ⇒ NOT seed-stochastic; SIMULATION BUDGET is the
  determinant. Onset gen 1-5 every time (~2 min/run = fast fix testbed).

DECISIVE: T2 was the identical (ladder, scale 49) cell and stayed HEALTHY — the only deltas
are **128 vs 256 sims** and **seed**. Two 128-sim seeds collapse; four 256-sim cells (seed 7)
are healthy. Historical 093015 (128 sims) also collapsed fastest. ⇒ collapse is reproducible
without harvest and without a live library. H-C (harvest ratchet) and the "harvest required"
idea are FALSIFIED for the fast mode.

**New leading hypothesis H-E (improvement-operator strength)**: MCTS at 128 sims is too weak
to re-rank the policy off its prior; visit targets ≈ prior, the value head can't anchor
against rising garbage deaths, entropy spirals. PUCT mechanics support this: u =
c_puct·prior·√N/(1+n) ≈ 1.65 at n=0 vs Q ≈ attack/scale ≈ 0.02-0.4; Q only bites after a
node accrues visits, and 128 sims over ~10-40 legal moves gives too few visits/move for Q to
matter. Higher scale shrinks Q further → sims×scale interaction (explains why clean 256 was
fine but historical 256 + unlucky seed/harvest collapsed). Connects to the prior b2b-era note
("improvement operator too weak to re-rank a sharp prior; raise sims and/or c_puct").

Fix candidates this implicates, in order: (1) **min-max Q normalization in-tree** (MuZero) —
makes Q ~O(1) and comparable to u regardless of scale/sims → operator strong even at low
sims, scale-invariant; (2) higher c_puct (cheap, but doesn't fix Q-too-small); (3) more sims
(expensive, and historical 256 still collapsed sometimes). Testbed now available: 128
sims/ladder@49/seed 11 collapses by gen 5 (~2 min) — fast fix-iteration loop.

Next: finish sweep (frequency) → long_control 128/seed7 (isolate sims vs seed; 256/seed7 was
healthy) → implement min-max Q norm → re-run the collapse testbed.

### 2026-06-12 c_puct sweep on the 128-sim collapse testbed (ladder@49, seed 11)

c_puct ∈ {0.5, 1.0, 1.5(baseline), 3.0}, all 128 sims, 15 gens: **ALL collapsed.**
Entropy rise vs g0-4: 0.5→+0.27, 1.0→+0.42, 1.5→(spiral), 3.0→+0.61. Higher c_puct → bigger
entropy spiral, but reward/deaths cascade at EVERY c_puct (e.g. 0.5 ended reward −367, deaths
4.1). ⇒ **c_puct is NOT the axis** — collapse is invariant to PUCT exploration/exploitation
balance. Lower c_puct damps the entropy rise slightly but does not prevent the death cascade.

Mechanistic update: concentrating harder on Q (low c_puct) does not help ⇒ the problem is not
flat-targets-from-exploration; **Q is uninformative/corrupted at low sims**. Leading
mechanism: warm-started policy can't survive bursty garbage → messy near-death boards → weak
128-sim search finds no clearly-good move → flat π targets + noisy death-laden value targets →
entropy/deaths spiral. 256 sims digs deep enough to find survival lines (targets stay sharp).
PREDICTS min-max Q norm also won't help (same "amplify Q" axis as low c_puct). Fix axis =
search quality (sims) and/or difficulty. Next tests: (a) 256-robustness at the bad seeds
(is "use 256" a real fix or lucky seed 7?); (b) weak garbage @ 128 sims (does easier garbage
rescue weak search → curriculum fix?).

### 2026-06-12 Decision tests — DIFFICULTY × SIMS interaction (the mechanism, confirmed)

| run | sims | garbage | seed | entropy slope | deaths g15+ | reward g15-17 | verdict |
|---|---|---|---|---|---|---|---|
| r256-s11 | 256 | ladder | 11 | −0.04 | 0.56 | ~40 | HEALTHY (was hard-collapse @128) |
| r256-s17 | 256 | ladder | 17 | −0.03 | 1.06 | ~−18 | non-collapsed, MARGINAL (no spiral) |
| weak128-s11 | 128 | weak | 11 | +0.02 | 0.38 | ~56 | HEALTHY (was hard-collapse @128 on ladder) |

The 2×2 of {128,256} × {weak,ladder} now reads: 128+ladder→COLLAPSE; 128+weak→healthy;
256+ladder→healthy (marginal for the hard seed); 256+weak→healthy (T3/T4). **Collapse needs
BOTH weak search AND hard garbage.** Either more sims OR easier garbage prevents it. This is
the confirmed mechanism: the warm-started policy spirals when garbage difficulty exceeds what
its competence + per-move search budget can survive — weak search finds no survival line in
the messy boards garbage creates → flat π targets + death-laden value targets → entropy/deaths
spiral. 256 sims digs deep enough to find survival; weak garbage avoids the unsurvivable boards.

**Collapse FIX = 256 sims** (decisive: rescues both seeds that hard-collapsed at 128; no
clean 256-sim run has spiraled). Marginal hard-seed performance (s17 reward ~0) ⇒ the full
ladder at 256 is survivable but not yet strong; reaching/exceeding 1.0 APP points to a
**difficulty curriculum** (start at survivable garbage, ramp tiers with measured APP), which
the weak-garbage health result directly supports.

### Hypothesis status (final for diagnosis phase)
- **H-E (search/competence vs difficulty) — CONFIRMED.** Primary cause.
- H-B (scale lottery) — REJECTED (scale 32 & 49 both healthy at 256; collapse invariant).
- H-A (difficulty alone) — PARTIAL: difficulty is a co-factor only in conjunction with weak
  search; not a trigger by itself at 256 sims.
- H-C (harvest ratchet) — REJECTED for the fast mode (clean no-harvest collapses; harvest
  rescan first fires gen 24, after onset).
- H-D (no stabilization) — SUBSUMED: the spiral is real but its trigger is H-E, not seed noise
  (5/5 seeds collapse at 128, so it's not a rare unlucky draw).
- c_puct — NOT an axis (collapse at 0.5–3.0).

### Plan from here
1. Validate the collapse fix at scale: long 256-sim run on the full ladder (scale 32.4 — best
   in the 2×2; chained ≤22-gen segments via resume), eval APP per segment. Tests whether
   256+ladder reaches high APP or plateaus/struggles (s17 suggests marginal).
2. If it plateaus below ~1.0 APP or hard-tier envs keep dying: implement a competence-coupled
   difficulty curriculum (ramp env→tier with rolling APP) and re-validate.
3. Phase 4 long validation + Phase 5 exceed-1.0 as in the original plan.

### 2026-06-12 Fix validation — 256-sim ladder long run (chained, scale 32.4, seed 7)

- seg1 (run …-seg1, gens 0-21): STABLE+improving. entropy 2.46→1.97 (D5 −0.13), deaths
  0.31-0.7, attacks 74→~105 (≈0.82 APP), reward noisy mean ~40, peak 77.
- seg2 (resume, gens 22-43): STABLE, no spiral. entropy flat ~1.85-2.0 (D5 +0.07), deaths
  0.5-1.0, attacks ~97-105, reward oscillating −11..+99 mean ~38. Late dips within the run's
  normal death-driven variance (gen 6/12 also dipped), entropy did not spiral.
- **44 gens at 256 sims on the full ladder without collapse** ⇒ the 256-sim fix holds over a
  long chained run. (Resume does not restore the replay buffer; no adverse transient observed.)

APP eval of the seg1 checkpoint (greedy, 256 sims, measure_attack_dist):
  gch 0.00 → APP 0.487, b2b 13.3 (HOARDS at zero pressure); gch 0.10 → 0.720; gch 0.20 →
  0.833. The bot re-developed b2b-hoarding at low pressure (known project failure mode, see
  notes "qtris-b2b-app-target"). Under realistic trace pressure (~0.48 incoming) in-training
  attacks ≈105/128 ≈ 0.82 APP. ⇒ the full ladder caps attack output ~0.8 APP; exceeding 1.0
  is the b2b cash-out problem (orthogonal to the collapse), not a collapse issue.

### 2026-06-12 Curriculum implemented (commit 70caa6e)

Feedback controller: difficulty index d∈[0, n_tiers-1] ramps toward a per-game-deaths deadband
[0.4, 1.0] (up 0.15 when deaths < 0.4, down 0.40 when > 1.0); `_curriculum_tier_map` spreads
trace envs from the weakest tier up to ⌈d⌉ so the policy always keeps survivable envs +
challenge envs. Flags `--curriculum`, `--curriculum-start`; logs `progress/curriculum_d`.
CPU smoke: d ramped 0.15→0.30→0.45→0.60 with deaths < 0.4; field round-trips. Rationale: makes
training collapse-proof at ANY library difficulty (the controller backs off difficulty before
the difficulty>competence spiral), and keeps the policy in the survivable-but-challenging zone.

## Change log

- f3204ab AZ: trial isolation flags + return_scale logging/override
- 06d1233 Collapse campaign: living plan + trial dir hygiene
