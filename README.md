# QTris

Vision-transformer reinforcement-learning agents for modern Tetris. A shared transformer
encoder feeds three interchangeable policy families, trained by behavioral cloning from a
fast C beam-search oracle and then refined with PPO or AlphaZero-style MCTS, all in a vendored C
Tetris environment.

| Demo | |
| --- | --- |
| <img src="https://github.com/m-sher/QTris/blob/main/Demo.gif" width="200"> | A 500-piece rollout of an early autoregressive checkpoint, already landing the occasional T-spin, though far from competitive. (This checkpoint predates the back-to-back / combo / garbage inputs the current models receive.) |

## Overview

Every model shares one **vision-transformer encoder**: the board is cut into patches by a
small CNN, the piece queue and the back-to-back/combo/garbage (BCG) scalars become tokens,
and paired cross-attention decoder layers mix board and piece representations. Family-specific
**heads** then turn that latent state into an action. Agents are bootstrapped by **behavioral
cloning** against a C beam-search oracle (`CB2BSearch`), then improved with on-policy
**PPO** (single-player and 1v1 self-play) or, for the placement family, **AlphaZero-style
MCTS** self-play.

## Models

All three share the encoder above and differ only in how they represent an action.

| Family | Action representation | Policy / value | Modes |
| --- | --- | --- | --- |
| **AR** | Variable-length key-press **sequence**, decoded autoregressively one key at a time | Separate policy + `ValueModel` (asymmetric, two-board value in 1v1) | pretrain, PPO single, PPO 1v1 |
| **Flat** | Fixed **320-way categorical** over a pre-enumerated placement set, scored in one pass | Separate policy + `ValueModel` | pretrain, PPO single, PPO 1v1 |
| **Placement** | **Ranks the env's legal candidate placements** via cross-attention (up to 128 candidates) | Merged `PlacementPolicyValueNet` (state-only value) | pretrain, PPO single, AlphaZero-style MCTS |

## Training & data pipeline

- **Pretraining (BC):** distill the beam/oracle datasets into a policy (and value). Soft
  cross-entropy to the oracle's per-candidate scores; value regresses the oracle return.
  `uv run pretrain {ar,flat,placement}`.
- **PPO:** on-policy refinement. **Single-player** for all families; **1v1 self-play** for
  AR and Flat, rotating an opponent pool of past checkpoints through an asymmetric value
  head. Placement PPO can keep a BC **expert anchor** via `--expert-dataset`.
  `uv run train {ar,flat} [--mode 1v1]`, `uv run train placement`.
- **AlphaZero-style MCTS** *(placement only)*: PUCT self-play with a multi-generation replay
  buffer; the policy imitates the search visit counts and the value regresses the
  search-bootstrapped return. `uv run train placement --algo az`.
- **Data generation:** collect expert datasets with the C beam search, or **DAgger**
  (roll a trained policy forward and relabel its states with the oracle).
  `uv run datagen {ar,flat,placement} [--dagger]`.

The expert throughout is the C beam-search engine `CB2BSearch`.

## Quickstart

```bash
uv sync                                                    # install (requires Python 3.11)

# Behavioral-cloning pretrain from the beam/oracle dataset
uv run pretrain placement                                  # or: pretrain ar / pretrain flat

# PPO refinement
uv run train ar                                            # single-player
uv run train flat --mode 1v1                               # 1v1 self-play w/ opponent pool

# AlphaZero-style MCTS self-play (placement only)
uv run train placement --algo az --num-simulations 128

# Generate / relabel training data
uv run datagen placement --steps 200000
uv run datagen ar --dagger --policy-checkpoint checkpoints/ar_pretrained_policy

# Watch a checkpoint play (pygame)
uv run demo placement --checkpoint checkpoints/placement_az --mcts-sims 256
uv run demo vs --left checkpoints/ar_policy_445k --right checkpoints/ar_pretrained_policy
```

Run any command with `--help` for the full flag surface (MCTS knobs, garbage schedule,
search depth/beam, etc.).

## Environment

`tetrisenv/` is a subtree of [TFTetrisEnv](https://github.com/m-sher/TFTetrisEnv).
It provides `PyTetrisEnv` and `PyTetris1v1Env` (tf-agents environments implementing modern
Tetris: SRS rotation, hold, garbage, back-to-back/combo scoring) plus the C
`b2b_search` core (a beam-search oracle and a PUCT MCTS engine) exposed to Python through
`CB2BSearch` and `qtris.search.cmcts`.
