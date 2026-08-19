# QTris

Vision-transformer reinforcement-learning agent for modern Tetris, trained by behavioral
cloning from a fast C beam-search oracle then refined with PPO or AlphaZero-style MCTS, in
a Tetris environment added here as a subtree (mine).

The current placement agent (AlphaZero-style MCTS, trained by 1v1 self-play), ranks env's legal candidate placements optimizing APP while handling incoming garbage:

https://github.com/user-attachments/assets/c9977fe0-0242-44c0-862b-2001b0db2b20

## Overview

The model is a **vision-transformer encoder**: the board is cut into patches by a small
CNN, the piece queue and the back-to-back/combo/garbage (BCG) scalars become tokens, and
paired cross-attention decoder layers mix board and piece representations. The
`PlacementPolicyValueNet` head **ranks the env's legal candidate placements** (up to 128)
by cross-attention and scores the state, one merged policy + value net. It is bootstrapped
by behavioral cloning against a C beam-search oracle (`CB2BSearch`), then improved with
on-policy **PPO** or **AlphaZero-style MCTS** self-play (single-player and 1v1).

## Training & data pipeline

- **Pretraining (BC):** distill the beam/oracle datasets into a policy (and value). Soft
  cross-entropy to the oracle's per-candidate scores; value regresses the oracle return.
  `uv run pretrain`.
- **PPO:** single-player on-policy refinement, which can keep a BC **expert anchor** via
  `--expert-dataset`. `uv run train`.
- **AlphaZero-style MCTS:** PUCT self-play with a multi-generation replay buffer; the policy
  imitates the search visit counts and the value regresses the search-bootstrapped return.
  `uv run train --algo az`, or `--mode 1v1` to rotate an opponent pool of past checkpoints.
- **Data generation:** collect expert datasets with the C beam search, or **DAgger**
  (roll a trained policy forward and relabel its states with the oracle).
  `uv run datagen [--dagger]`.

The expert throughout is the C beam-search engine `CB2BSearch`.

## Quickstart

```bash
uv sync                                                    # install (requires Python 3.11)

# Behavioral-cloning pretrain from the beam/oracle dataset
uv run pretrain

# PPO refinement
uv run train                                               # single-player

# AlphaZero-style MCTS self-play
uv run train --algo az --num-simulations 128
uv run train --algo az --mode 1v1                          # self-play w/ opponent pool

# Generate / relabel training data
uv run datagen --num-steps 200000
uv run datagen --dagger --checkpoint checkpoints/placement_pretrained_policy

# Watch a checkpoint play (pygame)
uv run demo --checkpoint checkpoints/placement_az --num-simulations 256
uv run demo --mode 1v1 --checkpoint checkpoints/placement_az --opponent checkpoints/placement_pretrained_policy
```

Run any command with `--help` for the full flag surface (MCTS knobs, garbage schedule,
search depth/beam, etc.).

## Environment

`tetrisenv/` is a subtree of [TFTetrisEnv](https://github.com/m-sher/TFTetrisEnv).
It provides `PyTetrisEnv` and `PyTetris1v1Env` (tf-agents environments implementing modern
Tetris: SRS rotation, hold, garbage, back-to-back/combo scoring) plus the C
`b2b_search` core (a beam-search oracle and a PUCT MCTS engine) exposed to Python through
`CB2BSearch` and `qtris.search.cmcts`.
