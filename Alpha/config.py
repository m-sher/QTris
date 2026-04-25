"""Phase 1 hyperparameters.

Phase 1 = Solo MCTS, no learning.  V comes from the existing 21-component
decomposition (b2b_decompose_c) summed; P is uniform over enumerated
placements.  GPU-resident heads land in Phase 2 — keep this file
opinionated but cheap to edit.
"""

# --- Game / env -------------------------------------------------------
QUEUE_SIZE = 5
BOARD_HEIGHT = 24
GARBAGE_PUSH_DELAY = 1

# --- MCTS -------------------------------------------------------------
NUM_SIMULATIONS = 200            # per move during evaluation
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.0              # off during evaluation; on during self-play (Phase 2)
LEAF_BATCH_SIZE = 32             # Phase 2 batches NN evals; oracle ignores
VIRTUAL_LOSS = 1.0
MAX_PLACEMENTS_PER_NODE = 256    # 2× MAX_PLACEMENTS in C; safe upper bound

# --- Action selection at root ----------------------------------------
TEMP_HIGH_MOVES = 15
TEMP_DECAY_MOVES = 45            # τ linearly decays from 1.0 to 0.3 across these
EVAL_TEMPERATURE = 0.0           # argmax for evaluation runs

# --- Beam-search baseline (for the matched-compute comparison) -------
BEAM_SEARCH_DEPTH = 7
BEAM_SEARCH_BEAM = 128

# --- Eval -------------------------------------------------------------
NUM_EVAL_GAMES = 5
NUM_STEPS_PER_GAME = 200
EVAL_GARBAGE_CHANCE = 0.0        # solo, no random garbage for Phase 1 baseline

# --- Decomposition oracle (Phase 1 valuator) -------------------------
NUM_DECOMPOSE_COMPONENTS = 21    # matches CB2BSearch.NUM_DECOMPOSE
