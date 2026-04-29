#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


// ============================================================
// Constants
// ============================================================

#define BOARD_ROWS 40
#define VISIBLE_ROWS 20
#define BOARD_COLS 10
#define ROTATIONS 4
#define SPIN_STATES 2

// Keys (same as pathfinder.c / Moves.py)
#define KEY_START 0
#define KEY_HOLD 1
#define KEY_TAP_LEFT 2
#define KEY_TAP_RIGHT 3
#define KEY_DAS_LEFT 4
#define KEY_DAS_RIGHT 5
#define KEY_CLOCKWISE 6
#define KEY_ANTICLOCKWISE 7
#define KEY_ROTATE_180 8
#define KEY_SOFT_DROP 9
#define KEY_HARD_DROP 10
#define KEY_PAD 11

// Piece Types
#define PIECE_N 0
#define PIECE_I 1
#define PIECE_J 2
#define PIECE_L 3
#define PIECE_O 4
#define PIECE_S 5
#define PIECE_T 6
#define PIECE_Z 7

// Spin types (matching Scorer.py Spins enum)
#define SPIN_NONE 0
#define SPIN_T_MINI 1
#define SPIN_T_FULL 2
#define SPIN_ALL_MINI 3

// Garbage row marker: bit 10 set on a uint16_t row means the row is
// simulated garbage during search — it is treated as occupied but
// cannot be cleared by clear_lines().
#define GARB_ROW_MARKER (1u << 10)  // 0x0400

// BFS limits
#define BFS_QUEUE_CAPACITY 8192
#define BFS_STATE_SPACE (BOARD_ROWS * BOARD_COLS * ROTATIONS)

// Placement limits
#define MAX_PLACEMENTS 512

// Beam search limits
#define MAX_BEAM_WIDTH 256
#define MAX_SEARCH_DEPTH 16

// ============================================================
// Piece / Orientation structures (same as pathfinder.c)
// ============================================================

typedef struct {
    uint16_t row_masks[4];
    int min_col;
    int max_col;
    int min_row;
    int max_row;
    int row_offsets[4];
} PieceOrientation;

typedef struct {
    PieceOrientation orientations[4];
} PieceDef;

// BFS state tracking (for internal placement finder)
typedef struct {
    int16_t parent;
    int8_t last_move;
    int16_t depth;
    int8_t delta_r;
    int8_t delta_row;
    int8_t delta_col;
} BFSStateMeta;

// A single placement result from the internal BFS
typedef struct {
    int rot;
    int col;
    int landing_row;
    int spin_type;      // SPIN_NONE, SPIN_T_MINI, SPIN_T_FULL, SPIN_ALL_MINI
    int delta_r;        // Rotation delta (for scorer logic)
    int delta_loc_sum;  // abs(delta_row) + abs(delta_col) for T-spin mini/full distinction
    int bfs_state;      // BFS state index (for key sequence reconstruction)
} Placement;

// Search state for beam search
typedef struct {
    uint16_t board[BOARD_ROWS];
    int8_t col_heights[BOARD_COLS];  // Cached per-column top-filled height
    int b2b;
    int combo;
    float total_attack;
    float max_single_attack;
    float b2b_attack;          // Attack only from b2b-maintaining clears (spins/tetrises/PCs)
    float max_b2b_attack;      // Largest single b2b-maintaining attack in this path
    int total_lines_cleared;   // Total lines cleared in this search path
    int pieces_placed;         // Pieces placed along this search path (for APP denom)
    int hold_piece;
    int next_queue_idx;
    int depth0_placement_idx;  // Which placement was chosen at depth 0 (for output)
    bool b2b_broken;
    int prev_b2b;
    int garbage_remaining;   // Simulated garbage rows not yet pushed
    int garbage_timer;       // Ticks until next garbage push (decremented on non-clear)
    uint8_t bag_seen;        // Bitmask of pieces consumed from current 7-bag (bits 1-7)
    float score;
} SearchState;

// Fully recompute col heights by scanning every column.  O(BOARD_COLS * board_height).
static inline void compute_col_heights_full(const uint16_t* board, int board_height,
                                            int8_t* out_heights) {
    for (int c = 0; c < BOARD_COLS; c++) {
        out_heights[c] = 0;
        uint16_t bit = (uint16_t)(1u << c);
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) { out_heights[c] = (int8_t)(board_height - r); break; }
        }
    }
}

// Cheap pre-prune: a placement that would leave the effective height inside
// the death zone is unconditionally worse than any alternative; skip it
// before paying for the full compute_board_stats pass in evaluate_state.
static inline bool placement_is_dead(const SearchState* s, int board_height) {
    int mh = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        if (s->col_heights[c] > mh) mh = s->col_heights[c];
    }
    return (mh + s->garbage_remaining) >= (board_height - 4);
}

// ============================================================
// Zobrist hashing for beam-state dedupe
//
// Two beam states that share (board, b2b, combo, hold, next_queue_idx,
// bag_seen, garbage_remaining, garbage_timer) have identical future
// subtrees — expanding both is pure waste.  Dedupe keeps the higher-
// scored path and drops the other.
// ============================================================

#define Z_B2B_SLOTS      64
#define Z_COMBO_SLOTS    64
#define Z_HOLD_SLOTS     8
#define Z_QIDX_SLOTS     32
#define Z_BAG_SLOTS      256
#define Z_GARB_REM_SLOTS 64
#define Z_GARB_T_SLOTS   32

static uint64_t Z_BOARD[BOARD_ROWS][BOARD_COLS];
static uint64_t Z_GARB_ROW[BOARD_ROWS];
static uint64_t Z_B2B[Z_B2B_SLOTS];
static uint64_t Z_COMBO[Z_COMBO_SLOTS];
static uint64_t Z_HOLD[Z_HOLD_SLOTS];
static uint64_t Z_QIDX[Z_QIDX_SLOTS];
static uint64_t Z_BAG[Z_BAG_SLOTS];
static uint64_t Z_GARB_REM[Z_GARB_REM_SLOTS];
static uint64_t Z_GARB_T[Z_GARB_T_SLOTS];
static bool zobrist_initialized = false;

static inline uint64_t splitmix64(uint64_t* x) {
    *x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = *x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void zobrist_init(void) {
    uint64_t s = 0xb2bc0de51ce5eedULL;
    for (int r = 0; r < BOARD_ROWS; r++)
        for (int c = 0; c < BOARD_COLS; c++) Z_BOARD[r][c] = splitmix64(&s);
    for (int r = 0; r < BOARD_ROWS; r++) Z_GARB_ROW[r] = splitmix64(&s);
    for (int i = 0; i < Z_B2B_SLOTS; i++)      Z_B2B[i] = splitmix64(&s);
    for (int i = 0; i < Z_COMBO_SLOTS; i++)    Z_COMBO[i] = splitmix64(&s);
    for (int i = 0; i < Z_HOLD_SLOTS; i++)     Z_HOLD[i] = splitmix64(&s);
    for (int i = 0; i < Z_QIDX_SLOTS; i++)     Z_QIDX[i] = splitmix64(&s);
    for (int i = 0; i < Z_BAG_SLOTS; i++)      Z_BAG[i] = splitmix64(&s);
    for (int i = 0; i < Z_GARB_REM_SLOTS; i++) Z_GARB_REM[i] = splitmix64(&s);
    for (int i = 0; i < Z_GARB_T_SLOTS; i++)   Z_GARB_T[i] = splitmix64(&s);
    zobrist_initialized = true;
}

typedef struct {
    uint64_t hash;
    int32_t idx;
} HashSlot;

static inline uint64_t state_hash_rows(const uint16_t* board, int board_height,
                                       int b2b, int combo, int hold_piece,
                                       int next_queue_idx, uint8_t bag_seen,
                                       int garbage_remaining, int garbage_timer) {
    uint64_t h = 0;
    for (int r = 0; r < board_height; r++) {
        uint16_t row = board[r];
        uint16_t play = (uint16_t)(row & ((1u << BOARD_COLS) - 1u));
        while (play) {
            int c = __builtin_ctz(play);
            h ^= Z_BOARD[r][c];
            play &= (uint16_t)(play - 1);
        }
        if (row & GARB_ROW_MARKER) h ^= Z_GARB_ROW[r];
    }
    int b2b_i = (b2b < 0) ? (Z_B2B_SLOTS - 1) : (b2b % (Z_B2B_SLOTS - 1));
    int combo_i = (combo < 0) ? (Z_COMBO_SLOTS - 1) : (combo % (Z_COMBO_SLOTS - 1));
    h ^= Z_B2B[b2b_i];
    h ^= Z_COMBO[combo_i];
    h ^= Z_HOLD[hold_piece & (Z_HOLD_SLOTS - 1)];
    h ^= Z_QIDX[next_queue_idx & (Z_QIDX_SLOTS - 1)];
    h ^= Z_BAG[bag_seen];
    int gr_i = (garbage_remaining < 0) ? 0
             : (garbage_remaining >= Z_GARB_REM_SLOTS ? (Z_GARB_REM_SLOTS - 1) : garbage_remaining);
    int gt_i = (garbage_timer < 0) ? 0
             : (garbage_timer >= Z_GARB_T_SLOTS ? (Z_GARB_T_SLOTS - 1) : garbage_timer);
    h ^= Z_GARB_REM[gr_i];
    h ^= Z_GARB_T[gt_i];
    return h;
}

// ============================================================
// Bag Tracking Helpers (for speculative search beyond known queue)
// ============================================================

// Update bag tracking when a piece is consumed.
// Returns the new bag_seen bitmask.
// When all 7 pieces (PIECE_I=1 through PIECE_Z=7) have been consumed,
// resets to 0 to represent a fresh bag.
static uint8_t bag_consume_piece(uint8_t bag_seen, int piece_type) {
    bag_seen |= (uint8_t)(1 << piece_type);
    // Bits 1-7 all set = 0xFE means all 7 pieces consumed
    if ((bag_seen & 0xFE) == 0xFE) bag_seen = 0;
    return bag_seen;
}

// Get the remaining pieces in the current bag.
// Writes piece types to out_pieces[], returns count.
static int bag_get_remaining(uint8_t bag_seen, int* out_pieces) {
    int count = 0;
    for (int p = PIECE_I; p <= PIECE_Z; p++) {
        if (!(bag_seen & (1 << p))) out_pieces[count++] = p;
    }
    return count;
}

static inline uint64_t state_hash(const SearchState* s, int board_height) {
    return state_hash_rows(s->board, board_height,
                           s->b2b, s->combo, s->hold_piece,
                           s->next_queue_idx, s->bag_seen,
                           s->garbage_remaining, s->garbage_timer);
}

// Transposition-cache hash: like state_hash but also mixes in path-dependent
// fields that feed evaluate_state (total_attack, b2b_attack, max_single,
// pieces_placed).  Two states sharing board+b2b+combo+hold+queue_idx but
// arriving via different clear histories would otherwise map to the same
// TT slot and return the wrong cached score.
static inline uint64_t tt_hash(const SearchState* s, int board_height) {
    uint64_t h = state_hash(s, board_height);
    union { float f; uint32_t u; } a, b, c;
    a.f = s->total_attack;
    b.f = s->b2b_attack;
    c.f = s->max_single_attack;
    h ^= (uint64_t)a.u * 0x9e3779b97f4a7c15ULL;
    h ^= (uint64_t)b.u * 0xbf58476d1ce4e5b9ULL;
    h ^= (uint64_t)c.u * 0x94d049bb133111ebULL;
    h ^= (uint64_t)(uint32_t)s->pieces_placed * 0x1f2c5d1e9b7e8f3dULL;
    return h;
}

// ── Cross-call transposition cache ───────────────────────────
// Caches leaf evaluate_state scores across successive b2b_search_c calls.
// Since the queue shifts by one piece per external call, ~6/7 of candidate
// leaves reuse work from the previous call.  Entries older than
// TT_GENERATION_EXPIRY generations are treated as misses.
#define TT_SIZE (1u << 16)
#define TT_MASK (TT_SIZE - 1u)
#define TT_GENERATION_EXPIRY 4u

typedef struct {
    uint64_t hash;
    float    score;
    uint32_t generation;
} TTEntry;

static TTEntry g_tt[TT_SIZE];
static uint32_t g_tt_generation = 0;
static bool g_tt_initialized = false;

static inline void tt_reset(void) {
    memset(g_tt, 0, sizeof(g_tt));
    g_tt_generation = 1;
    g_tt_initialized = true;
}

static inline void tt_new_generation(void) {
    if (!g_tt_initialized) { tt_reset(); return; }
    g_tt_generation++;
}

// Min-heap sift-down over the first `size` elements of `heap`.
static void beam_sift_down(SearchState* heap, int size, int i) {
    while (1) {
        int smallest = i;
        int l = 2 * i + 1, r = 2 * i + 2;
        if (l < size && heap[l].score < heap[smallest].score) smallest = l;
        if (r < size && heap[r].score < heap[smallest].score) smallest = r;
        if (smallest == i) break;
        SearchState tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
    }
}

// Keep the top-K highest-scored entries in `beam[0..K-1]` (unsorted, in
// min-heap order — the final pick is done by a linear max-scan anyway).
// O(N log K) vs qsort's O(N log N); N is beam_width*MAX_PLACEMENTS ≈ 130k
// so the saving per depth level is substantial.
static int beam_select_top_k(SearchState* beam, int n, int K) {
    if (n <= K) return n;
    // Heapify the first K (build min-heap)
    for (int i = K / 2 - 1; i >= 0; i--) beam_sift_down(beam, K, i);
    // Walk the rest; anything better than the current min replaces the root.
    for (int i = K; i < n; i++) {
        if (beam[i].score > beam[0].score) {
            beam[0] = beam[i];
            beam_sift_down(beam, K, 0);
        }
    }
    return K;
}

// Deduplicate a freshly-expanded beam in place.
// For each unique state hash, keeps the entry with the highest score.
// Returns the new length of the compacted prefix.
// `table` is a caller-supplied buffer of size `cap` (must be a power of 2).
static int dedupe_beam(SearchState* beam, int n, int board_height,
                       HashSlot* table, int cap) {
    if (n <= 1) return n;

    int mask = cap - 1;
    for (int i = 0; i < cap; i++) { table[i].hash = 0; table[i].idx = -1; }

    int w = 0;
    for (int i = 0; i < n; i++) {
        uint64_t h = state_hash(&beam[i], board_height);
        if (h == 0) h = 0x9e3779b97f4a7c15ULL; // reserve 0 as "empty"

        int slot = (int)(h & (uint64_t)mask);
        int keep = -1;
        while (table[slot].hash != 0) {
            if (table[slot].hash == h) {
                // Duplicate: overwrite the survivor if this one is better.
                int j = table[slot].idx;
                if (beam[i].score > beam[j].score) {
                    beam[j] = beam[i];
                }
                keep = -2;
                break;
            }
            slot = (slot + 1) & mask;
        }
        if (keep == -1) {
            if (w != i) beam[w] = beam[i];
            table[slot].hash = h;
            table[slot].idx = w;
            w++;
        }
    }
    return w;
}

// ============================================================
// Globals (static to this translation unit)
// ============================================================

static PieceDef B2B_PIECES[8];
static bool b2b_initialized = false;

// Last search result — read by the C game loop to avoid lossy action_idx decode.
// b2b_search_c() writes this; b2b_run_eval_games() reads it.
static Placement b2b_last_placement;

// Kick tables: [from_rot][to_rot][kick_index][0=row, 1=col]
static int8_t B2B_KICKS[4][4][5][2];
static int8_t B2B_I_KICKS[4][4][5][2];

// Heuristic weights — hand-designed, bounded, survival-first.
// Rationale is documented at each use site in evaluate_state().
//
// Scale discipline:
//   - Instant-death: -1e6 (inviolable).
//   - Near-death cliff: -5000..-10000 (dominates any achievable positive reward).
//   - Max achievable positive reward in the reachable regime ≈ 160.
//   - Hole penalties cap so the bot cannot suicide to "avoid fixing" a bad board.
//   - B2B value is sublinear (sqrt) so surge payoff wins at high chains,
//     while holding wins at low chains.
static const float W_NEAR_DEATH      = 5000.0f;  // per row of slack inside last-2 zone
static const float W_HEIGHT_QUARTIC  = 80.0f;    // -W * h_ratio^4
static const float W_AVG_HEIGHT      = 10.0f;     // -W * avg_height.  Linear penalty on total stack volume
                                                   // (cells occupied per column, averaged).  Encourages
                                                   // "board emptiness" — a clean board with b2b=N beats a
                                                   // messy board with b2b=N because future options are
                                                   // preserved.  Pairs with b2b-reward terms to reward
                                                   // b2b-per-piece efficiency: hoarding spin slots without
                                                   // cashing them in is explicitly penalized via the cells
                                                   // those slots consume.  At avg_height=10 the penalty is
                                                   // -30 — big enough to discourage unnecessary upstacking
                                                   // but well below the survival wall and raw b2b rewards.
static const float W_BUMPINESS       = 1.0f;

static const float W_HOLES           = 6.0f;     // * min(holes, 8) * (1 + 0.5h)
static const int   HOLES_CAP         = 8;
static const float W_WASTED_HOLE     = 3.0f;     // * wasted_holes * (1 + 0.5h)
static const float W_HOLE_CEILING    = 1.5f;     // * hole_ceiling_weight (unused before this rework)
static const float W_HOLE_FORGIVE    = 1.5f;     // * min(setups, holes) * (1 + 0.5h)

static const float W_B2B_FLAT        = 5.0f;     // one-shot "b2b active" flag
static const float W_B2B_SQRT        = 8.0f;     // * sqrt(b2b) — sublinear store of potential
static const float W_B2B_LINEAR      = 25.0f;    // * b2b — linear holding reward that dominates surge break.
                                                   // Breaking at b2b=n surges ~n attack, contributing roughly
                                                   // (W_ATTACK_TOTAL + W_MAX_SINGLE + W_APP/pieces) * n ≈ 9.3n
                                                   // to the leaf score.  Setting W_B2B_LINEAR well above 9.3 makes
                                                   // holding one more piece of b2b strictly preferable to breaking
                                                   // for spike value alone; breaks occur only when the survival
                                                   // cliff (-5000 near-death) or smooth height quartic forces them.
                                                   // Empirically 25 balances no-garbage hold (max_b2b ≈ 20, limited
                                                   // by the 7-ply search horizon's ability to sustain spin chains)
                                                   // with garbage survival (≥80% at 0.15 chance).  Higher values
                                                   // (e.g. 50) push the bot into suicidal b2b-hoarding under garbage
                                                   // pressure and drop survival below 80%.
static const float W_ATTACK_TOTAL    = 1.2f;     // * total_attack in path
static const float W_MAX_SINGLE      = 0.5f;     // * max_single_attack (real spike signal).
                                                   // Down-tuned from 1.5: max_single_attack only peaks on a surge
                                                   // break, so this weight encodes "break for a big hit" — which
                                                   // conflicts with hold-indefinitely.  Kept at 0.5 as a tiebreaker
                                                   // so that IF a break happens (survival-forced in garbage), the
                                                   // bot still prefers the path that concentrates damage.
static const float W_B2B_ATTACK      = 1.5f;     // * b2b_attack — rewards damage dealt WHILE b2b is alive.
                                                   // Up-tuned from 0.4: this is the user's target efficiency
                                                   // metric — attack-per-clear and b2b-per-clear efficient
                                                   // damage comes from b2b-maintained attacks (TSD/TST, all-mini
                                                   // doubles, etc.), NOT from surge bursts.
static const float W_APP             = 3.0f;     // * (total_attack / pieces_placed) — direct APP optimization.
                                                   // Down-tuned from 6: APP per leaf is dominated by surge bursts
                                                   // which only happen on break, so a large W_APP subtly pushes
                                                   // toward breaking.  Halved to keep the tiebreaker value without
                                                   // steering the eval away from the hold strategy.

static const float W_COMBO           = 2.5f;     // * min(combo, 6)
static const int   COMBO_CAP         = 6;
static const float W_DOWNSTACK       = 2.0f;     // * min(accessible_9_rows, 4)
static const int   DOWNSTACK_CAP     = 4;
static const float W_WELL_ALIGNED_9  = 1.5f;     // * min(well_aligned_9, 3)
static const int   WELL_ALIGNED_9_CAP = 3;
static const float W_CASCADE         = 5.0f;     // * min(cascade_depth, 4) — stacked 9-rows
static const int   CASCADE_CAP       = 4;
static const float W_SURGE_POT       = 2.0f;     // * primed_multiplier * min(b2b, 20) — latent spike value
static const float W_BREAK_READY     = 6.0f;     // one-shot: b2b>=8 AND cascade_depth>=2

static const float W_TSLOT           = 6.0f;
static const float W_IMMOBILE_CLEAR  = 5.0f;     // * sqrt(immobile_clearing_placements)
static const float W_IMMOBILE_LINES  = 1.0f;     // * min(immobile_clearable_lines, 8)
static const int   CHAIN_ROLLOUT_K   = 5;        // Number of upcoming pieces the chain rollout simulates.
                                                   // Each unique leaf state runs one rollout; cost is bounded
                                                   // by K * (rotations * cols * scan_rows * lock+clear cost).
                                                   // K=5 extends horizon beyond the default depth-7 beam at
                                                   // negligible cost because the TT cache eliminates repeats.
                                                   // K=8 was empirically slower near tall boards; K=5 retains
                                                   // most of the b2b-chain signal while keeping per-leaf cost
                                                   // bounded when the board is near the ceiling.
static const float W_CHAIN_ROLLOUT   = 10.0f;    // * chain_rollout_length (counted spin-clears in rollout).
                                                   // Per-piece reward for a simulated b2b-growing continuation.
                                                   // A full-length chain (K=8 spin-clears) contributes +80 to
                                                   // the leaf — comparable to roughly 3 b2b levels via the
                                                   // W_B2B_LINEAR store, so it dominantly steers toward
                                                   // sustainable hold states without overriding survival gates.
                                                   // Tuned empirically: W=5 under-powered (max_b2b ~21),
                                                   // W=10 peaks around max_b2b ~26, W=15 degrades back down.
static const float W_FUTURE_B2B      = 2.0f;     // * future_immobile_clearing (queue-capped, next 3 pieces).
                                                   // Narrower-horizon proxy for "can b2b be sustained with the
                                                   // pieces about to arrive".  The 7-ply search often can't ITSELF
                                                   // find the sustaining placement, so it leaves a state that LOOKS
                                                   // undifferentiated from a break — this term gives the hold path
                                                   // an extra ~4 per sustainable near-term slot, pushing the eval
                                                   // over the break threshold when sustainability exists.
static const float W_TSPIN_MULTILINE = 3.0f;     // * min(t_multiline_setups, t_queue_count).
                                                   // Fires only when a T-slot clears >=2 lines (TSD/TST) AND a T
                                                   // piece is available in the queue.  This is the attack-per-clear
                                                   // efficient structure the user wants prioritized: TSD gives 4
                                                   // damage / 1 T + 3 setup pieces ≈ APP 1.0, TST gives 6/5 ≈ 1.2
                                                   // — both dominate Tetrises (4/5=0.8) in upstacking play.  No
                                                   // analogous bonus for I-tetrises per user direction.

// ============================================================
// Piece / Kick Initialization (copied from pathfinder.c)
// ============================================================

static void b2b_init_pieces(void) {
    if (b2b_initialized) return;

    memset(B2B_PIECES, 0, sizeof(B2B_PIECES));
    memset(B2B_KICKS, 0, sizeof(B2B_KICKS));
    memset(B2B_I_KICKS, 0, sizeof(B2B_I_KICKS));

    // I Piece
    B2B_PIECES[PIECE_I].orientations[0] = (PieceOrientation){ .row_masks={0, 15, 0, 0}, .min_col=0, .max_col=3, .min_row=1, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_I].orientations[1] = (PieceOrientation){ .row_masks={4, 4, 4, 4}, .min_col=2, .max_col=2, .min_row=0, .max_row=3, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_I].orientations[2] = (PieceOrientation){ .row_masks={0, 0, 15, 0}, .min_col=0, .max_col=3, .min_row=2, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_I].orientations[3] = (PieceOrientation){ .row_masks={2, 2, 2, 2}, .min_col=1, .max_col=1, .min_row=0, .max_row=3, .row_offsets={0,1,2,3} };

    // J Piece
    B2B_PIECES[PIECE_J].orientations[0] = (PieceOrientation){ .row_masks={1, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_J].orientations[1] = (PieceOrientation){ .row_masks={6, 2, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_J].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 4, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_J].orientations[3] = (PieceOrientation){ .row_masks={2, 2, 3, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // L Piece
    B2B_PIECES[PIECE_L].orientations[0] = (PieceOrientation){ .row_masks={4, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_L].orientations[1] = (PieceOrientation){ .row_masks={2, 2, 6, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_L].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 1, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_L].orientations[3] = (PieceOrientation){ .row_masks={3, 2, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // O Piece
    PieceOrientation o_orient = (PieceOrientation){ .row_masks={6, 6, 0, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    for (int i = 0; i < 4; i++) B2B_PIECES[PIECE_O].orientations[i] = o_orient;

    // S Piece
    B2B_PIECES[PIECE_S].orientations[0] = (PieceOrientation){ .row_masks={6, 3, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_S].orientations[1] = (PieceOrientation){ .row_masks={2, 6, 4, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_S].orientations[2] = (PieceOrientation){ .row_masks={0, 6, 3, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_S].orientations[3] = (PieceOrientation){ .row_masks={1, 3, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // T Piece
    B2B_PIECES[PIECE_T].orientations[0] = (PieceOrientation){ .row_masks={2, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_T].orientations[1] = (PieceOrientation){ .row_masks={2, 6, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_T].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 2, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_T].orientations[3] = (PieceOrientation){ .row_masks={2, 3, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // Z Piece
    B2B_PIECES[PIECE_Z].orientations[0] = (PieceOrientation){ .row_masks={3, 6, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_Z].orientations[1] = (PieceOrientation){ .row_masks={4, 6, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_Z].orientations[2] = (PieceOrientation){ .row_masks={0, 3, 6, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    B2B_PIECES[PIECE_Z].orientations[3] = (PieceOrientation){ .row_masks={2, 3, 1, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // --- Standard Kicks ---
    int8_t k01[4][2] = {{0,-1}, {-1,-1}, {2,0}, {2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[0][1][i][0] = k01[i][0]; B2B_KICKS[0][1][i][1] = k01[i][1]; }

    int8_t k03[4][2] = {{0,1}, {-1,1}, {2,0}, {2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[0][3][i][0] = k03[i][0]; B2B_KICKS[0][3][i][1] = k03[i][1]; }

    int8_t k10[4][2] = {{0,1}, {1,1}, {-2,0}, {-2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[1][0][i][0] = k10[i][0]; B2B_KICKS[1][0][i][1] = k10[i][1]; }

    int8_t k12[4][2] = {{0,1}, {1,1}, {-2,0}, {-2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[1][2][i][0] = k12[i][0]; B2B_KICKS[1][2][i][1] = k12[i][1]; }

    int8_t k21[4][2] = {{0,-1}, {-1,-1}, {2,0}, {2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[2][1][i][0] = k21[i][0]; B2B_KICKS[2][1][i][1] = k21[i][1]; }

    int8_t k23[4][2] = {{0,1}, {-1,1}, {2,0}, {2,1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[2][3][i][0] = k23[i][0]; B2B_KICKS[2][3][i][1] = k23[i][1]; }

    int8_t k30[4][2] = {{0,-1}, {1,-1}, {-2,0}, {-2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[3][0][i][0] = k30[i][0]; B2B_KICKS[3][0][i][1] = k30[i][1]; }

    int8_t k32[4][2] = {{0,-1}, {1,-1}, {-2,0}, {-2,-1}};
    for (int i = 0; i < 4; i++) { B2B_KICKS[3][2][i][0] = k32[i][0]; B2B_KICKS[3][2][i][1] = k32[i][1]; }

    // 180 Standard Kicks
    int8_t k02[5][2] = {{-1,0}, {-1,1}, {-1,-1}, {0,1}, {0,-1}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[0][2][i][0] = k02[i][0]; B2B_KICKS[0][2][i][1] = k02[i][1]; }

    int8_t k13[5][2] = {{0,1}, {-2,1}, {-1,1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[1][3][i][0] = k13[i][0]; B2B_KICKS[1][3][i][1] = k13[i][1]; }

    int8_t k20[5][2] = {{1,0}, {1,-1}, {1,1}, {0,-1}, {0,1}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[2][0][i][0] = k20[i][0]; B2B_KICKS[2][0][i][1] = k20[i][1]; }

    int8_t k31[5][2] = {{0,-1}, {-2,-1}, {-1,-1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_KICKS[3][1][i][0] = k31[i][0]; B2B_KICKS[3][1][i][1] = k31[i][1]; }

    // --- I Kicks ---
    int8_t ik01[4][2] = {{0,1}, {0,-2}, {1,-2}, {-2,1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[0][1][i][0] = ik01[i][0]; B2B_I_KICKS[0][1][i][1] = ik01[i][1]; }

    int8_t ik03[4][2] = {{0,-1}, {0,2}, {1,2}, {-2,-1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[0][3][i][0] = ik03[i][0]; B2B_I_KICKS[0][3][i][1] = ik03[i][1]; }

    int8_t ik10[4][2] = {{0,-1}, {0,2}, {2,-1}, {-1,2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[1][0][i][0] = ik10[i][0]; B2B_I_KICKS[1][0][i][1] = ik10[i][1]; }

    int8_t ik12[4][2] = {{0,-1}, {0,2}, {-2,-1}, {1,2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[1][2][i][0] = ik12[i][0]; B2B_I_KICKS[1][2][i][1] = ik12[i][1]; }

    int8_t ik21[4][2] = {{0,-2}, {0,1}, {-1,-2}, {2,1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[2][1][i][0] = ik21[i][0]; B2B_I_KICKS[2][1][i][1] = ik21[i][1]; }

    int8_t ik23[4][2] = {{0,2}, {0,-1}, {-1,2}, {2,-1}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[2][3][i][0] = ik23[i][0]; B2B_I_KICKS[2][3][i][1] = ik23[i][1]; }

    int8_t ik30[4][2] = {{0,1}, {0,-2}, {2,1}, {-1,-2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[3][0][i][0] = ik30[i][0]; B2B_I_KICKS[3][0][i][1] = ik30[i][1]; }

    int8_t ik32[4][2] = {{0,1}, {0,-2}, {-2,1}, {1,-2}};
    for (int i = 0; i < 4; i++) { B2B_I_KICKS[3][2][i][0] = ik32[i][0]; B2B_I_KICKS[3][2][i][1] = ik32[i][1]; }

    // 180 I Kicks
    int8_t ik02[5][2] = {{-1,0}, {-1,1}, {-1,-1}, {0,1}, {0,-1}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[0][2][i][0] = ik02[i][0]; B2B_I_KICKS[0][2][i][1] = ik02[i][1]; }

    int8_t ik13[5][2] = {{0,1}, {-2,1}, {-1,1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[1][3][i][0] = ik13[i][0]; B2B_I_KICKS[1][3][i][1] = ik13[i][1]; }

    int8_t ik20[5][2] = {{1,0}, {1,-1}, {1,1}, {0,-1}, {0,1}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[2][0][i][0] = ik20[i][0]; B2B_I_KICKS[2][0][i][1] = ik20[i][1]; }

    int8_t ik31[5][2] = {{0,-1}, {-2,-1}, {-1,-1}, {-2,0}, {-1,0}};
    for (int i = 0; i < 5; i++) { B2B_I_KICKS[3][1][i][0] = ik31[i][0]; B2B_I_KICKS[3][1][i][1] = ik31[i][1]; }

    b2b_initialized = true;
}

// ============================================================
// Collision / Movement Helpers (adapted from pathfinder.c)
// ============================================================

static int b2b_check_collision(const uint16_t* board_rows, int board_height,
                               int piece_type, int rot, int r, int c) {
    PieceOrientation* ori = &B2B_PIECES[piece_type].orientations[rot];

    if (c + ori->min_col < 0 || c + ori->max_col >= BOARD_COLS) return 1;
    if (r + ori->min_row < 0) return 1;
    if (r + ori->max_row >= board_height) return 1;

    for (int i = 0; i < 4; i++) {
        int board_row = r + i;
        if (board_row < 0 || board_row >= board_height) continue;
        uint16_t mask = ori->row_masks[i];
        uint16_t shifted = (c >= 0) ? (mask << c) : (mask >> (-c));
        if (board_rows[board_row] & shifted) return 1;
    }
    return 0;
}

static int b2b_hard_drop_row(const uint16_t* board_rows, int board_height,
                             int piece_type, int rot, int r, int c) {
    int curr = r;
    while (!b2b_check_collision(board_rows, board_height, piece_type, rot, curr + 1, c)) {
        curr++;
    }
    return curr;
}

static int b2b_encode_state(int r, int c, int rot, int piece_type) {
    int min_col = B2B_PIECES[piece_type].orientations[rot].min_col;
    int norm_col = c + min_col;
    if (norm_col < 0 || norm_col >= BOARD_COLS) return -1;
    if (r < 0 || r >= BOARD_ROWS) return -1;
    return ((r * BOARD_COLS) + norm_col) * 4 + rot;
}

static void b2b_decode_state(int state, int* r, int* c, int* rot, int piece_type) {
    *rot = state % 4;
    int base = state / 4;
    int norm_col = base % BOARD_COLS;
    *r = base / BOARD_COLS;
    int min_col = B2B_PIECES[piece_type].orientations[*rot].min_col;
    *c = norm_col - min_col;
}

// Incrementally update `out_heights` from a known parent-state's `parent_heights`
// by inspecting only the ≤4 columns touched by the placed piece.  Caller MUST
// only use this when no lines cleared and no garbage was pushed on this step —
// otherwise parent heights are no longer valid and a full rescan is required.
static inline void patch_col_heights_after_place(const int8_t* parent_heights,
                                                 int piece, int rot,
                                                 int landing_row, int col,
                                                 int board_height,
                                                 int8_t* out_heights) {
    memcpy(out_heights, parent_heights, sizeof(int8_t) * BOARD_COLS);
    const PieceOrientation* o = &B2B_PIECES[piece].orientations[rot];
    for (int dr = 0; dr < 4; dr++) {
        uint16_t m = o->row_masks[dr];
        if (!m) continue;
        int abs_r = landing_row + dr;
        int col_height = board_height - abs_r;
        while (m) {
            int dc = __builtin_ctz(m);
            int c = col + dc;
            if (c >= 0 && c < BOARD_COLS && col_height > out_heights[c]) {
                out_heights[c] = (int8_t)col_height;
            }
            m &= (uint16_t)(m - 1);
        }
    }
}

// ============================================================
// Spin Detection (adapted from pathfinder.c + Scorer.py)
// ============================================================

// Returns detailed T-spin type: SPIN_NONE, SPIN_T_MINI, or SPIN_T_FULL
static int detect_t_spin(const uint16_t* board, int board_height,
                         int r, int c, int rot, int delta_loc_sum) {
    // 3-corner rule from Scorer.py
    int corners[4][2] = {{0,0}, {0,2}, {2,2}, {2,0}}; // TL, TR, BR, BL
    bool filled[4];

    for (int i = 0; i < 4; i++) {
        int cr = r + corners[i][0];
        int cc = c + corners[i][1];
        if (cr >= board_height || cc < 0 || cc >= BOARD_COLS || cr < 0) {
            filled[i] = true;
        } else {
            filled[i] = (board[cr] & (1 << cc)) != 0;
        }
    }

    int total_filled = 0;
    for (int i = 0; i < 4; i++) if (filled[i]) total_filled++;
    if (total_filled < 3) return SPIN_NONE;

    // Determine front/back corners based on rotation
    // Scorer.py: back cell = the cell NOT in the piece's cells for that rotation
    // Rot 0: back=row2,col1 -> back_idx=3 (corner indices BL=3, TL=0 are back-side)
    // Rot 1: back=row1,col0 -> back_idx=0 (TL=0, BL=3 are back-side)
    // Rot 2: back=row0,col1 -> back_idx=0 (TL=0, TR=1 are back-side)
    // Rot 3: back=row1,col2 -> back_idx=1 (TR=1, BR=2 are back-side)

    // Actually: from Scorer.py, the "back" cell is the one missing from the T shape.
    // T orientations (cells in 3x3):
    // Rot 0: [0,1],[1,0],[1,1],[1,2] -> missing edge cell is [2,1] -> back direction is down
    //   Back corners: BL(2,0)=idx3, BR(2,2)=idx2. Front: TL(0,0)=idx0, TR(0,2)=idx1
    // Rot 1: [0,1],[1,1],[1,2],[2,1] -> missing [1,0] -> back is left
    //   Back corners: TL(0,0)=idx0, BL(2,0)=idx3. Front: TR(0,2)=idx1, BR(2,2)=idx2
    // Rot 2: [1,0],[1,1],[1,2],[2,1] -> missing [0,1] -> back is up
    //   Back corners: TL(0,0)=idx0, TR(0,2)=idx1. Front: BL(2,0)=idx3, BR(2,2)=idx2
    // Rot 3: [0,1],[1,0],[1,1],[2,1] -> missing [1,2] -> back is right
    //   Back corners: TR(0,2)=idx1, BR(2,2)=idx2. Front: TL(0,0)=idx0, BL(2,0)=idx3

    int front_filled = 0, back_filled = 0;
    switch (rot) {
        case 0: // front: TL(0), TR(1); back: BR(2), BL(3)
            front_filled = filled[0] + filled[1];
            back_filled = filled[2] + filled[3];
            break;
        case 1: // front: TR(1), BR(2); back: TL(0), BL(3)
            front_filled = filled[1] + filled[2];
            back_filled = filled[0] + filled[3];
            break;
        case 2: // front: BR(2), BL(3); back: TL(0), TR(1)
            front_filled = filled[2] + filled[3];
            back_filled = filled[0] + filled[1];
            break;
        case 3: // front: TL(0), BL(3); back: TR(1), BR(2)
            front_filled = filled[0] + filled[3];
            back_filled = filled[1] + filled[2];
            break;
    }

    if (front_filled == 2 && back_filled >= 1) {
        return SPIN_T_FULL;
    } else if (front_filled == 1 && back_filled == 2) {
        if (delta_loc_sum > 2) {
            return SPIN_T_FULL; // Kicked far enough
        } else {
            return SPIN_T_MINI;
        }
    }
    return SPIN_NONE;
}

// Check immobility for non-T pieces (ALL_MINI detection)
static bool b2b_check_immobility(const uint16_t* board, int board_height,
                                 int piece_type, int rot, int r, int c) {
    int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    for (int i = 0; i < 4; i++) {
        if (!b2b_check_collision(board, board_height, piece_type, rot,
                                 r + dirs[i][0], c + dirs[i][1])) {
            return false;
        }
    }
    return true;
}

// ============================================================
// Board Simulation
// ============================================================

// Lock a piece onto the board bitmask array
static void lock_piece_on_board(uint16_t* board, int board_height,
                                int piece_type, int rot, int r, int c) {
    PieceOrientation* ori = &B2B_PIECES[piece_type].orientations[rot];
    for (int i = 0; i < 4; i++) {
        int board_row = r + i;
        if (board_row < 0 || board_row >= board_height) continue;
        uint16_t mask = ori->row_masks[i];
        if (mask == 0) continue;
        uint16_t shifted = (c >= 0) ? (mask << c) : (mask >> (-c));
        board[board_row] |= shifted;
    }
}

// Clear full lines, shift rows down. Returns number of lines cleared.
// Rows with GARB_ROW_MARKER set are simulated garbage and are never cleared
// even if all 10 playfield bits are filled.
static int clear_lines(uint16_t* board, int board_height) {
    uint16_t full_row = (1 << BOARD_COLS) - 1; // 0x3FF for 10 cols
    int clears = 0;

    // Iterate bottom-up, shift down when clearing
    int write = board_height - 1;
    for (int read = board_height - 1; read >= 0; read--) {
        if ((board[read] & full_row) == full_row &&
            !(board[read] & GARB_ROW_MARKER)) {
            clears++;
        } else {
            board[write] = board[read];
            write--;
        }
    }
    // Fill remaining top rows with empty
    for (int i = write; i >= 0; i--) {
        board[i] = 0;
    }
    return clears;
}

// Push simulated garbage rows onto the board during search.
// Rows are fully filled (all 10 bits) + GARB_ROW_MARKER so they act as
// unclearable occupied space. Everything above shifts up.
// Returns the number of rows actually pushed (may be less if board would
// overflow and cause instant death).
static int push_simulated_garbage(uint16_t* board, int board_height, int rows) {
    if (rows <= 0) return 0;

    uint16_t garb_row = ((1 << BOARD_COLS) - 1) | GARB_ROW_MARKER;

    // Shift existing rows up by 'rows' positions
    for (int r = 0; r < board_height - rows; r++) {
        board[r] = board[r + rows];
    }
    // Fill bottom 'rows' with garbage
    for (int r = board_height - rows; r < board_height; r++) {
        board[r] = garb_row;
    }
    return rows;
}

// Attack calculation — exact replica of Scorer.py
typedef struct {
    float attack;
    int new_b2b;
    int new_combo;
    bool b2b_broken;
    bool b2b_maintaining;  // true if this clear kept/started b2b (spin/tetris/PC)
    float surge;
} AttackResult;

static AttackResult compute_attack(int clears, int spin_type, int b2b, int combo,
                                   bool perfect_clear) {
    AttackResult res;
    res.attack = 0;
    res.new_b2b = b2b;
    res.new_combo = combo;
    res.b2b_broken = false;
    res.b2b_maintaining = false;
    res.surge = 0;

    if (clears > 0) {
        // B2B tracking
        if (spin_type != SPIN_NONE || clears == 4 || perfect_clear) {
            res.new_b2b = b2b + 1;
            res.b2b_maintaining = true;
        } else {
            // Breaking b2b
            if (b2b >= 4) {
                res.surge = (float)(b2b);
            }
            res.new_b2b = -1;
            if (b2b >= 0) res.b2b_broken = true;
        }

        res.new_combo = combo + 1;

        // Base attack
        if (perfect_clear) {
            int pc_table[5] = {0, 5, 6, 7, 9};
            res.attack += pc_table[clears < 5 ? clears : 4];
        } else if (spin_type == SPIN_T_FULL) {
            int ts_table[5] = {0, 2, 4, 6, 0};
            res.attack += ts_table[clears < 5 ? clears : 4];
        } else if (spin_type == SPIN_T_MINI) {
            int tm_table[5] = {0, 0, 1, 2, 0};
            res.attack += tm_table[clears < 5 ? clears : 4];
        } else {
            int no_table[5] = {0, 0, 1, 2, 4};
            res.attack += no_table[clears < 5 ? clears : 4];
        }

        // B2B bonus (applied if b2b was > -1 BEFORE this clear)
        if (b2b > -1) {
            res.attack += 1;
        }

        // Combo multiplier
        if (combo > 0) {
            if (res.attack > 0) {
                res.attack = floorf(res.attack * (1.0f + 0.25f * combo));
            } else {
                res.attack = floorf(logf(1.0f + 1.25f * combo));
            }
        }

        // Surge
        res.attack += res.surge;
    } else {
        res.new_combo = -1;
    }

    return res;
}

// ============================================================
// Internal Placement Finder (simplified BFS)
// Returns all unique (rot, col, landing_row, spin) placements.
// For depth 0, also stores BFS state for key sequence reconstruction.
// ============================================================

static int find_placements(const uint16_t* board_rows, int board_height,
                           int piece_type, Placement* out, int max_out,
                           BFSStateMeta* meta_out) {
    // meta_out may be NULL if we don't need sequence reconstruction (depth > 0)
    static BFSStateMeta meta[BFS_STATE_SPACE];
    static bool visited[BFS_STATE_SPACE];
    static int queue[BFS_QUEUE_CAPACITY];

    for (int i = 0; i < BFS_STATE_SPACE; i++) {
        visited[i] = false;
        meta[i].parent = -1;
    }

    // Spawn position: row=0, col=3, rot=0
    int start_r = 0, start_c = 3, start_rot = 0;
    int start_state = b2b_encode_state(start_r, start_c, start_rot, piece_type);

    if (start_state == -1 ||
        b2b_check_collision(board_rows, board_height, piece_type, start_rot, start_r, start_c)) {
        return 0; // Can't spawn — board is topped out
    }

    int head = 0, tail = 0;
    queue[tail++] = start_state;
    visited[start_state] = true;
    meta[start_state].depth = 0;
    meta[start_state].last_move = KEY_START;
    meta[start_state].delta_r = 0;
    meta[start_state].delta_row = 0;
    meta[start_state].delta_col = 0;

    int num_placements = 0;
    int visible_start = board_height - VISIBLE_ROWS;

    while (head != tail) {
        int curr_state = queue[head++];
        head %= BFS_QUEUE_CAPACITY;

        int r, c, rot;
        b2b_decode_state(curr_state, &r, &c, &rot, piece_type);
        int depth = meta[curr_state].depth;

        // Only emit a placement when this BFS state IS the landing state
        // (the piece can no longer fall from here). Non-landing states will
        // reach their landing via SOFT_DROP, which sets delta_r=0 — matching
        // PyTetrisEnv._move clearing piece.delta_r whenever delta_loc[0]!=0.
        // Rotation-with-kick that directly lands the piece preserves delta_r,
        // correctly identifying canonical T-spins.
        int land_r = b2b_hard_drop_row(board_rows, board_height, piece_type, rot, r, c);

        if (r == land_r && land_r >= visible_start && num_placements < max_out) {
            int spin = SPIN_NONE;
            int delta_r_val = meta[curr_state].delta_r;
            int dloc_sum = abs(meta[curr_state].delta_row) + abs(meta[curr_state].delta_col);

            if (delta_r_val != 0) {
                if (piece_type == PIECE_T) {
                    spin = detect_t_spin(board_rows, board_height, land_r, c, rot, dloc_sum);
                } else {
                    if (b2b_check_immobility(board_rows, board_height, piece_type, rot, land_r, c)) {
                        spin = SPIN_ALL_MINI;
                    }
                }
            }

            // Check for duplicate placement (same rot, col, landing_row, spin)
            bool dup = false;
            for (int i = 0; i < num_placements; i++) {
                if (out[i].rot == rot && out[i].col == c &&
                    out[i].landing_row == land_r && out[i].spin_type == spin) {
                    dup = true;
                    break;
                }
            }

            if (!dup) {
                out[num_placements].rot = rot;
                out[num_placements].col = c;
                out[num_placements].landing_row = land_r;
                out[num_placements].spin_type = spin;
                out[num_placements].delta_r = delta_r_val;
                out[num_placements].delta_loc_sum = dloc_sum;
                out[num_placements].bfs_state = curr_state;
                num_placements++;
            }
        }

        // BFS depth limit — keep paths short enough for max_len=15 sequences
        // START + (opt HOLD) + path + HARD_DROP <= 15, so path <= 12
        if (depth >= 12) continue;

        // Try all 8 moves
        int moves[] = {KEY_TAP_LEFT, KEY_TAP_RIGHT, KEY_DAS_LEFT, KEY_DAS_RIGHT,
                       KEY_CLOCKWISE, KEY_ANTICLOCKWISE, KEY_ROTATE_180, KEY_SOFT_DROP};

        for (int m = 0; m < 8; m++) {
            int key = moves[m];
            int nr = r, nc = c, nrot = rot;
            int dr = 0, drow = 0, dcol = 0;
            bool valid = false;

            if (key == KEY_TAP_LEFT) {
                if (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, c - 1)) {
                    nc--; valid = true; dcol = -1;
                }
            } else if (key == KEY_TAP_RIGHT) {
                if (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, c + 1)) {
                    nc++; valid = true; dcol = 1;
                }
            } else if (key == KEY_DAS_LEFT) {
                int tmp = c;
                while (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, tmp - 1)) tmp--;
                if (tmp != c) { nc = tmp; valid = true; dcol = nc - c; }
            } else if (key == KEY_DAS_RIGHT) {
                int tmp = c;
                while (!b2b_check_collision(board_rows, board_height, piece_type, rot, r, tmp + 1)) tmp++;
                if (tmp != c) { nc = tmp; valid = true; dcol = nc - c; }
            } else if (key == KEY_SOFT_DROP) {
                int tmp = r;
                int max_row = B2B_PIECES[piece_type].orientations[rot].max_row;
                while (!b2b_check_collision(board_rows, board_height, piece_type, rot, tmp + 1, c)) {
                    tmp++;
                    if (tmp + max_row >= board_height - 1) break;
                }
                if (tmp != r) { nr = tmp; valid = true; drow = nr - r; }
            } else {
                // Rotation
                int delta = 0;
                if (key == KEY_CLOCKWISE) delta = 1;
                else if (key == KEY_ANTICLOCKWISE) delta = 3;
                else delta = 2; // ROTATE_180

                int next_rot = (rot + delta) % 4;

                if (!b2b_check_collision(board_rows, board_height, piece_type, next_rot, r, c)) {
                    nrot = next_rot; valid = true;
                    dr = (delta == 3) ? -1 : delta;
                } else {
                    int8_t (*table)[2];
                    if (piece_type == PIECE_I) table = B2B_I_KICKS[rot][next_rot];
                    else table = B2B_KICKS[rot][next_rot];

                    int count = (key == KEY_ROTATE_180) ? 5 : 4;

                    for (int k = 0; k < count; k++) {
                        int kdr = table[k][0];
                        int kdc = table[k][1];
                        if (kdr == 0 && kdc == 0 && count == 5) continue;

                        if (!b2b_check_collision(board_rows, board_height, piece_type,
                                                 next_rot, r + kdr, c + kdc)) {
                            nr = r + kdr; nc = c + kdc; nrot = next_rot;
                            valid = true;
                            dr = (delta == 3) ? -1 : delta;
                            drow = kdr; dcol = kdc;
                            break;
                        }
                    }
                }
            }

            if (valid) {
                int next_s = b2b_encode_state(nr, nc, nrot, piece_type);
                if (next_s != -1 && !visited[next_s]) {
                    visited[next_s] = true;
                    meta[next_s].parent = curr_state;
                    meta[next_s].last_move = key;
                    meta[next_s].depth = depth + 1;
                    meta[next_s].delta_r = dr;
                    meta[next_s].delta_row = drow;
                    meta[next_s].delta_col = dcol;

                    queue[tail++] = next_s;
                    tail %= BFS_QUEUE_CAPACITY;
                }
            }
        }
    }

    // Copy BFS meta if caller wants it (for key sequence reconstruction)
    if (meta_out != NULL) {
        memcpy(meta_out, meta, sizeof(meta));
    }

    return num_placements;
}

// ============================================================
// Heuristic Evaluation
// ============================================================

// Flood-fill reachability from top of board.
// Fills reachable[] with bitmasks indicating which empty cells are reachable
// from the top row via orthogonal movement through empty cells.
static void compute_reachability(const uint16_t* board, int board_height,
                                  uint16_t* reachable) {
    memset(reachable, 0, sizeof(uint16_t) * board_height);

    static int flood_queue[BOARD_ROWS * BOARD_COLS * 2]; // row, col pairs
    int fh = 0, ft = 0;

    for (int c = 0; c < BOARD_COLS; c++) {
        if (!(board[0] & (1 << c))) {
            reachable[0] |= (1 << c);
            flood_queue[ft++] = 0;
            flood_queue[ft++] = c;
        }
    }

    while (fh < ft) {
        int r = flood_queue[fh++];
        int c = flood_queue[fh++];

        int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        for (int d = 0; d < 4; d++) {
            int nr = r + dirs[d][0];
            int nc = c + dirs[d][1];
            if (nr < 0 || nr >= board_height || nc < 0 || nc >= BOARD_COLS) continue;
            uint16_t bit = (1 << nc);
            if ((board[nr] & bit) || (reachable[nr] & bit)) continue;
            reachable[nr] |= bit;
            flood_queue[ft++] = nr;
            flood_queue[ft++] = nc;
        }
    }
}

// Count enclosed hole SECTIONS (connected components) using precomputed
// reachability.  A "section" is a maximal 4-connected group of enclosed
// empty cells.  Cells in immobile_cells[] are excluded.
//
// Fewer, larger cavities score lower than many scattered single-cell holes.
static int count_hole_sections(const uint16_t* board,
                               int board_height,
                               const uint16_t* reachable,
                               const uint16_t* immobile_cells) {
    uint16_t full_mask = (1 << BOARD_COLS) - 1;

    // Build per-row bitmask of hole cells
    uint16_t hole_mask[BOARD_ROWS];
    for (int r = 0; r < board_height; r++) {
        uint16_t empty = (~board[r]) & full_mask;
        uint16_t enclosed = empty & (~reachable[r]);
        enclosed &= ~immobile_cells[r];
        hole_mask[r] = enclosed;
    }

    // Flood-fill BFS to count connected components (4-connected)
    uint16_t visited[BOARD_ROWS];
    memset(visited, 0, sizeof(uint16_t) * board_height);

    // Queue — worst case is every cell on the board
    int queue_r[BOARD_ROWS * BOARD_COLS];
    int queue_c[BOARD_ROWS * BOARD_COLS];

    int sections = 0;
    for (int r = 0; r < board_height; r++) {
        uint16_t remaining = hole_mask[r] & ~visited[r];
        while (remaining) {
            // Pick lowest-set-bit column
            int c = __builtin_ctz(remaining);

            // New connected component — flood fill from (r, c)
            sections++;
            int front = 0, back = 0;
            queue_r[back] = r;
            queue_c[back] = c;
            back++;
            visited[r] |= (uint16_t)(1 << c);

            while (front < back) {
                int cr = queue_r[front];
                int cc = queue_c[front];
                front++;

                // 4 neighbors: up, down, left, right
                const int dr[4] = {-1, 1, 0, 0};
                const int dc[4] = {0, 0, -1, 1};
                for (int d = 0; d < 4; d++) {
                    int nr = cr + dr[d];
                    int nc = cc + dc[d];
                    if (nr < 0 || nr >= board_height ||
                        nc < 0 || nc >= BOARD_COLS) continue;
                    uint16_t nbit = (uint16_t)(1 << nc);
                    if (!(hole_mask[nr] & nbit)) continue;
                    if (visited[nr] & nbit) continue;
                    visited[nr] |= nbit;
                    queue_r[back] = nr;
                    queue_c[back] = nc;
                    back++;
                }
            }

            remaining = hole_mask[r] & ~visited[r];
        }
    }

    return sections;
}

// Compute "hole ceiling weight": for each enclosed hole, count the number
// of filled cells above it in the same column, weighted by how high the
// hole is in the stack (higher holes = more urgent = higher weight).
//
// This penalizes upstacking over enclosed holes — each filled cell placed
// above an enclosed hole makes it harder to clear, and holes near the top
// of the stack are more dangerous.
//
// Returns a float score (not an integer count) because of height weighting.
static float compute_hole_ceiling_weight(const uint16_t* board, int board_height,
                                          const uint16_t* reachable,
                                          const uint16_t* immobile_cells) {
    float total_weight = 0.0f;

    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);

        // Scan column top-to-bottom, track filled cells above
        int filled_above = 0;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                filled_above++;
            } else {
                // Empty cell — check if it's an enclosed hole
                bool enclosed = !(reachable[r] & bit);
                bool is_setup = (immobile_cells[r] & bit) != 0;
                if (enclosed && !is_setup && filled_above > 0) {
                    int hole_height = board_height - r;
                    float height_factor = (float)hole_height / (float)board_height;
                    total_weight += (float)filled_above * (1.0f + height_factor);
                }
            }
        }
    }

    return total_weight;
}

// ============================================================
// Spin Setup Detection
// ============================================================

// Check if a cell is filled (or out of bounds = filled)
static inline bool cell_filled(const uint16_t* board, int board_height, int r, int c) {
    if (r < 0 || r >= board_height || c < 0 || c >= BOARD_COLS) return true;
    return (board[r] & (1 << c)) != 0;
}

static inline bool cell_empty(const uint16_t* board, int board_height, int r, int c) {
    if (r < 0 || r >= board_height || c < 0 || c >= BOARD_COLS) return false;
    return (board[r] & (1 << c)) == 0;
}

// Detect T-spin setups: look for T-shaped slots where a T piece could spin in.
// A T-slot is a pattern where:
//   - There's a T-shaped cavity (3 empty cells in T formation)
//   - At least 3 of the 4 corners around the T center are filled
//   - The T piece could actually reach and spin into this slot
//
// We check for all 4 T-spin orientations at each board position.
// Returns: number of T-spin setups found (0, 1, or more)
// t_slot_quality: set to best quality found (0=none, 1=mini possible, 2=full T-spin)
// t_multiline_setups: [optional, may be NULL] number of setups that would clear >=2 lines
//                     (TSD/TST); these are the attack/clear-efficient T-spins we want to
//                     actively reward when T is in the queue.
static int detect_t_spin_setups(const uint16_t* board, int board_height,
                                int* t_slot_quality,
                                int* t_multiline_setups) {
    int setups = 0;
    int best_quality = 0;
    int multi = 0;
    uint16_t full_mask_ = (1 << BOARD_COLS) - 1;

    // For each possible T-piece center position (row, col),
    // check if a T-spin could happen in each of the 4 rotations.
    //
    // T piece orientations (center at [1,1] in 3x3 grid):
    // Rot 0: [0,1] [1,0] [1,1] [1,2] -> points up, spins from above
    // Rot 1: [0,1] [1,1] [1,2] [2,1] -> points right, spins from right
    // Rot 2: [1,0] [1,1] [1,2] [2,1] -> points down, spins from below
    // Rot 3: [0,1] [1,0] [1,1] [2,1] -> points left, spins from left

    // We look for the T-slot pattern: the 3 cells of the T (excluding center-back)
    // must be empty, and at least 3 of 4 corners must be filled.

    for (int r = 0; r < board_height - 2; r++) {
        for (int c = 0; c < BOARD_COLS - 2; c++) {
            // Corner positions (in the 3x3 grid anchored at r,c)
            bool tl = cell_filled(board, board_height, r, c);
            bool tr = cell_filled(board, board_height, r, c + 2);
            bool bl = cell_filled(board, board_height, r + 2, c);
            bool br = cell_filled(board, board_height, r + 2, c + 2);
            int corner_count = tl + tr + bl + br;

            if (corner_count < 3) continue;

            // Check each T rotation for a valid slot

            // Helper to count lines cleared by a T placement at this slot:
            // given the 4 piece cells as (row, col_bit) pairs, OR each into
            // its board row and count how many become full.  Inlined per
            // rotation so we don't allocate arrays in a hot loop.
            #define TSPIN_COUNT_LINES(r0,m0, r1,m1, r2,m2, r3,m3) ({        \
                int _lines = 0;                                                \
                int _rs[4] = {(r0),(r1),(r2),(r3)};                            \
                uint16_t _ms[4] = {(uint16_t)(m0),(uint16_t)(m1),              \
                                    (uint16_t)(m2),(uint16_t)(m3)};            \
                /* dedupe rows: multiple T cells can be on same row */         \
                for (int _i = 0; _i < 4; _i++) {                               \
                    bool _seen = false;                                        \
                    for (int _j = 0; _j < _i; _j++)                            \
                        if (_rs[_j] == _rs[_i]) { _seen = true; break; }       \
                    if (_seen) continue;                                       \
                    uint16_t _comb = _ms[_i];                                  \
                    for (int _j = _i + 1; _j < 4; _j++)                        \
                        if (_rs[_j] == _rs[_i]) _comb |= _ms[_j];              \
                    if (_rs[_i] >= 0 && _rs[_i] < board_height &&              \
                        (((uint16_t)board[_rs[_i]] | _comb) & full_mask_) == full_mask_) \
                        _lines++;                                              \
                }                                                              \
                _lines;                                                        \
            })

            // Rot 2 (T points down, most common T-spin: overhang from above)
            // T cells: [1,0] [1,1] [1,2] [2,1] — center is [1,1]
            // Need these cells empty:
            if (cell_empty(board, board_height, r + 1, c) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 2) &&
                cell_empty(board, board_height, r + 2, c + 1)) {
                // Front corners for rot 2: BL and BR
                int front = bl + br;
                int back = tl + tr;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2; // Full T-spin
                else if (front == 1 && back == 2) quality = 1; // Mini possible
                if (quality > 0) {
                    // Check that the slot is accessible: the entry point [0,1] should
                    // have some path from above (at least the cell above should be empty)
                    if (cell_empty(board, board_height, r, c + 1)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                        if (quality == 2) {
                            int lines = TSPIN_COUNT_LINES(
                                r+1, 1u << c,
                                r+1, 1u << (c+1),
                                r+1, 1u << (c+2),
                                r+2, 1u << (c+1));
                            if (lines >= 2) multi++;
                        }
                    }
                }
            }

            // Rot 0 (T points up — less common, needs slot below)
            // T cells: [0,1] [1,0] [1,1] [1,2] — center is [1,1]
            if (cell_empty(board, board_height, r, c + 1) &&
                cell_empty(board, board_height, r + 1, c) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 2)) {
                int front = tl + tr;
                int back = bl + br;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2;
                else if (front == 1 && back == 2) quality = 1;
                if (quality > 0) {
                    // Check accessibility from above
                    if (cell_empty(board, board_height, r, c) ||
                        cell_empty(board, board_height, r, c + 2)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                        if (quality == 2) {
                            int lines = TSPIN_COUNT_LINES(
                                r,   1u << (c+1),
                                r+1, 1u << c,
                                r+1, 1u << (c+1),
                                r+1, 1u << (c+2));
                            if (lines >= 2) multi++;
                        }
                    }
                }
            }

            // Rot 1 (T points right)
            // T cells: [0,1] [1,1] [1,2] [2,1] — center is [1,1]
            if (cell_empty(board, board_height, r, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 1, c + 2) &&
                cell_empty(board, board_height, r + 2, c + 1)) {
                int front = tr + br;
                int back = tl + bl;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2;
                else if (front == 1 && back == 2) quality = 1;
                if (quality > 0) {
                    if (cell_empty(board, board_height, r, c + 2) ||
                        r == 0 ||
                        cell_empty(board, board_height, r - 1, c + 1)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                        if (quality == 2) {
                            int lines = TSPIN_COUNT_LINES(
                                r,   1u << (c+1),
                                r+1, 1u << (c+1),
                                r+1, 1u << (c+2),
                                r+2, 1u << (c+1));
                            if (lines >= 2) multi++;
                        }
                    }
                }
            }

            // Rot 3 (T points left)
            // T cells: [0,1] [1,0] [1,1] [2,1] — center is [1,1]
            if (cell_empty(board, board_height, r, c + 1) &&
                cell_empty(board, board_height, r + 1, c) &&
                cell_empty(board, board_height, r + 1, c + 1) &&
                cell_empty(board, board_height, r + 2, c + 1)) {
                int front = tl + bl;
                int back = tr + br;
                int quality = 0;
                if (front == 2 && back >= 1) quality = 2;
                else if (front == 1 && back == 2) quality = 1;
                if (quality > 0) {
                    if (cell_empty(board, board_height, r, c) ||
                        r == 0 ||
                        cell_empty(board, board_height, r - 1, c + 1)) {
                        setups++;
                        if (quality > best_quality) best_quality = quality;
                        if (quality == 2) {
                            int lines = TSPIN_COUNT_LINES(
                                r,   1u << (c+1),
                                r+1, 1u << c,
                                r+1, 1u << (c+1),
                                r+2, 1u << (c+1));
                            if (lines >= 2) multi++;
                        }
                    }
                }
            }
            #undef TSPIN_COUNT_LINES
        }
    }

    *t_slot_quality = best_quality;
    if (t_multiline_setups) *t_multiline_setups = multi;
    return setups;
}

// ============================================================
// Spin-Placement Counting (immobile placements)
//
// For each piece type (excluding O and T) × each orientation × each
// board position, check:
//   1. FITS:      All piece cells are empty on the board
//   2. REACHABLE: At least one piece cell is reachable from surface
//   3. IMMOBILE:  Piece cannot move in any cardinal direction
//                 (the actual ALL_MINI criterion — matches
//                 b2b_check_immobility used during placement)
//
// O is excluded (can't spin). T is excluded (T-spins use the
// 3-corner rule, handled separately by detect_t_spin_setups).
//
// Uses the existing B2B_PIECES definitions (row_masks bitmask format).
// Pieces are indexed 1-7 (PIECE_I through PIECE_Z), skipping PIECE_N=0.
// ============================================================

typedef struct {
    float weighted_immobile;            // Queue-weighted sum of truly immobile placements
    float weighted_immobile_clearing;   // Queue-weighted sum of immobile + line-clearing placements
    float weighted_immobile_lines;      // Queue-weighted sum of clearable lines from immobile placements
} ImmobilePlacementResult;

// count_immobile_placements scans only pieces that appear in the upcoming queue
// (hold piece + next queue pieces). Each piece's contribution is weighted by
// how soon it appears: the next piece gets weight 1.0, the one after gets 0.5,
// then 0.33, etc. (1/position). This ensures the heuristic rewards setups
// that can be resolved SOON rather than speculative cavities for distant pieces.
//
// Populates two per-row bitmasks:
//   immobile_cells[]   — cells in ANY valid immobile placement (used for
//                         wasted-hole detection: reachable holes not in here
//                         are "wasted").
//   clearing_cells[]   — cells in immobile placements that CLEAR at least one
//                         line (used to exempt productive spin-setup holes from
//                         hole penalties — only clearing setups earn exemption).
//
// upcoming_pieces: array of piece types (1-7) to check, ordered by priority
// num_upcoming: length of upcoming_pieces
static ImmobilePlacementResult count_immobile_placements(
    const uint16_t* board, int board_height,
    const uint16_t* reachable,
    uint16_t* immobile_cells,        // output: ALL immobile placement cells
    uint16_t* clearing_cells,        // output: only clearing immobile cells
    const int* upcoming_pieces,
    int num_upcoming,
    const int* piece_queue_count     // [8], count of each piece type in upcoming;
                                     // per-type reward is capped at this count so
                                     // four S-slots with one S in queue reward only
                                     // as much as one usable slot.
) {
    ImmobilePlacementResult res = {0.0f, 0.0f, 0.0f};
    memset(immobile_cells, 0, sizeof(uint16_t) * board_height);
    memset(clearing_cells, 0, sizeof(uint16_t) * board_height);

    if (num_upcoming <= 0) return res;

    uint16_t full_mask = (1 << BOARD_COLS) - 1;

    // Find highest filled row to bound the scan
    int top_filled = board_height;
    for (int r = 0; r < board_height; r++) {
        if (board[r] != 0) { top_filled = r; break; }
    }
    // Start a few rows above to catch pieces partially above the stack.
    // End a few rows below because an immobile placement requires adjacent
    // stack cells — pieces sitting multiple rows below the stack top cannot
    // be immobile (no walls to trap them except in rare hole cavities, which
    // we intentionally skip as a cost/value trade-off).
    int scan_start = top_filled - 3;
    if (scan_start < 0) scan_start = 0;
    int scan_end = top_filled + 5;
    if (scan_end > board_height) scan_end = board_height;

    // Build per-piece-type best weight from queue position.
    // If a piece appears multiple times in the queue, use the best (earliest)
    // weight. Weight for position i = 1.0 / (i + 1).
    float piece_weight[8]; // indexed by piece type (0=N unused, 1-7)
    memset(piece_weight, 0, sizeof(piece_weight));
    for (int i = 0; i < num_upcoming; i++) {
        int pt = upcoming_pieces[i];
        if (pt < PIECE_I || pt > PIECE_Z) continue;
        float w = 1.0f / (float)(i + 1);
        if (w > piece_weight[pt]) {
            piece_weight[pt] = w;
        }
    }

    // Iterate over piece types that have non-zero weight
    for (int pt = PIECE_I; pt <= PIECE_Z; pt++) {
        if (piece_weight[pt] <= 0.0f) continue;
        // Skip O piece (can't spin) and T piece (uses corner rule, not immobility)
        if (pt == PIECE_O || pt == PIECE_T) continue;

        float w = piece_weight[pt];

        // Per-piece accumulators: count all valid placements/lines for this
        // piece type, then apply the queue-count cap at the end.  This caps
        // the scalar reward without restricting which slots populate the
        // immobile_cells / clearing_cells bitmasks (those remain full so
        // hole-exemption still works on every valid spin slot).
        int placements_this_piece = 0;
        int clearing_placements_this_piece = 0;
        int lines_this_piece = 0;

        for (int rot = 0; rot < ROTATIONS; rot++) {
            PieceOrientation* ori = &B2B_PIECES[pt].orientations[rot];

            for (int r = scan_start; r < scan_end; r++) {
                if (r + ori->min_row < 0) continue;
                if (r + ori->max_row >= board_height) break;

                for (int c = -ori->min_col; c < BOARD_COLS - ori->max_col; c++) {
                    // 1. FITS: all piece cells are empty
                    bool fits = true;
                    for (int i = 0; i < 4; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        if (board[br] & shifted) { fits = false; break; }
                    }
                    if (!fits) continue;

                    // 2. REACHABLE: at least one piece cell is reachable
                    bool any_reachable = false;
                    for (int i = 0; i < 4 && !any_reachable; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        if (reachable[br] & shifted) any_reachable = true;
                    }
                    if (!any_reachable) continue;

                    // 3. TRULY IMMOBILE: piece cannot move in any cardinal
                    //    direction (ALL_MINI criterion). Bail on first
                    //    unblocked direction.
                    if (!b2b_check_collision(board, board_height, pt, rot, r + 1, c)) continue;
                    if (!b2b_check_collision(board, board_height, pt, rot, r - 1, c)) continue;
                    if (!b2b_check_collision(board, board_height, pt, rot, r, c + 1)) continue;
                    if (!b2b_check_collision(board, board_height, pt, rot, r, c - 1)) continue;

                    // Valid immobile placement! Count clearable lines first.
                    int lines = 0;
                    for (int i = 0; i < 4; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        uint16_t combined = board[br] | shifted;
                        if ((combined & full_mask) == full_mask) lines++;
                    }

                    // Mark cells: ALL placements go into immobile_cells
                    // (for wasted-hole detection); only CLEARING placements
                    // go into clearing_cells (for hole-penalty exemption).
                    for (int i = 0; i < 4; i++) {
                        if (ori->row_masks[i] == 0) continue;
                        int br = r + i;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i]) << c;
                        immobile_cells[br] |= shifted;
                        if (lines > 0) clearing_cells[br] |= shifted;
                    }

                    placements_this_piece++;
                    if (lines > 0) {
                        clearing_placements_this_piece++;
                        lines_this_piece += lines;
                    }
                }
            }
        }

        // Apply queue-count cap.  A piece that appears once in the queue can
        // only use ONE spin slot on the next drop, so crediting four slots
        // equally was over-rewarding redundant construction.  For non-T
        // immobile placements, clears are almost always singles (ALL_MINI
        // logic), so capping lines at queue_count * 4 is a safe ceiling.
        int qc = 0;
        if (piece_queue_count != NULL && pt >= 0 && pt < 8) qc = piece_queue_count[pt];
        if (qc <= 0) qc = 1; // defensive: weight was non-zero, so piece IS in queue

        int capped_placements = placements_this_piece < qc ? placements_this_piece : qc;
        int capped_clearing   = clearing_placements_this_piece < qc ? clearing_placements_this_piece : qc;
        int lines_cap         = qc * 4;
        int capped_lines      = lines_this_piece < lines_cap ? lines_this_piece : lines_cap;

        res.weighted_immobile          += w * (float)capped_placements;
        res.weighted_immobile_clearing += w * (float)capped_clearing;
        res.weighted_immobile_lines    += w * (float)capped_lines;
    }

    return res;
}

// ============================================================
// b2b_chain_rollout
// ============================================================
//
// Simulate placing the next `max_rollout_depth` upcoming pieces on the frozen
// `board` using a scored-greedy policy that prefers b2b-maintaining moves.
// Returns the number of SPIN CLEARS produced during the rollout — these are
// the placements that actually grow the b2b counter.  Safe drops (non-clearing
// placements) are allowed as transitional steps so the rollout can survive
// pieces that don't immediately slot into a spin, but they don't count toward
// the reward; rewarding them would push the search toward passive upstacking.
// Non-spin line clears (which break b2b) are never selected.
//
// Stops early when:
//   - a piece has no safe, b2b-preserving placement (rollout terminated),
//   - a placement would exceed the survivable height ceiling, or
//   - the queue / depth budget is exhausted.
//
// Used as a leaf heuristic to approximate high-depth search: the leaf score
// grows with how many future pieces demonstrably continue the b2b chain on
// this frozen board.  This extends the effective planning horizon without
// expanding the beam.
//
// Cost per call is bounded by the same scan range `count_immobile_placements`
// uses (top_filled ± ~5 rows), times 4 rotations, times 10 columns, times
// max_rollout_depth pieces.  Each candidate does an O(board_height) lock+clear
// simulation on a scratch copy.  All buffers are on the stack.
static int b2b_chain_rollout(
    const uint16_t* board, int board_height,
    const int* upcoming, int num_upcoming,
    int max_rollout_depth
) {
    if (num_upcoming <= 0 || max_rollout_depth <= 0) return 0;

    uint16_t sim_board[BOARD_ROWS];
    memcpy(sim_board, board, sizeof(uint16_t) * board_height);

    int spin_clears = 0;
    int limit = max_rollout_depth < num_upcoming ? max_rollout_depth : num_upcoming;
    int max_allowed = board_height - 4;

    for (int pi = 0; pi < limit; pi++) {
        int pt = upcoming[pi];
        if (pt < PIECE_I || pt > PIECE_Z) break;

        // Scan bounds: same pattern as count_immobile_placements.
        int top_filled = board_height;
        for (int r = 0; r < board_height; r++) {
            if (sim_board[r] != 0) { top_filled = r; break; }
        }
        int scan_start = top_filled - 3;
        if (scan_start < 0) scan_start = 0;
        int scan_end = top_filled + 5;
        if (scan_end > board_height) scan_end = board_height;

        uint16_t reachable[BOARD_ROWS];
        compute_reachability(sim_board, board_height, reachable);

        int best_score = -(1 << 30);
        int best_r = -1, best_c = 0, best_rot = 0;
        bool best_is_spin_clear = false;

        for (int rot = 0; rot < ROTATIONS; rot++) {
            PieceOrientation* ori = &B2B_PIECES[pt].orientations[rot];

            for (int r = scan_start; r < scan_end; r++) {
                if (r + ori->min_row < 0) continue;
                if (r + ori->max_row >= board_height) break;

                for (int c = -ori->min_col; c < BOARD_COLS - ori->max_col; c++) {
                    // FITS
                    bool fits = true;
                    for (int i2 = 0; i2 < 4; i2++) {
                        if (ori->row_masks[i2] == 0) continue;
                        int br = r + i2;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i2]) << c;
                        if (sim_board[br] & shifted) { fits = false; break; }
                    }
                    if (!fits) continue;

                    // RESTING (a piece cell must have collision one row below,
                    // i.e. the piece is sitting on the stack or floor).
                    if (!b2b_check_collision(sim_board, board_height, pt, rot, r + 1, c)) {
                        continue;
                    }

                    // REACHABLE from above (flood-fill intersects the piece).
                    bool any_reachable = false;
                    for (int i2 = 0; i2 < 4 && !any_reachable; i2++) {
                        if (ori->row_masks[i2] == 0) continue;
                        int br = r + i2;
                        uint16_t shifted = (uint16_t)(ori->row_masks[i2]) << c;
                        if (reachable[br] & shifted) any_reachable = true;
                    }
                    if (!any_reachable) continue;

                    // IMMOBILE at lock-time drives the ALL_MINI/T-spin signal.
                    // Uses the full 4-direction collision test.  Sufficient for
                    // rollout heuristic — we don't need to distinguish T-full vs
                    // T-mini since both preserve b2b.
                    bool is_spin = b2b_check_immobility(sim_board, board_height, pt, rot, r, c);

                    // Simulate lock + clear on a scratch board.
                    uint16_t after[BOARD_ROWS];
                    memcpy(after, sim_board, sizeof(uint16_t) * board_height);
                    lock_piece_on_board(after, board_height, pt, rot, r, c);
                    int lines = clear_lines(after, board_height);

                    // A non-spin line clear breaks b2b — skip; the chain would
                    // terminate here, so we don't want the rollout to select it.
                    if (lines >= 1 && !is_spin) continue;

                    // Compute post-placement max height for the survival gate.
                    int max_h_after = 0;
                    for (int col2 = 0; col2 < BOARD_COLS; col2++) {
                        uint16_t bit = (uint16_t)(1 << col2);
                        for (int r2 = 0; r2 < board_height; r2++) {
                            if (after[r2] & bit) {
                                int h = board_height - r2;
                                if (h > max_h_after) max_h_after = h;
                                break;
                            }
                        }
                    }
                    if (max_h_after >= max_allowed) continue;

                    // Cheap hole-creation estimate: count air pockets in the
                    // columns the piece touched.  Dedupe columns via OR'd mask
                    // so a vertical-I doesn't get quadruple-penalized.
                    uint16_t touched_cols = 0;
                    for (int i2 = 0; i2 < 4; i2++) {
                        touched_cols |= (uint16_t)(ori->row_masks[i2]);
                    }
                    touched_cols <<= c;

                    int holes_touch = 0;
                    uint16_t tc = touched_cols;
                    while (tc) {
                        int col2 = __builtin_ctz(tc);
                        tc &= (uint16_t)(tc - 1);
                        uint16_t bit = (uint16_t)(1 << col2);
                        int found_top = 0;
                        for (int r2 = 0; r2 < board_height; r2++) {
                            if (after[r2] & bit) found_top = 1;
                            else if (found_top) holes_touch++;
                        }
                    }

                    int score;
                    if (lines >= 1 && is_spin) {
                        // Strong preference for spin-clears: they reduce height
                        // AND grow the b2b chain.  Weight lines for TSD/TST.
                        score = 10000 + 100 * lines - max_h_after - 10 * holes_touch;
                    } else {
                        // Safe non-clearing drop; rewarded for keeping the stack
                        // short and not creating holes.
                        score = 1000 - max_h_after - 15 * holes_touch;
                    }

                    if (score > best_score) {
                        best_score = score;
                        best_r = r;
                        best_c = c;
                        best_rot = rot;
                        best_is_spin_clear = (lines >= 1 && is_spin);
                    }
                }
            }
        }

        // No b2b-preserving placement exists for this piece — chain ends.
        if (best_r < 0) break;

        lock_piece_on_board(sim_board, board_height, pt, best_rot, best_r, best_c);
        clear_lines(sim_board, board_height);
        if (best_is_spin_clear) spin_clears++;
    }

    return spin_clears;
}

// Count how many holes are "deep" (have 2+ filled cells above them in the same column).
// Holes that are part of a spin setup (immobile_cells) are excluded.
static int count_deep_holes(const uint16_t* board, int board_height,
                            const uint16_t* immobile_cells) {
    int deep = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);
        int filled_above = 0;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                filled_above++;
            } else if (filled_above >= 2) {
                if (!(immobile_cells[r] & bit)) {
                    deep++;
                }
            }
        }
    }
    return deep;
}

// Compute per-column heights and derived stats
typedef struct {
    int col_heights[BOARD_COLS];
    int max_height;
    float avg_height;
    float bumpiness;
    int holes;
    int hole_columns;       // Number of distinct columns containing holes
    int clearable_rows;     // Rows with >= 8 filled cells
    int almost_full_rows;   // Rows with exactly 9 filled cells
    int well_depth;         // Depth of the deepest single well
    int well_count;         // Number of wells (columns lower than both neighbors by 2+)
    int accessible_9_rows;  // Almost-full rows where the hole has clear path from above
    int blocked_9_rows;     // Almost-full rows where the hole is blocked
    int well_col;           // Column of the deepest well (-1 if no well)
    int well_aligned_9;     // Accessible 9-rows where the gap is in the well column
    int non_well_9;         // Accessible 9-rows where the gap is NOT in the well column
    int cascade_depth;      // Consecutive 9-rows from the top of the stack (after accessible top)
    int t_spin_setups;      // Number of T-spin setups detected
    int t_slot_quality;     // Best T-slot quality (0=none, 1=mini, 2=full)
    int t_multiline_setups; // Number of T-spin setups that clear >=2 lines (TSD/TST)
    int deep_holes;         // Holes buried under 2+ filled cells
    int edge_well_depth;    // Deepest well in columns 0, 1, 8, or 9
    float hole_ceiling_weight; // Weighted count of filled cells above enclosed holes
    float immobile_placements;           // Queue-weighted truly-immobile spin-placement count
    float immobile_clearing_placements;  // Queue-weighted immobile + line-clearing placements
    float immobile_clearable_lines;      // Queue-weighted sum of clearable lines from immobile placements
    int wasted_holes;                    // Non-enclosed holes not part of any immobile placement
    int t_queue_count;                   // Number of T pieces in the upcoming queue — used to cap W_TSLOT
    float bumpiness_exempted;            // Bumpiness excluding contributions adjacent to the well column
    float future_immobile_clearing;      // Immobile clearing placements considering ONLY the next 3 pieces.
                                         // A near-horizon proxy for "can we continue holding b2b with the
                                         // pieces we're actually about to receive" — the full-queue count
                                         // already captures this but dilutes near-term pieces under the
                                         // 1/(i+1) weighting; this narrower signal avoids that dilution.
    int chain_rollout_length;            // Pieces of scored-greedy rollout that maintain b2b on this frozen
                                         // board, over the next CHAIN_ROLLOUT_K upcoming pieces.  Dominant
                                         // horizon-extension signal: see b2b_chain_rollout() for the policy.
} BoardStats;

static BoardStats compute_board_stats(const uint16_t* board, int board_height,
                                      const int* upcoming_pieces, int num_upcoming,
                                      const int8_t* height_hint) {
    BoardStats s;
    memset(&s, 0, sizeof(s));

    uint16_t full_mask = (1 << BOARD_COLS) - 1;

    // Column heights — use pre-computed hint when available (cached on the
    // SearchState and patched incrementally after each placement).
    if (height_hint) {
        for (int c = 0; c < BOARD_COLS; c++) s.col_heights[c] = height_hint[c];
    } else {
        for (int c = 0; c < BOARD_COLS; c++) {
            s.col_heights[c] = 0;
            uint16_t bit = (1 << c);
            for (int r = 0; r < board_height; r++) {
                if (board[r] & bit) { s.col_heights[c] = board_height - r; break; }
            }
        }
    }

    // Max and average height
    float total_h = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        if (s.col_heights[c] > s.max_height) s.max_height = s.col_heights[c];
        total_h += s.col_heights[c];
    }
    s.avg_height = total_h / BOARD_COLS;

    // Bumpiness: sum of absolute column height differences.  We compute two
    // variants:
    //   - bumpiness:           the full sum, used when the penalty SHOULD
    //                          discourage welled boards (none currently).
    //   - bumpiness_exempted:  excludes adjacency contributions involving the
    //                          deepest well column.  A deliberate spin/Tetris
    //                          well is, by construction, a 2-sided height
    //                          discontinuity that adds ~2×well_depth to raw
    //                          bumpiness.  With W_BUMPINESS=1 this can exceed
    //                          the positive rewards for the well (e.g. a
    //                          depth-5 well contributes −10 vs only +4.5 from
    //                          W_WELL_ALIGNED_9) — net-penalizing exactly the
    //                          geometry we want the bot to build.  Exempting
    //                          the well column from bumpiness penalty resolves
    //                          this conflict without disabling the rest of the
    //                          flatness pressure.
    //
    // well_col is computed below, so we finish it first and come back.
    for (int c = 0; c < BOARD_COLS - 1; c++) {
        s.bumpiness += fabsf((float)(s.col_heights[c] - s.col_heights[c + 1]));
    }

    // Flood-fill reachability (used for holes and spin placements)
    uint16_t reachable[BOARD_ROWS];
    compute_reachability(board, board_height, reachable);

    // ── Immobile spin-placement counting FIRST ──────────────────
    // Computed before hole metrics so that immobile_cells[] is available
    // to exempt spin-setup holes from all hole penalties.  Holes that
    // are part of a valid immobile placement are intentional cavities,
    // not structural damage.
    uint16_t immobile_cells[BOARD_ROWS];
    uint16_t clearing_cells[BOARD_ROWS];

    // Count each piece type in the upcoming queue — used to cap the per-piece
    // reward inside count_immobile_placements so that redundant slots don't
    // get rewarded beyond the number of pieces actually coming.
    int piece_queue_count[8] = {0};
    for (int i = 0; i < num_upcoming; i++) {
        int pt = upcoming_pieces[i];
        if (pt >= 0 && pt < 8) piece_queue_count[pt]++;
    }
    s.t_queue_count = piece_queue_count[PIECE_T];

    ImmobilePlacementResult ipr = count_immobile_placements(board, board_height, reachable,
                                                             immobile_cells, clearing_cells,
                                                             upcoming_pieces, num_upcoming,
                                                             piece_queue_count);
    s.immobile_placements = ipr.weighted_immobile;
    s.immobile_clearing_placements = ipr.weighted_immobile_clearing;
    s.immobile_clearable_lines = ipr.weighted_immobile_lines;

    // Near-horizon projection: repeat the scan with only the first 3 upcoming
    // pieces.  This produces a signal that emphasizes whether the IMMEDIATELY
    // next few pieces can maintain b2b on this board — which is what the
    // 7-ply search needs to commit to a hold-indefinitely strategy.  Uses a
    // scratch buffer for the cell masks; we don't need those outputs.
    int near_n = num_upcoming < 3 ? num_upcoming : 3;
    if (near_n > 0) {
        int near_queue_count[8] = {0};
        for (int i = 0; i < near_n; i++) {
            int pt = upcoming_pieces[i];
            if (pt >= 0 && pt < 8) near_queue_count[pt]++;
        }
        uint16_t scratch_immobile[BOARD_ROWS];
        uint16_t scratch_clearing[BOARD_ROWS];
        ImmobilePlacementResult ipr_near = count_immobile_placements(
            board, board_height, reachable,
            scratch_immobile, scratch_clearing,
            upcoming_pieces, near_n, near_queue_count);
        s.future_immobile_clearing = ipr_near.weighted_immobile_clearing;
    }

    // Chain rollout: scored-greedy simulation over the next CHAIN_ROLLOUT_K
    // pieces on the frozen board.  Dominant horizon-extension signal —
    // separates "b2b chain is sustainable" from "b2b exists but dies at
    // horizon+1" in a way count_immobile_placements cannot.  Cached per
    // unique state via the transposition table on the leaf score path.
    s.chain_rollout_length = b2b_chain_rollout(board, board_height,
                                               upcoming_pieces, num_upcoming,
                                               CHAIN_ROLLOUT_K);

    // ── Hole metrics (setup-aware) ──────────────────────────────
    // Hole penalties exclude cells in clearing_cells[] — those are
    // intentional cavities for spin setups that would clear lines
    // (b2b-maintaining downstack).  Non-clearing immobile placements
    // are NOT exempted: holes without clearing potential are still
    // structural damage.

    // ── Hole metrics ──────────────────────────────────────────
    // Holes are penalized uniformly — spin-setup holes are NOT exempted
    // here.  Instead, the spin-setup REWARDS in evaluate_state are strong
    // enough to outweigh the hole cost, so the bot only creates holes
    // when it has a productive clearing setup, not speculatively.
    uint16_t no_exempt[BOARD_ROWS];
    memset(no_exempt, 0, sizeof(no_exempt));

    s.holes = count_hole_sections(board, board_height, reachable, no_exempt);

    s.hole_columns = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);
        bool found_filled = false;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                found_filled = true;
            } else if (found_filled) {
                s.hole_columns++;
                break;
            }
        }
    }

    s.deep_holes = count_deep_holes(board, board_height, no_exempt);
    s.hole_ceiling_weight = compute_hole_ceiling_weight(board, board_height, reachable, no_exempt);

    // T-spin setup detection
    s.t_spin_setups = detect_t_spin_setups(board, board_height, &s.t_slot_quality, &s.t_multiline_setups);

    // Nearly-complete rows (8+ out of 10 cells filled)
    for (int r = 0; r < board_height; r++) {
        uint16_t row = board[r] & full_mask;
        int bits = 0;
        uint16_t v = row;
        while (v) { bits++; v &= v - 1; }
        if (bits >= 8) s.clearable_rows++;
        if (bits == 9) s.almost_full_rows++;
    }

    // Well detection: columns lower than both neighbors by 2+
    s.well_depth = 0;
    s.well_count = 0;
    s.well_col = -1;
    s.edge_well_depth = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        int left_h = (c > 0) ? s.col_heights[c - 1] : board_height;
        int right_h = (c < BOARD_COLS - 1) ? s.col_heights[c + 1] : board_height;
        int min_neighbor = (left_h < right_h) ? left_h : right_h;
        int depth = min_neighbor - s.col_heights[c];
        if (depth >= 2) {
            s.well_count++;
            if (depth > s.well_depth) {
                s.well_depth = depth;
                s.well_col = c;
            }
            if (c <= 1 || c >= 8) {
                if (depth > s.edge_well_depth) s.edge_well_depth = depth;
            }
        }
    }

    // Bumpiness exemption: now that well_col is known, compute the variant
    // that skips the two adjacencies around the deepest well.
    s.bumpiness_exempted = s.bumpiness;
    if (s.well_col >= 0) {
        if (s.well_col > 0) {
            s.bumpiness_exempted -= fabsf((float)(s.col_heights[s.well_col - 1] - s.col_heights[s.well_col]));
        }
        if (s.well_col < BOARD_COLS - 1) {
            s.bumpiness_exempted -= fabsf((float)(s.col_heights[s.well_col] - s.col_heights[s.well_col + 1]));
        }
        if (s.bumpiness_exempted < 0.0f) s.bumpiness_exempted = 0.0f;
    }

    // Accessible vs blocked almost-full rows
    s.accessible_9_rows = 0;
    s.blocked_9_rows = 0;
    s.well_aligned_9 = 0;
    s.non_well_9 = 0;
    for (int r = 0; r < board_height; r++) {
        uint16_t row = board[r] & full_mask;
        int bits = 0;
        uint16_t v = row;
        while (v) { bits++; v &= v - 1; }
        if (bits != 9) continue;

        uint16_t hole_mask = (~row) & full_mask;
        int hole_col = -1;
        for (int c = 0; c < BOARD_COLS; c++) {
            if (hole_mask & (1 << c)) { hole_col = c; break; }
        }
        if (hole_col < 0) continue;

        bool accessible = true;
        uint16_t hbit = (1 << hole_col);
        for (int rr = r - 1; rr >= 0; rr--) {
            if (board[rr] & hbit) { accessible = false; break; }
        }
        if (accessible) {
            s.accessible_9_rows++;
            if (s.well_col >= 0 && hole_col == s.well_col) {
                s.well_aligned_9++;
            } else {
                s.non_well_9++;
            }
        } else {
            s.blocked_9_rows++;
        }
    }

    // Cascade depth: count consecutive 9-rows starting from the topmost filled
    // row, requiring that the top 9-row's gap is accessible from above.  This
    // is the structural signal for a multi-clear downstack: once the top is
    // popped, each subsequent 9-row becomes the new top and can be cleared.
    s.cascade_depth = 0;
    for (int r = 0; r < board_height; r++) {
        uint16_t row = board[r] & full_mask;
        if (row == 0) continue;  // skip empty rows above the stack

        int bits = 0;
        uint16_t v = row;
        while (v) { bits++; v &= v - 1; }
        if (bits != 9) break;    // topmost filled row is not a 9-row

        uint16_t hole_mask = (~row) & full_mask;
        int hole_col = __builtin_ctz(hole_mask);
        bool accessible = true;
        uint16_t hbit = (uint16_t)(1u << hole_col);
        for (int rr = r - 1; rr >= 0; rr--) {
            if (board[rr] & hbit) { accessible = false; break; }
        }
        if (!accessible) break;

        for (int rr = r; rr < board_height; rr++) {
            uint16_t row2 = board[rr] & full_mask;
            int bits2 = 0;
            uint16_t v2 = row2;
            while (v2) { bits2++; v2 &= v2 - 1; }
            if (bits2 != 9) break;
            s.cascade_depth++;
        }
        break;
    }

    // Wasted holes: non-enclosed holes (reachable empty cells below a filled
    // cell in the same column) that are NOT part of any immobile placement.
    s.wasted_holes = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        uint16_t bit = (1 << c);
        bool found_filled = false;
        for (int r = 0; r < board_height; r++) {
            if (board[r] & bit) {
                found_filled = true;
            } else if (found_filled) {
                bool is_reachable = (reachable[r] & bit) != 0;
                bool is_immobile = (immobile_cells[r] & bit) != 0;
                if (is_reachable && !is_immobile) {
                    s.wasted_holes++;
                }
            }
        }
    }

    return s;
}

// ── Cheap pre-score ──────────────────────────────────────────
// Approximates evaluate_state at roughly 10x the speed by skipping the
// expensive reachability / immobile / spin / hole-ceiling computations.
// Uses only per-column heights (already cached on SearchState) plus a
// direct scan for "simple holes" (empty cells below column tops).  Used
// as a lower-bound filter for aspiration pruning before running the full
// eval.  Important property: cheap_prescore <= evaluate_state + SLACK in
// expectation, because all positive spin/cascade/downstack bonuses are
// only additive (the cheap version can only UNDERESTIMATE, never exceed).
static float cheap_prescore(const SearchState* state, int board_height) {
    int max_height = 0;
    int sum_height = 0;
    float bumpiness = 0.0f;
    for (int c = 0; c < BOARD_COLS; c++) {
        int h = state->col_heights[c];
        if (h > max_height) max_height = h;
        sum_height += h;
        if (c > 0) bumpiness += fabsf((float)(h - state->col_heights[c - 1]));
    }

    int effective_h = max_height + state->garbage_remaining;
    int max_allowed = board_height - 4;
    if (effective_h >= max_allowed) return -1e6f;

    float h_ratio = (float)effective_h / (float)max_allowed;
    float score = 0.0f;

    if (effective_h >= max_allowed - 2) {
        int slack = max_allowed - 1 - effective_h;
        score -= W_NEAR_DEATH * (float)(2 - slack);
    }
    score -= W_HEIGHT_QUARTIC * h_ratio * h_ratio * h_ratio * h_ratio;
    score -= W_AVG_HEIGHT * ((float)sum_height / (float)BOARD_COLS);
    score -= W_BUMPINESS * bumpiness;

    int simple_holes = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        int h = state->col_heights[c];
        if (h <= 0) continue;
        int col_top_row = board_height - h;
        uint16_t bit = (uint16_t)(1u << c);
        for (int r = col_top_row + 1; r < board_height; r++) {
            if (!(state->board[r] & bit)) simple_holes++;
        }
    }
    if (simple_holes > 0) {
        int capped = simple_holes < HOLES_CAP ? simple_holes : HOLES_CAP;
        float hole_mult = 1.0f + 0.5f * h_ratio;
        score -= W_HOLES * (float)capped * hole_mult;
    }

    if (state->b2b >= 0) score += W_B2B_FLAT;
    if (state->b2b > 0) {
        score += W_B2B_SQRT * sqrtf((float)state->b2b);
        score += W_B2B_LINEAR * (float)state->b2b;
    }
    if (state->total_attack > 0.0f) score += W_ATTACK_TOTAL * state->total_attack;
    if (state->max_single_attack > 0.0f) score += W_MAX_SINGLE * state->max_single_attack;
    if (state->total_attack > 0.0f && state->pieces_placed > 0)
        score += W_APP * (state->total_attack / (float)state->pieces_placed);
    if (state->combo > 0) {
        int c = state->combo < COMBO_CAP ? state->combo : COMBO_CAP;
        score += W_COMBO * (float)c;
    }

    return score;
}

// ── Running top-K score tracker (min-heap of size K) ─────────
// Used during beam expansion to maintain a rolling lower bound so that
// children whose cheap_prescore + ASPIRATION_SLACK < current K-th best
// full score can be skipped entirely.
static float g_topk_heap[MAX_BEAM_WIDTH];
static int   g_topk_size = 0;
static int   g_topk_cap  = 0;

static const float ASPIRATION_SLACK = 15.0f;

static inline void topk_reset(int cap) {
    g_topk_size = 0;
    g_topk_cap = cap;
}

static inline void topk_sift_up(int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (g_topk_heap[p] > g_topk_heap[i]) {
            float t = g_topk_heap[i]; g_topk_heap[i] = g_topk_heap[p]; g_topk_heap[p] = t;
            i = p;
        } else break;
    }
}

static inline void topk_sift_down(int i) {
    int n = g_topk_size;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < n && g_topk_heap[l] < g_topk_heap[s]) s = l;
        if (r < n && g_topk_heap[r] < g_topk_heap[s]) s = r;
        if (s == i) break;
        float t = g_topk_heap[i]; g_topk_heap[i] = g_topk_heap[s]; g_topk_heap[s] = t;
        i = s;
    }
}

static inline void topk_insert(float score) {
    if (g_topk_cap <= 0) return;
    if (g_topk_size < g_topk_cap) {
        g_topk_heap[g_topk_size++] = score;
        topk_sift_up(g_topk_size - 1);
    } else if (score > g_topk_heap[0]) {
        g_topk_heap[0] = score;
        topk_sift_down(0);
    }
}

static inline float topk_floor(void) {
    if (g_topk_cap <= 0 || g_topk_size < g_topk_cap) return -1e9f;
    return g_topk_heap[0];
}

static float evaluate_state(const SearchState* state, int board_height,
                            const int* queue, int queue_len) {
    uint64_t _tt_h = tt_hash(state, board_height);
    uint32_t _tt_idx = (uint32_t)(_tt_h & TT_MASK);
    TTEntry* _tt_e = &g_tt[_tt_idx];
    if (_tt_e->hash == _tt_h && _tt_e->generation != 0 &&
        (g_tt_generation - _tt_e->generation) <= TT_GENERATION_EXPIRY) {
        return _tt_e->score;
    }

    float score = 0.0f;

    int upcoming[MAX_SEARCH_DEPTH + 2];
    int num_upcoming = 0;
    if (state->hold_piece != PIECE_N) {
        upcoming[num_upcoming++] = state->hold_piece;
    }
    for (int i = state->next_queue_idx; i < queue_len && num_upcoming < MAX_SEARCH_DEPTH + 2; i++) {
        upcoming[num_upcoming++] = queue[i];
    }

    BoardStats bs = compute_board_stats(state->board, board_height,
                                        upcoming, num_upcoming, state->col_heights);

    int max_allowed = board_height - 4; // rows 0..3 are the death zone
    int effective_h = bs.max_height + state->garbage_remaining;

    // Instant death — inviolable floor.
    if (effective_h >= max_allowed) {
        return -1e6f;
    }

    float h_ratio = (float)effective_h / (float)max_allowed;

    // ── §2.1 SURVIVAL WALL ────────────────────────────────────

    // Near-death cliff: within 2 rows of the death zone, stack an enormous
    // penalty that beats every positive term combined.  slack=0 means the
    // very next block kills us.
    if (effective_h >= max_allowed - 2) {
        int slack = max_allowed - 1 - effective_h; // 0 or 1
        score -= W_NEAR_DEATH * (float)(2 - slack);
    }

    // Smooth height — quartic.  Nearly free in the playable zone, flares
    // hard near the top (h=0.5 → -6.25, h=0.8 → -41, h=0.9 → -65.6,
    // h=1.0 → -100).  At h≥0.9 this alone outweighs the max b2b store.
    score -= W_HEIGHT_QUARTIC * h_ratio * h_ratio * h_ratio * h_ratio;

    // Linear volume penalty — rewards board emptiness.  While W_HEIGHT_QUARTIC
    // barely fires until the stack is tall, this fires proportionally with
    // every added cell from the first piece.  Prevents the bot from hoarding
    // spin slots / deep wells that are "safe" (height ratio still low) but
    // that are actually consuming cells without cashing them in — the
    // mechanism that lets indefinite-b2b devolve into "stack to the moon
    // because the quartic penalty hasn't fired yet".  Together with the
    // b2b-reward store, this enforces b2b-PER-PIECE efficiency: the bot is
    // rewarded only when the b2b it builds outpaces the cells it consumes.
    score -= W_AVG_HEIGHT * bs.avg_height;

    // Bumpiness — linear, uncapped; a terrible surface must stay terrible.
    // Use the well-column-exempted variant so deliberate spin/Tetris wells
    // (which are REQUIRED for b2b-maintaining clears) are not double-penalized
    // against the reward terms that fire for them.
    score -= W_BUMPINESS * bs.bumpiness_exempted;

    // ── §2.2 HOLE ACCOUNTING ──────────────────────────────────

    float hole_mult = 1.0f + 0.5f * h_ratio;

    // Capped hole count: many holes are bad, but caping at 8 prevents the
    // pathological "20 holes so I should suicide to escape the penalty".
    if (bs.holes > 0) {
        int capped = bs.holes < HOLES_CAP ? bs.holes : HOLES_CAP;
        score -= W_HOLES * (float)capped * hole_mult;
    }

    // Reachable holes not part of any immobile placement are wasted.
    if (bs.wasted_holes > 0) {
        score -= W_WASTED_HOLE * (float)bs.wasted_holes * hole_mult;
    }

    // Burying holes deeper is worse than leaving them near the surface.
    // (This term was computed but unused before the rework.)
    if (bs.hole_ceiling_weight > 0.0f) {
        score -= W_HOLE_CEILING * bs.hole_ceiling_weight;
    }

    // Forgiveness for holes that are part of a CLEARING spin setup.
    // Bounded below hole cost so the bot cannot net-reward creating holes.
    if (bs.holes > 0) {
        float setup_count = bs.immobile_clearing_placements + (float)bs.t_spin_setups;
        if (setup_count > 0.0f) {
            float forgiveness = W_HOLE_FORGIVE * fminf(setup_count, (float)bs.holes);
            score += forgiveness * hole_mult;
        }
    }

    // ── §2.7 WELL SHAPING ─────────────────────────────────────

    if (bs.well_count == 1 && bs.well_depth >= 4 && bs.well_depth <= 8) {
        score += 3.0f;
    } else if (bs.well_count == 1 && bs.well_depth >= 2 && bs.well_depth <= 3) {
        score += 1.0f;
    }
    if (bs.well_count > 1) {
        score -= 1.5f * (float)(bs.well_count - 1);
    }
    if (bs.edge_well_depth >= 3) {
        score -= 2.0f * (float)(bs.edge_well_depth - 2);
    }

    // ── §2.3 B2B ECONOMY ──────────────────────────────────────
    //
    // B2B value is SUBLINEAR (sqrt).  Rationale:
    //   - Surge payoff is LINEAR in prev_b2b and is realized through
    //     total_attack on break (captured in §2.4).
    //   - Making hold value sublinear ensures that at high chains the
    //     linear surge beats the marginal sqrt increment, so the bot
    //     proactively breaks onto downstack setups.
    //   - Crossover (with W_ATTACK_TOTAL=1.2, W_MAX_SINGLE=1.5):
    //       break_gain ≈ 1.2*b2b + 1.5*b2b = 2.7*b2b  (pure surge, no tail)
    //       hold_increment ≈ 8 * (sqrt(b2b+1) - sqrt(b2b)) ≈ 4/sqrt(b2b)
    //     → hold wins until roughly b2b=15–20 without a combo-ready board,
    //       break wins earlier once a downstack bank exists (§2.5).
    //
    // No explicit break penalty: losing W_B2B_FLAT + W_B2B_SQRT*sqrt(prev_b2b)
    // IS the cost, natively encoded by the eval delta.

    if (state->b2b >= 0) {
        score += W_B2B_FLAT;
    }
    if (state->b2b > 0) {
        score += W_B2B_SQRT * sqrtf((float)state->b2b);
        // Linear hold term: makes indefinite b2b growth the dominant strategy.
        // With W_B2B_LINEAR ≈ 12 and the sum of break-rewarding weights
        // (W_ATTACK_TOTAL + W_MAX_SINGLE + W_APP/pieces_placed) ≈ 9.3, holding
        // one more piece of b2b always pays more than breaking for a surge of
        // the current b2b value.  Only the near-death cliff (-5000) can make
        // the bot voluntarily drop b2b — which is the desired behavior.
        score += W_B2B_LINEAR * (float)state->b2b;
    }

    // ── §2.4 ATTACK REALIZATION ───────────────────────────────
    //
    // total_attack already includes surge (from breaks) and combo-multiplied
    // clears, because compute_attack() folds both into `ar.attack`.  Crediting
    // total_attack in the eval is what makes the bot value the spike.

    if (state->total_attack > 0.0f) {
        score += W_ATTACK_TOTAL * state->total_attack;
    }

    // Peak single attack: rewards concentration (one big hit > many small).
    // max_single_attack covers ALL clears including the surge-laden break,
    // unlike the old max_b2b_attack which missed the break by construction.
    if (state->max_single_attack > 0.0f) {
        score += W_MAX_SINGLE * state->max_single_attack;
    }

    // Mild tiebreaker in favor of b2b-maintaining attack of equal magnitude.
    if (state->b2b_attack > 0.0f) {
        score += W_B2B_ATTACK * state->b2b_attack;
    }

    // Direct APP (Attack Per Piece) optimization.  Complements total_attack
    // by preferring paths that concentrate attack into fewer pieces — the
    // natural signal for "spike over smear".
    if (state->total_attack > 0.0f && state->pieces_placed > 0) {
        score += W_APP * (state->total_attack / (float)state->pieces_placed);
    }

    // ── §2.5 COMBO POTENTIAL ──────────────────────────────────

    if (state->combo > 0) {
        int c = state->combo < COMBO_CAP ? state->combo : COMBO_CAP;
        score += W_COMBO * (float)c;
    }

    // Garbage-conditional multiplier for Tetris/surge geometry rewards.
    // These heuristics (9-rows, cascades, Tetris-well alignment, primed-spike
    // bonuses) encode value that only pays off via a b2b BREAK + combo-tail
    // surge.  That's the correct strategy under garbage pressure where deep
    // downstacking is required, but under no-garbage play it fights against
    // the "hold b2b indefinitely" objective and lowers attack-per-clear
    // efficiency compared to the T-spin / all-mini single-clear loop.
    //
    // The smooth clamp lets the weights fade in gradually as garbage piles up
    // (e.g. multiplier = 0.5 at 2 queued garbage lines, 1.0 at 4+).  This
    // avoids abrupt regime switches during the search and keeps the garbage
    // mode's downstack machinery fully active when it's actually needed.
    float garbage_mul = (float)state->garbage_remaining * 0.25f;
    if (garbage_mul > 1.0f) garbage_mul = 1.0f;
    if (garbage_mul < 0.0f) garbage_mul = 0.0f;

    // Downstack bank: nearly-full rows reachable from above (gap not buried).
    // Post-break, each piece into the gap continues the combo.  This is the
    // structural signal that tells the bot "the board is primed to spike".
    if (bs.accessible_9_rows > 0 && garbage_mul > 0.0f) {
        int d = bs.accessible_9_rows < DOWNSTACK_CAP ? bs.accessible_9_rows : DOWNSTACK_CAP;
        score += W_DOWNSTACK * (float)d * garbage_mul;
    }

    // Extra bonus when the 9-row gap aligns with the Tetris well — an I-piece
    // already knows where to go for the tetris/surge.
    if (bs.well_aligned_9 > 0 && garbage_mul > 0.0f) {
        int w = bs.well_aligned_9 < WELL_ALIGNED_9_CAP ? bs.well_aligned_9 : WELL_ALIGNED_9_CAP;
        score += W_WELL_ALIGNED_9 * (float)w * garbage_mul;
    }

    // Cascade depth: stacked 9-rows at the top of the pile.  Clearing one
    // exposes the next as the new top, enabling a multi-clear combo tail
    // after the break.  This is what turns a 13-damage surge into a 25+ spike.
    if (bs.cascade_depth > 0 && garbage_mul > 0.0f) {
        int cd = bs.cascade_depth < CASCADE_CAP ? bs.cascade_depth : CASCADE_CAP;
        score += W_CASCADE * (float)cd * garbage_mul;
    }

    // Surge potential: latent value of an untriggered spike.  Scales with
    // both the stored b2b chain and the downstack readiness, so a b2b=15
    // chain on a 3-layer cascade is valued ~12 even if the break is outside
    // the visible search horizon.
    if (state->b2b > 0 && garbage_mul > 0.0f) {
        float primed = 0.0f;
        if (bs.cascade_depth >= 1) primed = 1.0f;
        else if (bs.accessible_9_rows > 0) primed = 0.5f;
        if (primed > 0.0f) {
            int b_cap = state->b2b < 20 ? state->b2b : 20;
            score += W_SURGE_POT * primed * (float)b_cap * garbage_mul;
        }
    }

    // Break-readiness: one-shot bonus when a spike is a single placement away
    // (high b2b, 2+ cascade layers available).  Flips tie-ordering in favor
    // of committing to the break at exactly the right moment.
    if (state->b2b >= 8 && bs.cascade_depth >= 2 && garbage_mul > 0.0f) {
        score += W_BREAK_READY * garbage_mul;
    }

    // ── §2.6 SPIN SETUP REWARDS ───────────────────────────────

    // Queue-cap the T-slot reward: a T-slot that has no T in the upcoming
    // queue can't be used in the current horizon, so it should pay nothing.
    // With one T in queue, reward only ONE slot (not a multiplier on t_spin_setups).
    if (bs.t_spin_setups > 0 && bs.t_queue_count > 0) {
        float t_reward = 0.0f;
        if (bs.t_slot_quality == 2) {
            t_reward = W_TSLOT;
        } else if (bs.t_slot_quality == 1) {
            t_reward = W_TSLOT * 0.4f;
        }
        int usable_setups = bs.t_spin_setups < bs.t_queue_count ? bs.t_spin_setups : bs.t_queue_count;
        t_reward *= (1.0f + 0.3f * fminf((float)(usable_setups - 1), 2.0f));
        score += t_reward;
    }

    // T-spin multi-line bonus: only for detected TSD/TST slots AND only
    // when a T is actually coming.  This rewards the attack/clear-efficient
    // multi-line b2b clears the user wants prioritized over I-tetrises.
    if (bs.t_multiline_setups > 0 && bs.t_queue_count > 0) {
        int usable_multi = bs.t_multiline_setups < bs.t_queue_count ? bs.t_multiline_setups : bs.t_queue_count;
        score += W_TSPIN_MULTILINE * (float)usable_multi;
    }

    if (bs.immobile_clearing_placements > 0.0f) {
        float line_reward = W_IMMOBILE_CLEAR * sqrtf(bs.immobile_clearing_placements);
        line_reward += W_IMMOBILE_LINES * fminf(bs.immobile_clearable_lines, 8.0f);
        score += line_reward;
    }

    // Near-term b2b sustainability bonus.  Amplifies the hold strategy by
    // making states with many sustainable near-future placements strictly
    // better than equally-valued states without a clear continuation.
    //
    // Gated on TWO conditions to protect garbage survival:
    //   1. No pending garbage (`1 - garbage_mul`) — in garbage mode the bot
    //      must prioritize downstacking; over-rewarding future-b2b can flip
    //      the eval toward suicidal b2b-hoarding.
    //   2. Low board height (`1 - h_ratio`) — even in no-garbage, if the
    //      stack is climbing the bot must keep a path to survival clears
    //      over future-b2b sustainability.  Without this, step F pushes the
    //      bot to 18-row stacks in garbage benchmark.
    //
    // Both gates close toward 0 as conditions worsen, so the term cleanly
    // vanishes before it can tip the survival balance.
    if (bs.future_immobile_clearing > 0.0f && state->b2b > 0) {
        float no_garbage_mul = 1.0f - garbage_mul;
        float low_stack_mul = 1.0f - h_ratio;
        if (low_stack_mul < 0.0f) low_stack_mul = 0.0f;
        score += W_FUTURE_B2B * bs.future_immobile_clearing * no_garbage_mul * low_stack_mul;
    }

    // Chain rollout reward: one point per confirmed b2b-GROWING future spin-
    // clear discovered by the scored-greedy rollout.  Ungated — the rollout
    // already respects the survival ceiling internally (fatal placements are
    // skipped and non-spin clears are rejected), and a long rollout chain is
    // *especially* useful under garbage: it's the signal that says "this
    // downstack pattern is ALSO a spin-chain pattern", which is exactly the
    // compound survival+offense signal we want.  Active only while b2b flag
    // is alive so we don't tempt the eval to build spin chains from scratch.
    if (bs.chain_rollout_length > 0 && state->b2b >= 0) {
        score += W_CHAIN_ROLLOUT * (float)bs.chain_rollout_length;
    }

    _tt_e->hash = _tt_h;
    _tt_e->score = score;
    _tt_e->generation = g_tt_generation;
    return score;
}

// ============================================================
// Score Decomposition (for heuristic influence analysis)
// ============================================================

#define NUM_DECOMPOSE 21

#define D_HEIGHT         0
#define D_NEAR_DEATH     1
#define D_BUMPINESS      2
#define D_HOLES          3
#define D_WASTED_HOLES   4
#define D_HOLE_CEILING   5
#define D_HOLE_FORGIVE   6
#define D_WELL           7
#define D_B2B_FLAT       8
#define D_B2B_SQRT       9
#define D_COMBO          10
#define D_DOWNSTACK      11
#define D_TSLOT          12
#define D_IMMOBILE_CLEAR 13
#define D_MAX_SINGLE     14
#define D_ATTACK         15
#define D_CASCADE        16
#define D_SURGE_POT      17
#define D_APP            18
#define D_BREAK_READY    19
#define D_B2B_LINEAR     20

int b2b_get_num_decompose(void) { return NUM_DECOMPOSE; }

// Mirrors evaluate_state() exactly, but writes each term to d[] individually.
static void evaluate_state_decompose(const SearchState* state, int board_height,
                                      const int* queue, int queue_len,
                                      float* d) {
    memset(d, 0, sizeof(float) * NUM_DECOMPOSE);

    int upcoming[MAX_SEARCH_DEPTH + 2];
    int num_upcoming = 0;
    if (state->hold_piece != PIECE_N) upcoming[num_upcoming++] = state->hold_piece;
    for (int i = state->next_queue_idx; i < queue_len && num_upcoming < MAX_SEARCH_DEPTH + 2; i++)
        upcoming[num_upcoming++] = queue[i];

    BoardStats bs = compute_board_stats(state->board, board_height,
                                        upcoming, num_upcoming, state->col_heights);

    int max_allowed = board_height - 4;
    int effective_h = bs.max_height + state->garbage_remaining;
    if (effective_h >= max_allowed) { d[D_HEIGHT] = -1e6f; return; }

    float h_ratio = (float)effective_h / (float)max_allowed;
    float hole_mult = 1.0f + 0.5f * h_ratio;

    // SURVIVAL
    d[D_HEIGHT] = -W_HEIGHT_QUARTIC * h_ratio * h_ratio * h_ratio * h_ratio
                  - W_AVG_HEIGHT * bs.avg_height;
    if (effective_h >= max_allowed - 2) {
        int slack = max_allowed - 1 - effective_h;
        d[D_NEAR_DEATH] = -W_NEAR_DEATH * (float)(2 - slack);
    }
    d[D_BUMPINESS] = -W_BUMPINESS * bs.bumpiness_exempted;

    // HOLES
    if (bs.holes > 0) {
        int capped = bs.holes < HOLES_CAP ? bs.holes : HOLES_CAP;
        d[D_HOLES] = -W_HOLES * (float)capped * hole_mult;
    }
    if (bs.wasted_holes > 0)
        d[D_WASTED_HOLES] = -W_WASTED_HOLE * (float)bs.wasted_holes * hole_mult;
    if (bs.hole_ceiling_weight > 0.0f)
        d[D_HOLE_CEILING] = -W_HOLE_CEILING * bs.hole_ceiling_weight;
    if (bs.holes > 0) {
        float sc = bs.immobile_clearing_placements + (float)bs.t_spin_setups;
        if (sc > 0.0f)
            d[D_HOLE_FORGIVE] = W_HOLE_FORGIVE * fminf(sc, (float)bs.holes) * hole_mult;
    }

    // WELLS
    float well = 0.0f;
    if (bs.well_count == 1 && bs.well_depth >= 4 && bs.well_depth <= 8) well += 3.0f;
    else if (bs.well_count == 1 && bs.well_depth >= 2 && bs.well_depth <= 3) well += 1.0f;
    if (bs.well_count > 1) well -= 1.5f * (float)(bs.well_count - 1);
    if (bs.edge_well_depth >= 3) well -= 2.0f * (float)(bs.edge_well_depth - 2);
    d[D_WELL] = well;

    // B2B
    if (state->b2b >= 0) d[D_B2B_FLAT] = W_B2B_FLAT;
    if (state->b2b > 0) d[D_B2B_SQRT] = W_B2B_SQRT * sqrtf((float)state->b2b);
    if (state->b2b > 0) d[D_B2B_LINEAR] = W_B2B_LINEAR * (float)state->b2b;

    // COMBO + DOWNSTACK
    float combo_dscore = 0.0f;
    if (state->combo > 0) {
        int c = state->combo < COMBO_CAP ? state->combo : COMBO_CAP;
        combo_dscore = W_COMBO * (float)c;
    }
    d[D_COMBO] = combo_dscore;

    float garbage_mul = (float)state->garbage_remaining * 0.25f;
    if (garbage_mul > 1.0f) garbage_mul = 1.0f;
    if (garbage_mul < 0.0f) garbage_mul = 0.0f;

    float ds_score = 0.0f;
    if (bs.accessible_9_rows > 0) {
        int ds = bs.accessible_9_rows < DOWNSTACK_CAP ? bs.accessible_9_rows : DOWNSTACK_CAP;
        ds_score += W_DOWNSTACK * (float)ds * garbage_mul;
    }
    if (bs.well_aligned_9 > 0) {
        int w = bs.well_aligned_9 < WELL_ALIGNED_9_CAP ? bs.well_aligned_9 : WELL_ALIGNED_9_CAP;
        ds_score += W_WELL_ALIGNED_9 * (float)w * garbage_mul;
    }
    d[D_DOWNSTACK] = ds_score;

    // SPIN SETUPS (queue-capped to match evaluate_state)
    if (bs.t_spin_setups > 0 && bs.t_queue_count > 0) {
        float tr = 0.0f;
        if (bs.t_slot_quality == 2) tr = W_TSLOT;
        else if (bs.t_slot_quality == 1) tr = W_TSLOT * 0.4f;
        int usable_setups = bs.t_spin_setups < bs.t_queue_count ? bs.t_spin_setups : bs.t_queue_count;
        tr *= (1.0f + 0.3f * fminf((float)(usable_setups - 1), 2.0f));
        if (bs.t_multiline_setups > 0) {
            int usable_multi = bs.t_multiline_setups < bs.t_queue_count ? bs.t_multiline_setups : bs.t_queue_count;
            tr += W_TSPIN_MULTILINE * (float)usable_multi;
        }
        d[D_TSLOT] = tr;
    }
    {
        float lr = 0.0f;
        if (bs.immobile_clearing_placements > 0.0f) {
            lr += W_IMMOBILE_CLEAR * sqrtf(bs.immobile_clearing_placements);
            lr += W_IMMOBILE_LINES * fminf(bs.immobile_clearable_lines, 8.0f);
        }
        if (bs.future_immobile_clearing > 0.0f && state->b2b > 0) {
            // Match the double-gating in evaluate_state: garbage_mul and h_ratio
            // both modulate this term.  h_ratio is re-derived here because the
            // decompose caller doesn't share the same local.
            int max_h = 0;
            for (int c = 0; c < BOARD_COLS; c++) if (state->col_heights[c] > max_h) max_h = state->col_heights[c];
            int eff_h = max_h + state->garbage_remaining;
            int max_allowed = BOARD_ROWS - 4;
            float hr = (float)eff_h / (float)max_allowed;
            float low_stack_mul = 1.0f - hr;
            if (low_stack_mul < 0.0f) low_stack_mul = 0.0f;
            lr += W_FUTURE_B2B * bs.future_immobile_clearing * (1.0f - garbage_mul) * low_stack_mul;
        }
        if (bs.chain_rollout_length > 0 && state->b2b >= 0) {
            lr += W_CHAIN_ROLLOUT * (float)bs.chain_rollout_length;
        }
        if (lr != 0.0f) d[D_IMMOBILE_CLEAR] = lr;
    }

    // ATTACK
    if (state->max_single_attack > 0.0f)
        d[D_MAX_SINGLE] = W_MAX_SINGLE * state->max_single_attack;
    float atk = 0.0f;
    if (state->total_attack > 0.0f) atk += W_ATTACK_TOTAL * state->total_attack;
    if (state->b2b_attack > 0.0f)   atk += W_B2B_ATTACK * state->b2b_attack;
    d[D_ATTACK] = atk;

    // CASCADE / SURGE POTENTIAL / APP / BREAK READY (garbage-conditioned)
    if (bs.cascade_depth > 0) {
        int cd = bs.cascade_depth < CASCADE_CAP ? bs.cascade_depth : CASCADE_CAP;
        d[D_CASCADE] = W_CASCADE * (float)cd * garbage_mul;
    }
    if (state->b2b > 0) {
        float primed = 0.0f;
        if (bs.cascade_depth >= 1) primed = 1.0f;
        else if (bs.accessible_9_rows > 0) primed = 0.5f;
        if (primed > 0.0f) {
            int b_cap = state->b2b < 20 ? state->b2b : 20;
            d[D_SURGE_POT] = W_SURGE_POT * primed * (float)b_cap * garbage_mul;
        }
    }
    if (state->total_attack > 0.0f && state->pieces_placed > 0) {
        d[D_APP] = W_APP * (state->total_attack / (float)state->pieces_placed);
    }
    if (state->b2b >= 8 && bs.cascade_depth >= 2) {
        d[D_BREAK_READY] = W_BREAK_READY * garbage_mul;
    }
}

// Exported: enumerate depth-0 placements, return decomposed scores.
// decompose_out must hold max_placements * NUM_DECOMPOSE floats.
// Returns number of placements written.
int b2b_decompose_c(
    const uint16_t* board_rows, int board_height,
    int active_piece, int hold_piece,
    const int* queue, int queue_len,
    int b2b, int combo, int total_garbage,
    int garbage_push_delay,
    float* decompose_out, int max_placements
) {
    if (!b2b_initialized) b2b_init_pieces();

    int init_gt = (garbage_push_delay > 0) ? garbage_push_delay : 0;
    int count = 0;
    Placement placements[MAX_PLACEMENTS];
    int np;

    // Active piece placements
    np = find_placements(board_rows, board_height, active_piece, placements, MAX_PLACEMENTS, NULL);
    for (int i = 0; i < np && count < max_placements; i++) {
        Placement* pl = &placements[i];
        SearchState s;
        memset(&s, 0, sizeof(s));
        memcpy(s.board, board_rows, sizeof(uint16_t) * board_height);
        lock_piece_on_board(s.board, board_height, active_piece, pl->rot, pl->landing_row, pl->col);
        int clears = clear_lines(s.board, board_height);
        bool pc = true;
        for (int r = 0; r < board_height; r++) { if (s.board[r] != 0) { pc = false; break; } }
        AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, pc);
        s.b2b = ar.new_b2b; s.combo = ar.new_combo;
        s.total_attack = ar.attack; s.max_single_attack = ar.attack;
        s.b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
        s.max_b2b_attack = s.b2b_attack;
        s.total_lines_cleared = clears; s.hold_piece = hold_piece;
        s.pieces_placed = 1;
        s.next_queue_idx = 0; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
        s.garbage_remaining = total_garbage; s.garbage_timer = init_gt;
        compute_col_heights_full(s.board, board_height, s.col_heights);
        evaluate_state_decompose(&s, board_height, queue, queue_len,
                                  &decompose_out[count * NUM_DECOMPOSE]);
        count++;
    }

    // Hold piece placements
    if (hold_piece != 0) {
        np = find_placements(board_rows, board_height, hold_piece, placements, MAX_PLACEMENTS, NULL);
        for (int i = 0; i < np && count < max_placements; i++) {
            Placement* pl = &placements[i];
            SearchState s;
            memset(&s, 0, sizeof(s));
            memcpy(s.board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s.board, board_height, hold_piece, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s.board, board_height);
            bool pc = true;
            for (int r = 0; r < board_height; r++) { if (s.board[r] != 0) { pc = false; break; } }
            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, pc);
            s.b2b = ar.new_b2b; s.combo = ar.new_combo;
            s.total_attack = ar.attack; s.max_single_attack = ar.attack;
            s.b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s.max_b2b_attack = s.b2b_attack;
            s.total_lines_cleared = clears; s.hold_piece = active_piece;
            s.pieces_placed = 1;
            s.next_queue_idx = 0; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
            s.garbage_remaining = total_garbage; s.garbage_timer = init_gt;
            compute_col_heights_full(s.board, board_height, s.col_heights);
            evaluate_state_decompose(&s, board_height, queue, queue_len,
                                      &decompose_out[count * NUM_DECOMPOSE]);
            count++;
        }
    } else if (queue_len > 0) {
        int swap = queue[0];
        np = find_placements(board_rows, board_height, swap, placements, MAX_PLACEMENTS, NULL);
        for (int i = 0; i < np && count < max_placements; i++) {
            Placement* pl = &placements[i];
            SearchState s;
            memset(&s, 0, sizeof(s));
            memcpy(s.board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s.board, board_height, swap, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s.board, board_height);
            bool pc = true;
            for (int r = 0; r < board_height; r++) { if (s.board[r] != 0) { pc = false; break; } }
            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, pc);
            s.b2b = ar.new_b2b; s.combo = ar.new_combo;
            s.total_attack = ar.attack; s.max_single_attack = ar.attack;
            s.b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s.max_b2b_attack = s.b2b_attack;
            s.total_lines_cleared = clears; s.hold_piece = active_piece;
            s.pieces_placed = 1;
            s.next_queue_idx = 1; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
            s.garbage_remaining = total_garbage; s.garbage_timer = init_gt;
            compute_col_heights_full(s.board, board_height, s.col_heights);
            evaluate_state_decompose(&s, board_height, queue, queue_len,
                                      &decompose_out[count * NUM_DECOMPOSE]);
            count++;
        }
    }

    return count;
}

// ============================================================
// Key Sequence Reconstruction (for depth-0 move output)
// ============================================================

static void b2b_write_sequence(const BFSStateMeta* meta, int bfs_state,
                               int is_hold, int max_len, int64_t* out_row) {
    int len = 0;
    int path[BFS_STATE_SPACE];
    int curr = bfs_state;

    while (meta[curr].parent != -1) {
        path[len++] = meta[curr].last_move;
        curr = meta[curr].parent;
    }

    int p = 0;
    out_row[p++] = KEY_START;
    if (is_hold) out_row[p++] = KEY_HOLD;

    // Reserve 1 slot for HARD_DROP — truncate path if needed
    int max_path_keys = max_len - p - 1;
    int start = (len > max_path_keys) ? (len - max_path_keys) : 0;
    for (int i = len - 1; i >= start; i--) {
        out_row[p++] = path[i];
    }

    out_row[p++] = KEY_HARD_DROP; // Always fits now
    while (p < max_len) out_row[p++] = KEY_PAD;
}

// ============================================================
// Beam Search Entry Point
// ============================================================

void b2b_search_c(
    const uint16_t* board_rows,     // Board bitmasks (board_height rows)
    int board_height,               // Typically 24
    int active_piece,               // Current active piece type (1-7)
    int hold_piece,                 // Current hold piece type (0=N/empty, 1-7)
    const int* queue,               // Piece types in queue
    int queue_len,                  // Number of pieces in queue
    int b2b,                        // Current b2b counter (-1 = none)
    int combo,                      // Current combo counter (-1 = none)
    int total_garbage,              // Total garbage lines in queue
    int garbage_push_delay,         // Ticks until garbage pushes (0 = immediate)
    int bag_seen_init,              // Bitmask: pieces consumed from current bag (after queue)
    int search_depth,               // Max search depth
    int beam_width,                 // Beam width
    int max_len,                    // Max key sequence length
    int* out_action_index,          // Output: action index
    int64_t* out_best_sequence      // Output: key sequence (length max_len)
) {
    if (!b2b_initialized) b2b_init_pieces();
    if (!zobrist_initialized) zobrist_init();
    tt_new_generation();

    // Clamp parameters
    if (search_depth > MAX_SEARCH_DEPTH) search_depth = MAX_SEARCH_DEPTH;
    if (beam_width > MAX_BEAM_WIDTH) beam_width = MAX_BEAM_WIDTH;
    if (search_depth < 1) search_depth = 1;

    uint8_t initial_bag_seen = (uint8_t)(bag_seen_init & 0xFF);

    // Allocate beam arrays (current and next)
    // Use heap allocation since these can be large
    // Extra capacity when speculative depths are active (multiple pieces per beam state)
    int spec_mult = (search_depth > queue_len + 1) ? 2 : 1;
    int max_next = beam_width * MAX_PLACEMENTS * spec_mult;
    if (max_next > MAX_BEAM_WIDTH * MAX_PLACEMENTS * 2) max_next = MAX_BEAM_WIDTH * MAX_PLACEMENTS * 2;
    SearchState* curr_beam = (SearchState*)malloc(max_next * sizeof(SearchState));
    SearchState* next_beam = (SearchState*)malloc(max_next * sizeof(SearchState));
    int curr_beam_size = 0;
    int next_beam_size = 0;

    // Hash table for per-level dedupe (power-of-2 size, load factor <=0.5)
    int hash_cap = 1;
    while (hash_cap < 2 * max_next) hash_cap <<= 1;
    HashSlot* hash_table = (HashSlot*)malloc(hash_cap * sizeof(HashSlot));

    if (!curr_beam || !next_beam || !hash_table) {
        // Allocation failed
        *out_action_index = -1;
        for (int i = 0; i < max_len; i++) out_best_sequence[i] = KEY_PAD;
        free(curr_beam);
        free(next_beam);
        free(hash_table);
        return;
    }

    // Store depth-0 BFS meta and placement info for sequence reconstruction
    static BFSStateMeta depth0_meta_active[BFS_STATE_SPACE];
    static BFSStateMeta depth0_meta_hold[BFS_STATE_SPACE];
    static Placement depth0_placements[MAX_PLACEMENTS * 2];
    static int depth0_is_hold[MAX_PLACEMENTS * 2];
    int depth0_count = 0;

    // Initial garbage timer: the caller tells us the push delay.  A value
    // of 0 means garbage pushes immediately on the first non-clearing step;
    // 1 means it waits one tick, etc.  We use the delay as the initial
    // countdown for the simulated garbage timer in the search state.
    int init_garbage_timer = (garbage_push_delay > 0) ? garbage_push_delay : 0;

    // Precompute the root board's column heights once so every depth-0
    // placement can incrementally patch from it.
    int8_t root_col_heights[BOARD_COLS];
    compute_col_heights_full(board_rows, board_height, root_col_heights);

    // ---- Depth 0: enumerate placements for active piece and hold piece ----

    Placement placements[MAX_PLACEMENTS];
    int np;

    // Active piece placements
    np = find_placements(board_rows, board_height, active_piece, placements, MAX_PLACEMENTS,
                         depth0_meta_active);

    for (int i = 0; i < np && depth0_count < MAX_PLACEMENTS * 2; i++) {
        Placement* pl = &placements[i];
        SearchState* s = &next_beam[next_beam_size];

        memcpy(s->board, board_rows, sizeof(uint16_t) * board_height);
        lock_piece_on_board(s->board, board_height, active_piece, pl->rot, pl->landing_row, pl->col);
        int clears = clear_lines(s->board, board_height);

        bool perfect_clear = true;
        for (int r = 0; r < board_height; r++) {
            if (s->board[r] != 0) { perfect_clear = false; break; }
        }

        AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, perfect_clear);

        s->b2b = ar.new_b2b;
        s->combo = ar.new_combo;
        s->total_attack = ar.attack;
        s->max_single_attack = ar.attack;
        s->b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
        s->max_b2b_attack = s->b2b_attack;
        s->total_lines_cleared = clears;
        s->pieces_placed = 1;
        s->hold_piece = hold_piece;
        s->next_queue_idx = 0;
        s->depth0_placement_idx = depth0_count;
        s->b2b_broken = ar.b2b_broken;
        s->prev_b2b = b2b;
        s->bag_seen = initial_bag_seen;

        bool pushed_garbage = false;
        {
            int gr = total_garbage;
            int gt = init_garbage_timer;
            if (ar.attack > 0 && gr > 0) {
                int cancel = (int)ar.attack;
                gr = (gr > cancel) ? gr - cancel : 0;
            }
            if (clears == 0 && gr > 0) {
                if (gt <= 0) {
                    push_simulated_garbage(s->board, board_height, gr);
                    gr = 0;
                    pushed_garbage = true;
                } else {
                    gt--;
                }
            }
            s->garbage_remaining = gr;
            s->garbage_timer = gt;
        }

        if (clears == 0 && !pushed_garbage) {
            patch_col_heights_after_place(root_col_heights, active_piece, pl->rot,
                                          pl->landing_row, pl->col, board_height,
                                          s->col_heights);
        } else {
            compute_col_heights_full(s->board, board_height, s->col_heights);
        }

        if (placement_is_dead(s, board_height)) { continue; }
                s->score = evaluate_state(s, board_height, queue, queue_len);

        depth0_placements[depth0_count] = *pl;
        depth0_is_hold[depth0_count] = 0;
        depth0_count++;
        next_beam_size++;
    }

    // Hold piece placements
    if (hold_piece != PIECE_N) {
        // Swap: play hold piece, hold becomes active
        np = find_placements(board_rows, board_height, hold_piece, placements, MAX_PLACEMENTS,
                             depth0_meta_hold);

        for (int i = 0; i < np && depth0_count < MAX_PLACEMENTS * 2; i++) {
            Placement* pl = &placements[i];
            SearchState* s = &next_beam[next_beam_size];

            memcpy(s->board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s->board, board_height, hold_piece, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s->board, board_height);

            bool perfect_clear = true;
            for (int r = 0; r < board_height; r++) {
                if (s->board[r] != 0) { perfect_clear = false; break; }
            }

            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, perfect_clear);

            s->b2b = ar.new_b2b;
            s->combo = ar.new_combo;
            s->total_attack = ar.attack;
            s->max_single_attack = ar.attack;
            s->b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s->max_b2b_attack = s->b2b_attack;
            s->total_lines_cleared = clears;
            s->pieces_placed = 1;
            s->hold_piece = active_piece;
            s->next_queue_idx = 0;
            s->depth0_placement_idx = depth0_count;
            s->b2b_broken = ar.b2b_broken;
            s->prev_b2b = b2b;
            s->bag_seen = initial_bag_seen;

            bool pushed_garbage = false;
            {
                int gr = total_garbage;
                int gt = init_garbage_timer;
                if (ar.attack > 0 && gr > 0) {
                    int cancel = (int)ar.attack;
                    gr = (gr > cancel) ? gr - cancel : 0;
                }
                if (clears == 0 && gr > 0) {
                    if (gt <= 0) {
                        push_simulated_garbage(s->board, board_height, gr);
                        gr = 0;
                        pushed_garbage = true;
                    } else {
                        gt--;
                    }
                }
                s->garbage_remaining = gr;
                s->garbage_timer = gt;
            }

            if (clears == 0 && !pushed_garbage) {
                patch_col_heights_after_place(root_col_heights, hold_piece, pl->rot,
                                              pl->landing_row, pl->col, board_height,
                                              s->col_heights);
            } else {
                compute_col_heights_full(s->board, board_height, s->col_heights);
            }

            if (placement_is_dead(s, board_height)) { continue; }
                s->score = evaluate_state(s, board_height, queue, queue_len);

            depth0_placements[depth0_count] = *pl;
            depth0_is_hold[depth0_count] = 1;
            depth0_count++;
            next_beam_size++;
        }
    } else if (queue_len > 0) {
        // No hold piece yet — hold swaps active with first queue piece
        int swap_piece = queue[0];
        np = find_placements(board_rows, board_height, swap_piece, placements, MAX_PLACEMENTS,
                             depth0_meta_hold);

        for (int i = 0; i < np && depth0_count < MAX_PLACEMENTS * 2; i++) {
            Placement* pl = &placements[i];
            SearchState* s = &next_beam[next_beam_size];

            memcpy(s->board, board_rows, sizeof(uint16_t) * board_height);
            lock_piece_on_board(s->board, board_height, swap_piece, pl->rot, pl->landing_row, pl->col);
            int clears = clear_lines(s->board, board_height);

            bool perfect_clear = true;
            for (int r = 0; r < board_height; r++) {
                if (s->board[r] != 0) { perfect_clear = false; break; }
            }

            AttackResult ar = compute_attack(clears, pl->spin_type, b2b, combo, perfect_clear);

            s->b2b = ar.new_b2b;
            s->combo = ar.new_combo;
            s->total_attack = ar.attack;
            s->max_single_attack = ar.attack;
            s->b2b_attack = ar.b2b_maintaining ? ar.attack : 0.0f;
            s->max_b2b_attack = s->b2b_attack;
            s->total_lines_cleared = clears;
            s->pieces_placed = 1;
            s->hold_piece = active_piece;
            s->next_queue_idx = 1;
            s->depth0_placement_idx = depth0_count;
            s->b2b_broken = ar.b2b_broken;
            s->prev_b2b = b2b;
            s->bag_seen = initial_bag_seen;

            bool pushed_garbage = false;
            {
                int gr = total_garbage;
                int gt = init_garbage_timer;
                if (ar.attack > 0 && gr > 0) {
                    int cancel = (int)ar.attack;
                    gr = (gr > cancel) ? gr - cancel : 0;
                }
                if (clears == 0 && gr > 0) {
                    if (gt <= 0) {
                        push_simulated_garbage(s->board, board_height, gr);
                        gr = 0;
                        pushed_garbage = true;
                    } else {
                        gt--;
                    }
                }
                s->garbage_remaining = gr;
                s->garbage_timer = gt;
            }

            if (clears == 0 && !pushed_garbage) {
                patch_col_heights_after_place(root_col_heights, swap_piece, pl->rot,
                                              pl->landing_row, pl->col, board_height,
                                              s->col_heights);
            } else {
                compute_col_heights_full(s->board, board_height, s->col_heights);
            }

            if (placement_is_dead(s, board_height)) { continue; }
                s->score = evaluate_state(s, board_height, queue, queue_len);

            depth0_placements[depth0_count] = *pl;
            depth0_is_hold[depth0_count] = 1;
            depth0_count++;
            next_beam_size++;
        }
    }

    // Dedupe, then top-K select.  (Dedupe at depth 0 is safe because two
    // placements that collapse to the same post-state have identical futures
    // — picking either depth0 choice produces the same end state.)
    next_beam_size = dedupe_beam(next_beam, next_beam_size, board_height,
                                 hash_table, hash_cap);
    next_beam_size = beam_select_top_k(next_beam, next_beam_size, beam_width);

    // Swap beams
    {
        SearchState* tmp = curr_beam;
        curr_beam = next_beam;
        curr_beam_size = next_beam_size;
        next_beam = tmp;
        next_beam_size = 0;
    }

    // ---- Depths 1..search_depth-1 ----
    for (int depth = 1; depth < search_depth; depth++) {
        next_beam_size = 0;
        topk_reset(beam_width);

        for (int bi = 0; bi < curr_beam_size; bi++) {
            SearchState* parent = &curr_beam[bi];

            int qi = parent->next_queue_idx;
            if (qi >= queue_len) {
                // ---- Speculative: branch on remaining bag pieces ----
                int remaining[7];
                int n_remaining = bag_get_remaining(parent->bag_seen, remaining);

                if (n_remaining == 0) {
                    // All pieces consumed — shouldn't happen, keep as leaf
                    if (next_beam_size < max_next) {
                        next_beam[next_beam_size] = *parent;
                        next_beam_size++;
                    }
                    continue;
                }

                for (int pi = 0; pi < n_remaining; pi++) {
                    int spec_piece = remaining[pi];
                    uint8_t new_bag = bag_consume_piece(parent->bag_seen, spec_piece);

                    // Try placing speculative piece directly
                    np = find_placements(parent->board, board_height, spec_piece, placements, MAX_PLACEMENTS, NULL);

                    for (int i = 0; i < np && next_beam_size < max_next; i++) {
                        Placement* pl = &placements[i];
                        SearchState* s = &next_beam[next_beam_size];

                        memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                        lock_piece_on_board(s->board, board_height, spec_piece, pl->rot, pl->landing_row, pl->col);
                        int clears = clear_lines(s->board, board_height);

                        bool perfect_clear = true;
                        for (int r = 0; r < board_height; r++) {
                            if (s->board[r] != 0) { perfect_clear = false; break; }
                        }

                        AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                        s->b2b = ar.new_b2b;
                        s->combo = ar.new_combo;
                        s->total_attack = parent->total_attack + ar.attack;
                        s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                        { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                          s->b2b_attack = parent->b2b_attack + ba;
                          s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                        s->total_lines_cleared = parent->total_lines_cleared + clears;
                        s->pieces_placed = parent->pieces_placed + 1;
                        s->hold_piece = parent->hold_piece;
                        s->next_queue_idx = qi + 1;
                        s->depth0_placement_idx = parent->depth0_placement_idx;
                        s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                        s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                        s->bag_seen = new_bag;

                        bool pushed_garbage = false;
                        {
                            int gr = parent->garbage_remaining;
                            int gt = parent->garbage_timer;
                            if (ar.attack > 0 && gr > 0) {
                                int cancel = (int)ar.attack;
                                gr = (gr > cancel) ? gr - cancel : 0;
                            }
                            if (clears == 0 && gr > 0) {
                                if (gt <= 0) {
                                    push_simulated_garbage(s->board, board_height, gr);
                                    gr = 0;
                                    pushed_garbage = true;
                                } else {
                                    gt--;
                                }
                            }
                            s->garbage_remaining = gr;
                            s->garbage_timer = gt;
                        }

                        if (clears == 0 && !pushed_garbage) {
                            patch_col_heights_after_place(parent->col_heights, spec_piece, pl->rot,
                                                          pl->landing_row, pl->col, board_height,
                                                          s->col_heights);
                        } else {
                            compute_col_heights_full(s->board, board_height, s->col_heights);
                        }

                        if (placement_is_dead(s, board_height)) { continue; }
                        {
                            float _cheap = cheap_prescore(s, board_height);
                            float _floor = topk_floor();
                            if (_floor > -1e8f && _cheap + ASPIRATION_SLACK < _floor) { continue; }
                            s->score = evaluate_state(s, board_height, queue, queue_len);
                            topk_insert(s->score);
                        }

                        next_beam_size++;
                    }

                    // Try hold swap (if hold piece exists)
                    if (parent->hold_piece != PIECE_N) {
                        int held = parent->hold_piece;
                        np = find_placements(parent->board, board_height, held, placements, MAX_PLACEMENTS, NULL);

                        for (int i = 0; i < np && next_beam_size < max_next; i++) {
                            Placement* pl = &placements[i];
                            SearchState* s = &next_beam[next_beam_size];

                            memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                            lock_piece_on_board(s->board, board_height, held, pl->rot, pl->landing_row, pl->col);
                            int clears = clear_lines(s->board, board_height);

                            bool perfect_clear = true;
                            for (int r = 0; r < board_height; r++) {
                                if (s->board[r] != 0) { perfect_clear = false; break; }
                            }

                            AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                            s->b2b = ar.new_b2b;
                            s->combo = ar.new_combo;
                            s->total_attack = parent->total_attack + ar.attack;
                            s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                            { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                              s->b2b_attack = parent->b2b_attack + ba;
                              s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                            s->total_lines_cleared = parent->total_lines_cleared + clears;
                            s->pieces_placed = parent->pieces_placed + 1;
                            s->hold_piece = spec_piece; // speculative piece goes to hold
                            s->next_queue_idx = qi + 1;
                            s->depth0_placement_idx = parent->depth0_placement_idx;
                            s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                            s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                            s->bag_seen = new_bag;

                            bool pushed_garbage = false;
                            {
                                int gr = parent->garbage_remaining;
                                int gt = parent->garbage_timer;
                                if (ar.attack > 0 && gr > 0) {
                                    int cancel = (int)ar.attack;
                                    gr = (gr > cancel) ? gr - cancel : 0;
                                }
                                if (clears == 0 && gr > 0) {
                                    if (gt <= 0) {
                                        push_simulated_garbage(s->board, board_height, gr);
                                        gr = 0;
                                        pushed_garbage = true;
                                    } else {
                                        gt--;
                                    }
                                }
                                s->garbage_remaining = gr;
                                s->garbage_timer = gt;
                            }

                            if (clears == 0 && !pushed_garbage) {
                                patch_col_heights_after_place(parent->col_heights, held, pl->rot,
                                                              pl->landing_row, pl->col, board_height,
                                                              s->col_heights);
                            } else {
                                compute_col_heights_full(s->board, board_height, s->col_heights);
                            }

                            if (placement_is_dead(s, board_height)) { continue; }
                            {
                                float _cheap = cheap_prescore(s, board_height);
                                float _floor = topk_floor();
                                if (_floor > -1e8f && _cheap + ASPIRATION_SLACK < _floor) { continue; }
                                s->score = evaluate_state(s, board_height, queue, queue_len);
                                topk_insert(s->score);
                            }

                            next_beam_size++;
                        }
                    }
                }
                continue;
            }

            int piece = queue[qi];

            // Try placing active piece (no hold)
            np = find_placements(parent->board, board_height, piece, placements, MAX_PLACEMENTS, NULL);

            for (int i = 0; i < np && next_beam_size < max_next; i++) {
                Placement* pl = &placements[i];
                SearchState* s = &next_beam[next_beam_size];

                memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                lock_piece_on_board(s->board, board_height, piece, pl->rot, pl->landing_row, pl->col);
                int clears = clear_lines(s->board, board_height);

                bool perfect_clear = true;
                for (int r = 0; r < board_height; r++) {
                    if (s->board[r] != 0) { perfect_clear = false; break; }
                }

                AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                s->b2b = ar.new_b2b;
                s->combo = ar.new_combo;
                s->total_attack = parent->total_attack + ar.attack;
                s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                  s->b2b_attack = parent->b2b_attack + ba;
                  s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                s->total_lines_cleared = parent->total_lines_cleared + clears;
                s->pieces_placed = parent->pieces_placed + 1;
                s->hold_piece = parent->hold_piece;
                s->next_queue_idx = qi + 1;
                s->depth0_placement_idx = parent->depth0_placement_idx;
                s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                s->bag_seen = parent->bag_seen;

                bool pushed_garbage = false;
                {
                    int gr = parent->garbage_remaining;
                    int gt = parent->garbage_timer;
                    if (ar.attack > 0 && gr > 0) {
                        int cancel = (int)ar.attack;
                        gr = (gr > cancel) ? gr - cancel : 0;
                    }
                    if (clears == 0 && gr > 0) {
                        if (gt <= 0) {
                            push_simulated_garbage(s->board, board_height, gr);
                            gr = 0;
                            pushed_garbage = true;
                        } else {
                            gt--;
                        }
                    }
                    s->garbage_remaining = gr;
                    s->garbage_timer = gt;
                }

                if (clears == 0 && !pushed_garbage) {
                    patch_col_heights_after_place(parent->col_heights, piece, pl->rot,
                                                  pl->landing_row, pl->col, board_height,
                                                  s->col_heights);
                } else {
                    compute_col_heights_full(s->board, board_height, s->col_heights);
                }

                if (placement_is_dead(s, board_height)) { continue; }
                {
                    float _cheap = cheap_prescore(s, board_height);
                    float _floor = topk_floor();
                    if (_floor > -1e8f && _cheap + ASPIRATION_SLACK < _floor) { continue; }
                    s->score = evaluate_state(s, board_height, queue, queue_len);
                    topk_insert(s->score);
                }

                next_beam_size++;
            }

            // Try hold swap
            if (parent->hold_piece != PIECE_N) {
                int held = parent->hold_piece;
                np = find_placements(parent->board, board_height, held, placements, MAX_PLACEMENTS, NULL);

                for (int i = 0; i < np && next_beam_size < max_next; i++) {
                    Placement* pl = &placements[i];
                    SearchState* s = &next_beam[next_beam_size];

                    memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                    lock_piece_on_board(s->board, board_height, held, pl->rot, pl->landing_row, pl->col);
                    int clears = clear_lines(s->board, board_height);

                    bool perfect_clear = true;
                    for (int r = 0; r < board_height; r++) {
                        if (s->board[r] != 0) { perfect_clear = false; break; }
                    }

                    AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                    s->b2b = ar.new_b2b;
                    s->combo = ar.new_combo;
                    s->total_attack = parent->total_attack + ar.attack;
                    s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                    { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                      s->b2b_attack = parent->b2b_attack + ba;
                      s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                    s->total_lines_cleared = parent->total_lines_cleared + clears;
                    s->pieces_placed = parent->pieces_placed + 1;
                    s->hold_piece = piece;
                    s->next_queue_idx = qi + 1;
                    s->depth0_placement_idx = parent->depth0_placement_idx;
                    s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                    s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                    s->bag_seen = parent->bag_seen;

                    bool pushed_garbage = false;
                    {
                        int gr = parent->garbage_remaining;
                        int gt = parent->garbage_timer;
                        if (ar.attack > 0 && gr > 0) {
                            int cancel = (int)ar.attack;
                            gr = (gr > cancel) ? gr - cancel : 0;
                        }
                        if (clears == 0 && gr > 0) {
                            if (gt <= 0) {
                                push_simulated_garbage(s->board, board_height, gr);
                                gr = 0;
                                pushed_garbage = true;
                            } else {
                                gt--;
                            }
                        }
                        s->garbage_remaining = gr;
                        s->garbage_timer = gt;
                    }

                    if (clears == 0 && !pushed_garbage) {
                        patch_col_heights_after_place(parent->col_heights, held, pl->rot,
                                                      pl->landing_row, pl->col, board_height,
                                                      s->col_heights);
                    } else {
                        compute_col_heights_full(s->board, board_height, s->col_heights);
                    }

                    if (placement_is_dead(s, board_height)) { continue; }
                    {
                        float _cheap = cheap_prescore(s, board_height);
                        float _floor = topk_floor();
                        if (_floor > -1e8f && _cheap + ASPIRATION_SLACK < _floor) { continue; }
                        s->score = evaluate_state(s, board_height, queue, queue_len);
                        topk_insert(s->score);
                    }

                    next_beam_size++;
                }
            } else if (qi + 1 < queue_len) {
                // No hold piece — holding puts active into hold, play next from queue
                int next_piece = piece; // Goes to hold
                int play_piece_idx = qi + 1;
                int play_piece = queue[play_piece_idx];

                np = find_placements(parent->board, board_height, play_piece, placements, MAX_PLACEMENTS, NULL);

                for (int i = 0; i < np && next_beam_size < max_next; i++) {
                    Placement* pl = &placements[i];
                    SearchState* s = &next_beam[next_beam_size];

                    memcpy(s->board, parent->board, sizeof(uint16_t) * board_height);
                    lock_piece_on_board(s->board, board_height, play_piece, pl->rot, pl->landing_row, pl->col);
                    int clears = clear_lines(s->board, board_height);

                    bool perfect_clear = true;
                    for (int r = 0; r < board_height; r++) {
                        if (s->board[r] != 0) { perfect_clear = false; break; }
                    }

                    AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

                    s->b2b = ar.new_b2b;
                    s->combo = ar.new_combo;
                    s->total_attack = parent->total_attack + ar.attack;
                    s->max_single_attack = ar.attack > parent->max_single_attack ? ar.attack : parent->max_single_attack;
                    { float ba = ar.b2b_maintaining ? ar.attack : 0.0f;
                      s->b2b_attack = parent->b2b_attack + ba;
                      s->max_b2b_attack = ba > parent->max_b2b_attack ? ba : parent->max_b2b_attack; }
                    s->total_lines_cleared = parent->total_lines_cleared + clears;
                    s->pieces_placed = parent->pieces_placed + 1;
                    s->hold_piece = next_piece;
                    s->next_queue_idx = play_piece_idx + 1;
                    s->depth0_placement_idx = parent->depth0_placement_idx;
                    s->b2b_broken = parent->b2b_broken || ar.b2b_broken;
                    s->prev_b2b = parent->b2b_broken ? parent->prev_b2b : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
                    s->bag_seen = parent->bag_seen;

                    bool pushed_garbage = false;
                    {
                        int gr = parent->garbage_remaining;
                        int gt = parent->garbage_timer;
                        if (ar.attack > 0 && gr > 0) {
                            int cancel = (int)ar.attack;
                            gr = (gr > cancel) ? gr - cancel : 0;
                        }
                        if (clears == 0 && gr > 0) {
                            if (gt <= 0) {
                                push_simulated_garbage(s->board, board_height, gr);
                                gr = 0;
                                pushed_garbage = true;
                            } else {
                                gt--;
                            }
                        }
                        s->garbage_remaining = gr;
                        s->garbage_timer = gt;
                    }

                    if (clears == 0 && !pushed_garbage) {
                        patch_col_heights_after_place(parent->col_heights, play_piece, pl->rot,
                                                      pl->landing_row, pl->col, board_height,
                                                      s->col_heights);
                    } else {
                        compute_col_heights_full(s->board, board_height, s->col_heights);
                    }

                    if (placement_is_dead(s, board_height)) { continue; }
                    {
                        float _cheap = cheap_prescore(s, board_height);
                        float _floor = topk_floor();
                        if (_floor > -1e8f && _cheap + ASPIRATION_SLACK < _floor) { continue; }
                        s->score = evaluate_state(s, board_height, queue, queue_len);
                        topk_insert(s->score);
                    }

                    next_beam_size++;
                }
            }
        }

        // Dedupe, then top-K select.  At depth>=1 dedupe is especially
        // high-value: many (parent, placement) pairs collapse onto the same
        // post-state, which would otherwise multiply through expansion.
        next_beam_size = dedupe_beam(next_beam, next_beam_size, board_height,
                                     hash_table, hash_cap);
        next_beam_size = beam_select_top_k(next_beam, next_beam_size, beam_width);

        // Swap beams
        {
            SearchState* tmp = curr_beam;
            curr_beam = next_beam;
            curr_beam_size = next_beam_size;
            next_beam = tmp;
            next_beam_size = 0;
        }
    }

    // ---- Extract best result ----
    if (curr_beam_size == 0) {
        *out_action_index = -1;
        for (int i = 0; i < max_len; i++) out_best_sequence[i] = KEY_PAD;
        free(curr_beam);
        free(next_beam);
        free(hash_table);
        return;
    }

    // Find best state
    int best_idx = 0;
    for (int i = 1; i < curr_beam_size; i++) {
        if (curr_beam[i].score > curr_beam[best_idx].score) best_idx = i;
    }

    int d0_idx = curr_beam[best_idx].depth0_placement_idx;
    Placement* best_pl = &depth0_placements[d0_idx];
    int is_hold = depth0_is_hold[d0_idx];

    // Compute action index: hold * 160 + rot * 40 + norm_col * 4 + spin_type
    int played_piece;
    if (!is_hold) {
        played_piece = active_piece;
    } else {
        played_piece = (hold_piece != PIECE_N) ? hold_piece : queue[0];
    }
    int norm_col = best_pl->col + B2B_PIECES[played_piece].orientations[best_pl->rot].min_col;
    *out_action_index = is_hold * 160 + best_pl->rot * 40 + norm_col * 4 + best_pl->spin_type;

    // Store full placement info for C game loop (avoids lossy action_idx round-trip)
    b2b_last_placement.rot = best_pl->rot;
    b2b_last_placement.col = best_pl->col;
    b2b_last_placement.landing_row = best_pl->landing_row;
    b2b_last_placement.spin_type = best_pl->spin_type;

    // Reconstruct key sequence
    BFSStateMeta* meta_src = is_hold ? depth0_meta_hold : depth0_meta_active;
    b2b_write_sequence(meta_src, best_pl->bfs_state, is_hold, max_len, out_best_sequence);

    free(curr_beam);
    free(next_beam);
    free(hash_table);
}

// ============================================================
// Full Game Loop in C (for optimizer — no Python overhead)
// ============================================================

// --- TetrioRNG: exact replica of TetrioRandom.py ---

typedef struct {
    int64_t t;
} TetrioRNG;

static void rng_init(TetrioRNG* rng, int seed) {
    int64_t t = seed % 2147483647;
    if (t <= 0) t += 2147483646;
    rng->t = t;
}

static int rng_next_int(TetrioRNG* rng) {
    rng->t = (16807 * rng->t) % 2147483647;
    return (int)rng->t;
}

static float rng_next_float(TetrioRNG* rng) {
    return (float)(rng_next_int(rng) - 1) / 2147483646.0f;
}

// Tetromino order matching Python: [Z, L, O, S, I, J, T]
static void rng_next_bag(TetrioRNG* rng, int* bag) {
    int base[7] = {PIECE_Z, PIECE_L, PIECE_O, PIECE_S,
                   PIECE_I, PIECE_J, PIECE_T};
    memcpy(bag, base, sizeof(base));
    // Fisher-Yates shuffle (matching Python's while i > 0 loop)
    for (int i = 6; i > 0; i--) {
        int j = (int)(rng_next_float(rng) * (i + 1));
        int tmp = bag[i]; bag[i] = bag[j]; bag[j] = tmp;
    }
}

// --- Simple xorshift RNG for garbage generation ---

typedef struct {
    uint32_t state;
} SimpleRNG;

static void srng_init(SimpleRNG* rng, uint32_t seed) {
    rng->state = seed ? seed : 1;
}

static uint32_t srng_next(SimpleRNG* rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static float srng_float(SimpleRNG* rng) {
    return (float)(srng_next(rng) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

// --- Piece queue (bag-based) ---

#define BAG_SIZE 7
#define MAX_QUEUE 16

typedef struct {
    TetrioRNG rng;
    int bag[BAG_SIZE];
    int bag_pos;
    int queue[MAX_QUEUE];
    int queue_len;
    int queue_size;
} PieceQueue;

static void pq_init(PieceQueue* pq, int seed, int qsize) {
    rng_init(&pq->rng, seed);
    pq->bag_pos = BAG_SIZE; // force first bag generation
    pq->queue_len = 0;
    pq->queue_size = qsize;
}

static int pq_next_piece(PieceQueue* pq) {
    if (pq->bag_pos >= BAG_SIZE) {
        rng_next_bag(&pq->rng, pq->bag);
        pq->bag_pos = 0;
    }
    return pq->bag[pq->bag_pos++];
}

static void pq_fill(PieceQueue* pq) {
    while (pq->queue_len < pq->queue_size) {
        pq->queue[pq->queue_len++] = pq_next_piece(pq);
    }
}

static int pq_pop(PieceQueue* pq) {
    if (pq->queue_len <= 0) return PIECE_N;
    int piece = pq->queue[0];
    for (int i = 0; i < pq->queue_len - 1; i++)
        pq->queue[i] = pq->queue[i + 1];
    pq->queue_len--;
    return piece;
}

// --- Garbage queue ---

#define MAX_GARB_ENTRIES 32

typedef struct {
    int rows;
    int col;
    int timer;
} GarbEntry;

static void garb_cancel(GarbEntry* gq, int* cnt, int attack) {
    int rem = attack;
    while (rem > 0 && *cnt > 0) {
        if (gq[0].rows <= rem) {
            rem -= gq[0].rows;
            for (int i = 0; i < *cnt - 1; i++) gq[i] = gq[i + 1];
            (*cnt)--;
        } else {
            gq[0].rows -= rem;
            rem = 0;
        }
    }
}

static void garb_tick(GarbEntry* gq, int cnt) {
    for (int i = 0; i < cnt; i++) {
        if (gq[i].timer > 0) gq[i].timer--;
    }
}

static bool garb_push_one(uint16_t* board, int bh, GarbEntry* gq, int* cnt) {
    if (*cnt <= 0 || gq[0].timer > 0) return false;
    uint16_t full = (1 << BOARD_COLS) - 1;
    int rows = gq[0].rows;
    int col = gq[0].col;
    for (int r = 0; r < bh - rows; r++) board[r] = board[r + rows];
    uint16_t garb_row = full & ~(1 << col);
    for (int r = bh - rows; r < bh; r++) board[r] = garb_row;
    for (int i = 0; i < *cnt - 1; i++) gq[i] = gq[i + 1];
    (*cnt)--;
    return true;
}

static void garb_push_all(uint16_t* board, int bh, GarbEntry* gq, int* cnt) {
    while (garb_push_one(board, bh, gq, cnt)) {}
}

static int garb_total(const GarbEntry* gq, int cnt) {
    int t = 0;
    for (int i = 0; i < cnt; i++) t += gq[i].rows;
    return t;
}

// --- Game config & result structs (exported) ---

typedef struct {
    int seed;
    float garbage_chance;
    int garbage_min;
    int garbage_max;
    int garbage_push_delay;
} GameConfig;

typedef struct {
    int steps_completed;
    int survived;
    float total_attack;
    int max_b2b;
    int end_height;
    float avg_height;
    int max_height;
} GameResult;

// --- Game loop entry point ---

void b2b_run_eval_games(
    int num_games,
    const GameConfig* configs,
    int num_steps,
    int search_depth,
    int beam_width,
    int queue_size,
    GameResult* results
) {
    if (!b2b_initialized) b2b_init_pieces();

    for (int g = 0; g < num_games; g++) {
        GameConfig cfg = configs[g];
        GameResult* res = &results[g];

        // --- Init game state ---
        uint16_t board[BOARD_ROWS];
        memset(board, 0, sizeof(board));
        int bh = 24; // standard board height

        PieceQueue pq;
        pq_init(&pq, cfg.seed, queue_size);
        pq_fill(&pq);
        int active_piece = pq_pop(&pq);
        pq_fill(&pq);
        int hold_piece = PIECE_N;
        int b2b = -1, combo = -1;

        SimpleRNG grng;
        srng_init(&grng, (uint32_t)(cfg.seed * 7 + 12345));
        GarbEntry gq[MAX_GARB_ENTRIES];
        int gcnt = 0;

        float total_attack = 0.0f;
        int max_b2b_val = 0;
        float height_sum = 0.0f;
        int last_height = 0, peak_height = 0;
        int steps_done = 0;
        bool died = false;

        for (int step = 0; step < num_steps; step++) {
            if (b2b > max_b2b_val) max_b2b_val = b2b;

            int tg = garb_total(gq, gcnt);

            // --- Compute bag_seen for speculative search ---
            // pq.bag[0..bag_pos-1] are pieces already drawn from current bag
            uint8_t cur_bag_seen = 0;
            if (pq.bag_pos < BAG_SIZE) {
                for (int bi2 = 0; bi2 < pq.bag_pos; bi2++) {
                    cur_bag_seen |= (uint8_t)(1 << pq.bag[bi2]);
                }
            }
            // If bag_pos >= BAG_SIZE, cur_bag_seen stays 0 (fresh bag)

            // --- Beam search (reuses b2b_search_c — sequence is discarded) ---
            int action_idx;
            int64_t dummy_seq[15];
            b2b_search_c(
                board, bh, active_piece, hold_piece,
                pq.queue, pq.queue_len, b2b, combo, tg,
                cfg.garbage_push_delay,
                (int)cur_bag_seen,
                search_depth, beam_width, 15,
                &action_idx, dummy_seq
            );

            if (action_idx < 0) { died = true; break; }

            // --- Read placement directly from b2b_search_c's result ---
            // (avoids lossy action_idx round-trip — preserves BFS landing row)
            int is_hold   = action_idx / 160;
            int rot       = b2b_last_placement.rot;
            int col       = b2b_last_placement.col;
            int lr        = b2b_last_placement.landing_row;
            int spin_type = b2b_last_placement.spin_type;

            // Determine played piece & update hold/queue
            int played;
            if (!is_hold) {
                played = active_piece;
                active_piece = pq_pop(&pq);
            } else if (hold_piece != PIECE_N) {
                played = hold_piece;
                hold_piece = active_piece;
                active_piece = pq_pop(&pq);
            } else {
                hold_piece = active_piece;
                played = pq_pop(&pq);
                active_piece = pq_pop(&pq);
            }
            pq_fill(&pq);

            // --- Apply placement ---
            lock_piece_on_board(board, bh, played, rot, lr, col);
            int clears = clear_lines(board, bh);

            bool pc = true;
            for (int r = 0; r < bh; r++) {
                if (board[r] != 0) { pc = false; break; }
            }

            AttackResult ar = compute_attack(clears, spin_type, b2b, combo, pc);
            total_attack += ar.attack;
            b2b = ar.new_b2b;
            combo = ar.new_combo;

            // --- Garbage handling ---
            if (ar.attack > 0) garb_cancel(gq, &gcnt, (int)ar.attack);

            if (clears == 0) {
                garb_tick(gq, gcnt);
                garb_push_one(board, bh, gq, &gcnt); // one tier per step
            }

            // Generate new garbage
            if (cfg.garbage_chance > 0.0f && cfg.garbage_max > 0) {
                if (srng_float(&grng) <= cfg.garbage_chance) {
                    int nr;
                    if (cfg.garbage_min == cfg.garbage_max) {
                        nr = cfg.garbage_min;
                    } else {
                        nr = cfg.garbage_min +
                             (int)(srng_float(&grng) *
                                   (cfg.garbage_max - cfg.garbage_min + 1));
                        if (nr > cfg.garbage_max) nr = cfg.garbage_max;
                    }
                    if (nr > 0 && gcnt < MAX_GARB_ENTRIES) {
                        int gap = (int)(srng_float(&grng) * BOARD_COLS);
                        if (gap >= BOARD_COLS) gap = BOARD_COLS - 1;
                        gq[gcnt].rows = nr;
                        gq[gcnt].col = gap;
                        gq[gcnt].timer = cfg.garbage_push_delay;
                        gcnt++;
                    }
                }
            }

            // Immediate push for delay=0
            if (cfg.garbage_push_delay == 0) {
                garb_push_all(board, bh, gq, &gcnt);
            }

            // --- Track stats ---
            int h = 0;
            for (int r = 0; r < bh; r++) {
                if (board[r] != 0) { h = bh - r; break; }
            }
            height_sum += (float)h;
            last_height = h;
            if (h > peak_height) peak_height = h;
            steps_done++;

            // --- Death check (rows 0..3 for bh=24) ---
            for (int r = 0; r < bh - 20; r++) {
                if (board[r] != 0) { died = true; break; }
            }
            if (died) break;
        }

        // Check final b2b
        if (b2b > max_b2b_val) max_b2b_val = b2b;

        res->steps_completed = steps_done;
        res->survived = died ? 0 : 1;
        res->total_attack = total_attack;
        res->max_b2b = max_b2b_val;
        res->end_height = died ? 20 : last_height;
        res->avg_height = (steps_done > 0) ? height_sum / (float)steps_done : 0.0f;
        res->max_height = peak_height;
    }
}

