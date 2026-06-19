#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>


// ============================================================
// Constants
// ============================================================

#define BOARD_ROWS 40
#define VISIBLE_ROWS 20
#define BOARD_COLS 10
#define ROTATIONS 4

// Spawn / top-out geometry (matches PyTetrisEnv): pieces spawn at row SPAWN_ROW just above the
// 20-visible field; top-out = the piece-agnostic 7-cell spawn box (rows 17-18) is blocked OR a
// column reaches DEATH_HEIGHT_CAP. HEIGHT_REF is the beam's height-gradient anchor (= death
// height). NET_ROWS is the model-visible slice (bottom 24) emitted to the net.
#define SPAWN_ROW 17
#define DEATH_HEIGHT_CAP 35
#define HEIGHT_REF 22
#define NET_ROWS 24
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
// simulated garbage during search - it is treated as occupied but
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
    int total_lines_cleared;   // Total lines cleared in this search path
    int pieces_placed;         // Pieces placed along this search path (for APP denom)
    int hold_piece;
    int next_queue_idx;
    int depth0_placement_idx;  // Which placement was chosen at depth 0 (for output)
    bool b2b_broken;
    int prev_b2b;
    int garbage_remaining;   // Simulated garbage rows not yet pushed
    int garbage_timer;       // Ticks until next garbage push (decremented on non-clear)
    float garbage_prevented; // Cumulative garbage lines kept off the board along this path:
                             // counted only when the front of the queue was about to push
                             // (timer <= 0) and the move either cancelled the lines or
                             // blocked the push by clearing.
    uint8_t bag_seen;        // Bitmask of pieces consumed from current 7-bag (bits 1-7)
    float score;
    uint64_t sort_hash;      // state_hash cached at insert; sole tiebreak key for the
                             // deterministic beam sort (so parallel append order can't
                             // change which states survive).
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

// Top-out, shared by the env/MCTS/beam: the piece-agnostic 7-cell spawn box is blocked
// (row 17 cols 3-5 = 0x38, row 18 cols 3-6 = 0x78), or the tallest column hits the cap.
static inline bool spawn_envelope_blocked_c(const uint16_t* board) {
    return (board[SPAWN_ROW] & 0x38) || (board[SPAWN_ROW + 1] & 0x78);
}

static inline int max_stack_height_c(const uint16_t* board, int board_height) {
    for (int r = 0; r < board_height; r++) if (board[r]) return board_height - r;
    return 0;
}

static inline bool board_topped_out(const uint16_t* board, int board_height) {
    return spawn_envelope_blocked_c(board)
           || max_stack_height_c(board, board_height) >= DEATH_HEIGHT_CAP;
}

// Cheap pre-prune: a placement that tops out (spawn box blocked or a column at the cap) is
// unconditionally worse than any alternative; skip it before paying for compute_board_stats.
static inline bool placement_is_dead(const SearchState* s, int board_height) {
    int mh = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        if (s->col_heights[c] > mh) mh = s->col_heights[c];
    }
    return spawn_envelope_blocked_c(s->board)
           || (mh + s->garbage_remaining) >= DEATH_HEIGHT_CAP;
}

// ============================================================
// Zobrist hashing for beam-state dedupe
//
// Two beam states that share (board, b2b, combo, hold, next_queue_idx,
// bag_seen, garbage_remaining, garbage_timer) have identical future
// subtrees - expanding both is pure waste.  Dedupe keeps the higher-
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

// Transposition-cache hash: like state_hash but also mixes in the path-dependent
// fields that feed evaluate_state (total_attack, garbage_prevented).  Two states
// sharing board+b2b+combo+hold+queue_idx but arriving via different clear histories
// would otherwise map to the same TT slot and return the wrong cached score.
static inline uint64_t tt_hash(const SearchState* s, int board_height) {
    uint64_t h = state_hash(s, board_height);
    union { float f; uint32_t u; } a, gp;
    a.f = s->total_attack;
    gp.f = s->garbage_prevented;
    h ^= (uint64_t)a.u * 0x9e3779b97f4a7c15ULL;
    h ^= (uint64_t)gp.u * 0x517cc1b727220a95ULL;
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

// ============================================================
// Globals (static to this translation unit)
// ============================================================

static PieceDef B2B_PIECES[8];
static bool b2b_initialized = false;

// Last search result - read by the C game loop to avoid lossy action_idx decode.
// b2b_search_c() writes this; b2b_run_eval_games() reads it.
static __thread Placement b2b_last_placement;

// Kick tables: [from_rot][to_rot][kick_index][0=row, 1=col]
static int8_t B2B_KICKS[4][4][5][2];
static int8_t B2B_I_KICKS[4][4][5][2];

// Heuristic weights - hand-designed, bounded, survival-first.
// Rationale is documented at each use site in evaluate_state().
//
// Scale discipline:
//   - Instant-death: -1e6 (inviolable).
//   - Near-death cliff: -5000..-10000 (dominates any achievable positive reward).
//   - Max achievable positive reward in the reachable regime ≈ 160.
//   - Hole penalties cap so the bot cannot suicide to "avoid fixing" a bad board.
//   - B2B value is sublinear (sqrt) so surge payoff wins at high chains,
//     while holding wins at low chains.
static const int   NEAR_DEATH_ZONE   = 4;        // rows from the death line where the cliff fires
static float W_NEAR_DEATH      = 5000.0f;  // per row of slack inside the zone
static float W_HEIGHT_QUARTIC  = 80.0f;    // -W * h_ratio^4
static float W_AVG_HEIGHT      = 40.0f;     // -W * avg_height.  Linear penalty on total stack volume
                                                   // (cells occupied per column, averaged).  Encourages
                                                   // "board emptiness" - a clean board with b2b=N beats a
                                                   // messy board with b2b=N because future options are
                                                   // preserved.  Pairs with b2b-reward terms to reward
                                                   // b2b-per-piece efficiency: hoarding spin slots without
                                                   // cashing them in is explicitly penalized via the cells
                                                   // those slots consume.  At avg_height=10 the penalty is
                                                   // -30 - big enough to discourage unnecessary upstacking
                                                   // but well below the survival wall and raw b2b rewards.
static float W_BUMPINESS       = 1.0f;

static float W_HOLES           = 6.0f;     // * holes (enclosed cavities) * (1 + 0.5h), uncapped
static float W_HOLE_CEILING    = 1.5f;     // * hole_ceiling_weight (buried-hole depth)

// B2B store (W_B2B_LINEAR is the dominant hold/hoarding driver)
static float W_B2B_FLAT        = 5.0f;     // one-shot "b2b active" flag (b2b>=0, incl. starting b2b)
static float W_B2B_SQRT        = 8.0f;     // * sqrt(b2b)
static float W_B2B_LINEAR      = 20.0f;    // * b2b

// Attack realization
static float W_ATTACK_TOTAL    = 1.0f;     // * (total_attack + max(0, leaf_b2b))
static float W_APP             = 100.0f;   // * (total_attack + max(0,b2b)) / pieces_placed
static float W_GARBAGE_PREVENT = 4.0f;     // * garbage_prevented - keep imminent garbage off the board

// Spin-setup structure (b2b-maintaining clear potential)
static float W_TSLOT           = 6.0f;
static float W_IMMOBILE_CLEAR  = 5.0f;     // * sqrt(immobile_clearing_placements)
static float W_IMMOBILE_LINES  = 1.0f;     // * min(immobile_clearable_lines, 8)

// ============================================================
// Weight Override Table (for ablation / tuning)
//
// Each entry maps a string name to the address of a W_* global so callers can
// override or reset weights at runtime without recompiling.  Used by
// b2b_set_weight / b2b_reset_weights - invoked from Python for ablation sweeps.
// ============================================================

typedef struct {
    const char* name;
    float* ptr;
    float default_value;
} WeightEntry;

static WeightEntry W_TABLE[] = {
    {"W_NEAR_DEATH",       &W_NEAR_DEATH,       5000.0f},
    {"W_HEIGHT_QUARTIC",   &W_HEIGHT_QUARTIC,   80.0f},
    {"W_AVG_HEIGHT",       &W_AVG_HEIGHT,       40.0f},
    {"W_BUMPINESS",        &W_BUMPINESS,        1.0f},
    {"W_HOLES",            &W_HOLES,            6.0f},
    {"W_HOLE_CEILING",     &W_HOLE_CEILING,     1.5f},
    {"W_B2B_FLAT",         &W_B2B_FLAT,         5.0f},
    {"W_B2B_SQRT",         &W_B2B_SQRT,         8.0f},
    {"W_B2B_LINEAR",       &W_B2B_LINEAR,       20.0f},
    {"W_ATTACK_TOTAL",     &W_ATTACK_TOTAL,     1.0f},
    {"W_APP",              &W_APP,              100.0f},
    {"W_GARBAGE_PREVENT",  &W_GARBAGE_PREVENT,  4.0f},
    {"W_TSLOT",            &W_TSLOT,            6.0f},
    {"W_IMMOBILE_CLEAR",   &W_IMMOBILE_CLEAR,   5.0f},
    {"W_IMMOBILE_LINES",   &W_IMMOBILE_LINES,   1.0f},
};
static const int W_TABLE_LEN = (int)(sizeof(W_TABLE) / sizeof(W_TABLE[0]));

int b2b_set_weight(const char* name, float value) {
    for (int i = 0; i < W_TABLE_LEN; i++) {
        if (strcmp(W_TABLE[i].name, name) == 0) {
            *W_TABLE[i].ptr = value;
            // Invalidate the eval transposition cache so stale scores from
            // before the weight change can never satisfy a TT lookup.
            g_tt_generation += TT_GENERATION_EXPIRY + 1;
            return 1;
        }
    }
    return 0;
}

float b2b_get_weight(const char* name) {
    for (int i = 0; i < W_TABLE_LEN; i++) {
        if (strcmp(W_TABLE[i].name, name) == 0) return *W_TABLE[i].ptr;
    }
    return 0.0f;
}

void b2b_reset_weights(void) {
    for (int i = 0; i < W_TABLE_LEN; i++) *W_TABLE[i].ptr = W_TABLE[i].default_value;
}

int b2b_get_weight_count(void) { return W_TABLE_LEN; }

const char* b2b_get_weight_name(int idx) {
    if (idx < 0 || idx >= W_TABLE_LEN) return "";
    return W_TABLE[idx].name;
}

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
// only use this when no lines cleared and no garbage was pushed on this step -
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

// Attack calculation - exact replica of Scorer.py
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
    static __thread BFSStateMeta meta[BFS_STATE_SPACE];
    static __thread bool visited[BFS_STATE_SPACE];
    static __thread int queue[BFS_QUEUE_CAPACITY];

    for (int i = 0; i < BFS_STATE_SPACE; i++) {
        visited[i] = false;
        meta[i].parent = -1;
    }

    // Spawn position: row=SPAWN_ROW, col=3, rot=0 (just above the visible field)
    int start_r = SPAWN_ROW, start_c = 3, start_rot = 0;
    int start_state = b2b_encode_state(start_r, start_c, start_rot, piece_type);

    if (start_state == -1 ||
        b2b_check_collision(board_rows, board_height, piece_type, start_rot, start_r, start_c)) {
        return 0; // Can't spawn - board is topped out
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
    // Emit every placement reachable from the spawn line down; evaluate_state's envelope/cap
    // death prunes the topped-out ones (so no live placement is dropped for the oracle).
    int visible_start = SPAWN_ROW;

    while (head != tail) {
        int curr_state = queue[head++];
        head %= BFS_QUEUE_CAPACITY;

        int r, c, rot;
        b2b_decode_state(curr_state, &r, &c, &rot, piece_type);
        int depth = meta[curr_state].depth;

        // Only emit a placement when this BFS state IS the landing state
        // (the piece can no longer fall from here). Non-landing states will
        // reach their landing via SOFT_DROP, which sets delta_r=0 - matching
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

        // BFS depth limit - keep paths short enough for max_len=15 sequences
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

    int flood_queue[BOARD_ROWS * BOARD_COLS * 2]; // row, col pairs
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

    // Queue - worst case is every cell on the board
    int queue_r[BOARD_ROWS * BOARD_COLS];
    int queue_c[BOARD_ROWS * BOARD_COLS];

    int sections = 0;
    for (int r = 0; r < board_height; r++) {
        uint16_t remaining = hole_mask[r] & ~visited[r];
        while (remaining) {
            // Pick lowest-set-bit column
            int c = __builtin_ctz(remaining);

            // New connected component - flood fill from (r, c)
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
// This penalizes upstacking over enclosed holes - each filled cell placed
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
                // Empty cell - check if it's an enclosed hole
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
            // T cells: [1,0] [1,1] [1,2] [2,1] - center is [1,1]
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

            // Rot 0 (T points up - less common, needs slot below)
            // T cells: [0,1] [1,0] [1,1] [1,2] - center is [1,1]
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
            // T cells: [0,1] [1,1] [1,2] [2,1] - center is [1,1]
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
            // T cells: [0,1] [1,0] [1,1] [2,1] - center is [1,1]
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
//                 (the actual ALL_MINI criterion - matches
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
//   immobile_cells[]   - cells in ANY valid immobile placement (used for
//                         wasted-hole detection: reachable holes not in here
//                         are "wasted").
//   clearing_cells[]   - cells in immobile placements that CLEAR at least one
//                         line (used to exempt productive spin-setup holes from
//                         hole penalties - only clearing setups earn exemption).
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
    // stack cells - pieces sitting multiple rows below the stack top cannot
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


// Per-column heights + the board-shape stats the eval actually reads.
typedef struct {
    int col_heights[BOARD_COLS];
    int max_height;
    float avg_height;
    int holes;
    float hole_ceiling_weight;           // Weighted count of filled cells above enclosed holes
    float immobile_clearing_placements;  // Queue-weighted immobile + line-clearing placements
    float immobile_clearable_lines;      // Queue-weighted clearable lines from immobile placements
    int t_spin_setups;                   // Number of T-spin setups detected
    int t_slot_quality;                  // Best T-slot quality (0=none, 1=mini, 2=full)
    int t_queue_count;                   // T pieces in the upcoming queue (caps W_TSLOT)
    float bumpiness_exempted;            // Bumpiness excluding adjacencies around the deepest well
} BoardStats;


static BoardStats compute_board_stats(const uint16_t* board, int board_height,
                                      const int* upcoming_pieces, int num_upcoming,
                                      const int8_t* height_hint) {
    BoardStats s;
    memset(&s, 0, sizeof(s));

    // Column heights - cached hint when available, else scan.
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

    // Max + average height.
    float total_h = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        if (s.col_heights[c] > s.max_height) s.max_height = s.col_heights[c];
        total_h += s.col_heights[c];
    }
    s.avg_height = total_h / BOARD_COLS;

    // Raw bumpiness (local; the eval reads only the well-exempted variant below).
    float bumpiness = 0.0f;
    for (int c = 0; c < BOARD_COLS - 1; c++) {
        bumpiness += fabsf((float)(s.col_heights[c] - s.col_heights[c + 1]));
    }

    // Flood-fill reachability (feeds the hole + immobile scans).
    uint16_t reachable[BOARD_ROWS];
    compute_reachability(board, board_height, reachable);

    // Immobile (b2b-maintaining) spin placements + T-piece queue count.
    uint16_t immobile_cells[BOARD_ROWS];
    uint16_t clearing_cells[BOARD_ROWS];
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
    s.immobile_clearing_placements = ipr.weighted_immobile_clearing;
    s.immobile_clearable_lines = ipr.weighted_immobile_lines;

    // Hole metrics.  Spin-channel hole-forgiveness was removed, so pass an empty
    // exemption mask - identical to the prior W_SPIN_CHANNEL=0 behavior.
    uint16_t no_channel[BOARD_ROWS];
    memset(no_channel, 0, sizeof(uint16_t) * board_height);
    s.holes = count_hole_sections(board, board_height, reachable, no_channel);
    if (W_HOLE_CEILING != 0.0f) {
        s.hole_ceiling_weight = compute_hole_ceiling_weight(board, board_height, reachable, no_channel);
    }

    // T-spin setups (the multiline output is discarded - its reward was removed).
    int t_multiline_unused = 0;
    s.t_spin_setups = detect_t_spin_setups(board, board_height, &s.t_slot_quality, &t_multiline_unused);

    // Deepest well column - the only well stat the eval needs (bumpiness exemption).
    int well_col = -1, well_depth = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        int left_h = (c > 0) ? s.col_heights[c - 1] : board_height;
        int right_h = (c < BOARD_COLS - 1) ? s.col_heights[c + 1] : board_height;
        int min_neighbor = (left_h < right_h) ? left_h : right_h;
        int depth = min_neighbor - s.col_heights[c];
        if (depth >= 2 && depth > well_depth) { well_depth = depth; well_col = c; }
    }

    // Bumpiness, exempting the two adjacencies around the deepest well.
    s.bumpiness_exempted = bumpiness;
    if (well_col >= 0) {
        if (well_col > 0)
            s.bumpiness_exempted -= fabsf((float)(s.col_heights[well_col - 1] - s.col_heights[well_col]));
        if (well_col < BOARD_COLS - 1)
            s.bumpiness_exempted -= fabsf((float)(s.col_heights[well_col] - s.col_heights[well_col + 1]));
        if (s.bumpiness_exempted < 0.0f) s.bumpiness_exempted = 0.0f;
    }

    return s;
}

// ── Deterministic beam finalize ──────────────────────────────
// The OpenMP beam expansion appends children in nondeterministic (thread-timing)
// order.  Selection is therefore a strict TOTAL order - higher score first, ties
// broken by the cached state hash then the depth-0 origin - so which states
// survive (and which depth-0 move is ultimately credited) is independent of how
// the threads interleaved.  Identical states share a hash AND a score, so they
// sort adjacent and are deduped, keeping the canonical (lowest depth-0) ancestor.
static int beam_cmp(const void* pa, const void* pb) {
    const SearchState* a = (const SearchState*)pa;
    const SearchState* b = (const SearchState*)pb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->sort_hash < b->sort_hash) return -1;
    if (a->sort_hash > b->sort_hash) return 1;
    if (a->depth0_placement_idx < b->depth0_placement_idx) return -1;
    if (a->depth0_placement_idx > b->depth0_placement_idx) return 1;
    return 0;
}

// Sort the freshly-expanded beam by the total order, drop duplicate states
// (same-hash entries are adjacent because a state's score is deterministic),
// and keep at most K.  Fully deterministic regardless of append order.
static int finalize_beam(SearchState* beam, int n, int K) {
    if (n <= 0) return 0;
    qsort(beam, n, sizeof(SearchState), beam_cmp);
    int w = 1;
    for (int i = 1; i < n; i++) {
        if (beam[i].sort_hash == beam[w - 1].sort_hash) continue;  // dup of a kept entry
        beam[w++] = beam[i];
    }
    return w < K ? w : K;
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

    int effective_h = bs.max_height + state->garbage_remaining;

    // Instant death - inviolable floor: spawn box blocked, or a column at the height cap.
    if (spawn_envelope_blocked_c(state->board) || effective_h >= DEATH_HEIGHT_CAP) {
        return -1e6f;
    }

    float h_ratio = (float)effective_h / (float)HEIGHT_REF;

    // ── Survival wall ─────────────────────────────────────────
    // Near-death cliff: within NEAR_DEATH_ZONE rows of the death line, a penalty
    // that beats every positive term combined (slack=0 => the next block kills us).
    if (effective_h >= HEIGHT_REF - NEAR_DEATH_ZONE) {
        int slack = HEIGHT_REF - 1 - effective_h;
        score -= W_NEAR_DEATH * (float)(NEAR_DEATH_ZONE - slack);
    }
    // Smooth height (quartic) + linear volume penalty (rewards board emptiness).
    score -= W_HEIGHT_QUARTIC * h_ratio * h_ratio * h_ratio * h_ratio;
    score -= W_AVG_HEIGHT * bs.avg_height;
    // Bumpiness (well-column-exempted so deliberate spin/Tetris wells aren't double-penalized).
    score -= W_BUMPINESS * bs.bumpiness_exempted;

    // ── Hole accounting ───────────────────────────────────────
    float hole_mult = 1.0f + 0.5f * h_ratio;
    if (bs.holes > 0) {
        score -= W_HOLES * (float)bs.holes * hole_mult;
    }
    if (bs.hole_ceiling_weight > 0.0f) {
        score -= W_HOLE_CEILING * bs.hole_ceiling_weight;  // burying holes deeper is worse
    }

    // ── B2B economy (store terms; W_B2B_LINEAR is the hoarding driver) ─────────
    // W_B2B_FLAT fires for b2b >= 0, so it ALSO rewards starting b2b (b2b==0).
    if (state->b2b >= 0) {
        score += W_B2B_FLAT;
    }
    if (state->b2b > 0) {
        score += W_B2B_SQRT * sqrtf((float)state->b2b);
        score += W_B2B_LINEAR * (float)state->b2b;
    }

    // ── Attack realization ────────────────────────────────────
    // Realized attack along the path + banked b2b (pending surge ~= b2b on break).
    float atk_credit = state->total_attack + (state->b2b > 0 ? (float)state->b2b : 0.0f);
    if (atk_credit > 0.0f) {
        score += W_ATTACK_TOTAL * atk_credit;
    }
    // Direct APP (attack-per-piece), counting stored b2b as pending surge attack.
    if (state->pieces_placed > 0) {
        float b2b_val = state->b2b > 0 ? (float)state->b2b : 0.0f;
        score += W_APP * ((state->total_attack + b2b_val) / (float)state->pieces_placed);
    }
    // Garbage prevention: keeping imminent garbage off the board (cancel or block-push).
    if (state->garbage_prevented > 0.0f) {
        score += W_GARBAGE_PREVENT * state->garbage_prevented;
    }

    // ── Spin-setup structure (b2b-maintaining clear potential) ─────────────────
    // Queue-capped T-slot reward (a T-slot with no T coming in the queue pays nothing).
    if (bs.t_spin_setups > 0 && bs.t_queue_count > 0) {
        float t_reward = 0.0f;
        if (bs.t_slot_quality == 2) {
            t_reward = W_TSLOT;
        } else if (bs.t_slot_quality == 1) {
            t_reward = W_TSLOT * 0.4f;
        }
        int usable_setups = bs.t_spin_setups < bs.t_queue_count ? bs.t_spin_setups : bs.t_queue_count;
        t_reward *= (1.0f + 0.3f * (float)(usable_setups - 1));  // uncapped (still queue-bounded)
        score += t_reward;
    }
    // b2b-maintaining (immobile/spin) clear slots ready on this board.
    if (bs.immobile_clearing_placements > 0.0f) {
        float line_reward = W_IMMOBILE_CLEAR * sqrtf(bs.immobile_clearing_placements);
        line_reward += W_IMMOBILE_LINES * bs.immobile_clearable_lines;  // uncapped
        score += line_reward;
    }

    _tt_e->hash = _tt_h;
    _tt_e->score = score;
    _tt_e->generation = g_tt_generation;
    return score;
}

// ============================================================
// Score Decomposition (for heuristic influence analysis)
// ============================================================

#define NUM_DECOMPOSE 13

#define D_HEIGHT          0
#define D_NEAR_DEATH      1
#define D_BUMPINESS       2
#define D_HOLES           3
#define D_HOLE_CEILING    4
#define D_B2B_FLAT        5
#define D_B2B_SQRT        6
#define D_B2B_LINEAR      7
#define D_ATTACK          8
#define D_APP             9
#define D_TSLOT           10
#define D_IMMOBILE_CLEAR  11
#define D_GARBAGE_PREVENT 12

int b2b_get_num_decompose(void) { return NUM_DECOMPOSE; }

// Mirrors evaluate_state() term-by-term, writing each kept component into d[].
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

    int effective_h = bs.max_height + state->garbage_remaining;
    if (spawn_envelope_blocked_c(state->board) || effective_h >= DEATH_HEIGHT_CAP) {
        d[D_HEIGHT] = -1e6f;
        return;
    }

    float h_ratio = (float)effective_h / (float)HEIGHT_REF;
    float hole_mult = 1.0f + 0.5f * h_ratio;

    // Survival
    d[D_HEIGHT] = -W_HEIGHT_QUARTIC * h_ratio * h_ratio * h_ratio * h_ratio
                  - W_AVG_HEIGHT * bs.avg_height;
    if (effective_h >= HEIGHT_REF - NEAR_DEATH_ZONE) {
        int slack = HEIGHT_REF - 1 - effective_h;
        d[D_NEAR_DEATH] = -W_NEAR_DEATH * (float)(NEAR_DEATH_ZONE - slack);
    }
    d[D_BUMPINESS] = -W_BUMPINESS * bs.bumpiness_exempted;

    // Holes
    if (bs.holes > 0)
        d[D_HOLES] = -W_HOLES * (float)bs.holes * hole_mult;
    if (bs.hole_ceiling_weight > 0.0f)
        d[D_HOLE_CEILING] = -W_HOLE_CEILING * bs.hole_ceiling_weight;

    // B2B store
    if (state->b2b >= 0) d[D_B2B_FLAT] = W_B2B_FLAT;
    if (state->b2b > 0) {
        d[D_B2B_SQRT] = W_B2B_SQRT * sqrtf((float)state->b2b);
        d[D_B2B_LINEAR] = W_B2B_LINEAR * (float)state->b2b;
    }

    // Attack: realized attack + banked leaf-b2b credit
    float b2b_credit = state->b2b > 0 ? (float)state->b2b : 0.0f;
    if (state->total_attack + b2b_credit > 0.0f)
        d[D_ATTACK] = W_ATTACK_TOTAL * (state->total_attack + b2b_credit);

    // APP (attack-per-piece; counts stored b2b as pending surge attack)
    if (state->pieces_placed > 0)
        d[D_APP] = W_APP * ((state->total_attack + b2b_credit) / (float)state->pieces_placed);

    // T-slot (queue-capped)
    if (bs.t_spin_setups > 0 && bs.t_queue_count > 0) {
        float tr = 0.0f;
        if (bs.t_slot_quality == 2) tr = W_TSLOT;
        else if (bs.t_slot_quality == 1) tr = W_TSLOT * 0.4f;
        int usable_setups = bs.t_spin_setups < bs.t_queue_count ? bs.t_spin_setups : bs.t_queue_count;
        tr *= (1.0f + 0.3f * (float)(usable_setups - 1));
        d[D_TSLOT] = tr;
    }

    // b2b-maintaining (immobile/spin) clear slots
    if (bs.immobile_clearing_placements > 0.0f) {
        d[D_IMMOBILE_CLEAR] = W_IMMOBILE_CLEAR * sqrtf(bs.immobile_clearing_placements)
                            + W_IMMOBILE_LINES * bs.immobile_clearable_lines;
    }

    if (state->garbage_prevented > 0.0f)
        d[D_GARBAGE_PREVENT] = W_GARBAGE_PREVENT * state->garbage_prevented;
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

    // Treat any pending garbage as imminent (timer = 0).  The push timer in the
    // env has already ticked once between the last move and this one, so by the
    // time the search runs the front-of-queue item is effectively about to push.
    // This also lets the depth-0 garbage_prevented reward fire on the move that
    // actually executes.
    (void)garbage_push_delay;
    int init_gt = 0;
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
        s.total_attack = ar.attack;
        s.total_lines_cleared = clears; s.hold_piece = hold_piece;
        s.pieces_placed = 1;
        s.next_queue_idx = 0; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
        { int gr = total_garbage; int gc = 0; int prevented = 0;
          bool was_imminent = (init_gt <= 0 && gr > 0);
          if (ar.attack > 0 && gr > 0) { gc = ((int)ar.attack > gr) ? gr : (int)ar.attack; gr -= gc; }
          if (was_imminent) {
              bool would_push = (clears == 0 && gr > 0);
              prevented = would_push ? gc : total_garbage;
          }
          s.garbage_remaining = gr; s.garbage_timer = init_gt; s.garbage_prevented = (float)prevented; }
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
            s.total_attack = ar.attack;
            s.total_lines_cleared = clears; s.hold_piece = active_piece;
            s.pieces_placed = 1;
            s.next_queue_idx = 0; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
            { int gr = total_garbage; int gc = 0; int prevented = 0;
          bool was_imminent = (init_gt <= 0 && gr > 0);
          if (ar.attack > 0 && gr > 0) { gc = ((int)ar.attack > gr) ? gr : (int)ar.attack; gr -= gc; }
          if (was_imminent) {
              bool would_push = (clears == 0 && gr > 0);
              prevented = would_push ? gc : total_garbage;
          }
          s.garbage_remaining = gr; s.garbage_timer = init_gt; s.garbage_prevented = (float)prevented; }
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
            s.total_attack = ar.attack;
            s.total_lines_cleared = clears; s.hold_piece = active_piece;
            s.pieces_placed = 1;
            s.next_queue_idx = 1; s.b2b_broken = ar.b2b_broken; s.prev_b2b = b2b;
            { int gr = total_garbage; int gc = 0; int prevented = 0;
          bool was_imminent = (init_gt <= 0 && gr > 0);
          if (ar.attack > 0 && gr > 0) { gc = ((int)ar.attack > gr) ? gr : (int)ar.attack; gr -= gc; }
          if (was_imminent) {
              bool would_push = (clears == 0 && gr > 0);
              prevented = would_push ? gc : total_garbage;
          }
          s.garbage_remaining = gr; s.garbage_timer = init_gt; s.garbage_prevented = (float)prevented; }
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

    // Reserve 1 slot for HARD_DROP - truncate path if needed
    int max_path_keys = max_len - p - 1;
    int start = (len > max_path_keys) ? (len - max_path_keys) : 0;
    for (int i = len - 1; i >= start; i--) {
        out_row[p++] = path[i];
    }

    out_row[p++] = KEY_HARD_DROP; // Always fits now
    while (p < max_len) out_row[p++] = KEY_PAD;
}

// ============================================================
// Parallel beam child expansion helper (depths 1+)
// ============================================================

// Thread-safe child expansion: builds a child state from a parent + placement,
// evaluates it, and atomically inserts into next_beam if it passes all filters.
static inline void expand_and_insert(
    SearchState* next_beam, int* next_beam_size_ptr, int max_next,
    const SearchState* parent,
    int board_height,
    int piece_type, const Placement* pl,
    int new_hold_piece,
    int new_queue_idx,
    uint8_t new_bag_seen,
    const int* queue, int queue_len
) {
    SearchState child;
    memcpy(child.board, parent->board, sizeof(uint16_t) * board_height);
    lock_piece_on_board(child.board, board_height, piece_type, pl->rot, pl->landing_row, pl->col);
    int clears = clear_lines(child.board, board_height);

    bool perfect_clear = true;
    for (int r = 0; r < board_height; r++) {
        if (child.board[r] != 0) { perfect_clear = false; break; }
    }

    AttackResult ar = compute_attack(clears, pl->spin_type, parent->b2b, parent->combo, perfect_clear);

    child.b2b = ar.new_b2b;
    child.combo = ar.new_combo;
    child.total_attack = parent->total_attack + ar.attack;
    child.total_lines_cleared = parent->total_lines_cleared + clears;
    child.pieces_placed = parent->pieces_placed + 1;
    child.hold_piece = new_hold_piece;
    child.next_queue_idx = new_queue_idx;
    child.depth0_placement_idx = parent->depth0_placement_idx;
    child.b2b_broken = parent->b2b_broken || ar.b2b_broken;
    child.prev_b2b = parent->b2b_broken ? parent->prev_b2b
                   : (ar.b2b_broken ? parent->b2b : parent->prev_b2b);
    child.bag_seen = new_bag_seen;

    bool pushed_garbage = false;
    int prevented = 0;
    {
        int gr = parent->garbage_remaining;
        int gt = parent->garbage_timer;
        int gr_initial = gr;
        bool was_imminent = (gt <= 0 && gr_initial > 0);
        int cancelled = 0;
        if (ar.attack > 0 && gr > 0) {
            cancelled = ((int)ar.attack > gr) ? gr : (int)ar.attack;
            gr -= cancelled;
        }
        if (clears == 0 && gr > 0) {
            if (gt <= 0) {
                push_simulated_garbage(child.board, board_height, gr);
                gr = 0;
                pushed_garbage = true;
            } else {
                gt--;
            }
        }
        // Only count prevention when garbage was about to push.  If we pushed,
        // only the cancelled portion was kept off the board; otherwise a clear
        // (or full cancel) blocked the entire imminent push.
        if (was_imminent) {
            prevented = pushed_garbage ? cancelled : gr_initial;
        }
        child.garbage_remaining = gr;
        child.garbage_timer = gt;
    }
    child.garbage_prevented = parent->garbage_prevented + (float)prevented;

    if (clears == 0 && !pushed_garbage) {
        patch_col_heights_after_place(parent->col_heights, piece_type, pl->rot,
                                      pl->landing_row, pl->col, board_height,
                                      child.col_heights);
    } else {
        compute_col_heights_full(child.board, board_height, child.col_heights);
    }

    if (placement_is_dead(&child, board_height)) return;

    // Full eval for every surviving child (no aspiration prune - it was the main
    // source of nondeterminism and its omission only makes the beam MORE thorough).
    child.score = evaluate_state(&child, board_height, queue, queue_len);
    child.sort_hash = state_hash(&child, board_height);  // cached tiebreak key for finalize_beam

    int slot = __atomic_fetch_add(next_beam_size_ptr, 1, __ATOMIC_RELAXED);
    if (slot < max_next) {
        next_beam[slot] = child;
    }
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
    int64_t* out_best_sequence,     // Output: key sequence (length max_len)
    // --- Optional per-root candidate output (for policy/value distillation) ---
    // Pass max_roots<=0 or out_num_roots==NULL to skip.  For every root placement
    // that survives to the final beam, writes its action index, best-leaf score
    // (the value of the best continuation through that root), and reconstructed
    // key sequence.  The overall value target is max(out_root_scores).  Scores are
    // RAW search scores (no softmax - the caller decides any normalization).
    int max_roots,
    int* out_num_roots,
    int* out_root_action_indices,   // [max_roots]
    float* out_root_scores,         // [max_roots]
    int64_t* out_root_sequences,    // [max_roots * max_len]
    int* out_root_landing_rows      // [max_roots] BFS lock row (0..board_height-1)
) {
    if (out_num_roots) *out_num_roots = 0;
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

    if (!curr_beam || !next_beam) {
        // Allocation failed
        *out_action_index = -1;
        for (int i = 0; i < max_len; i++) out_best_sequence[i] = KEY_PAD;
        free(curr_beam);
        free(next_beam);
        return;
    }

    // Store depth-0 BFS meta and placement info for sequence reconstruction
    static __thread BFSStateMeta depth0_meta_active[BFS_STATE_SPACE];
    static __thread BFSStateMeta depth0_meta_hold[BFS_STATE_SPACE];
    static __thread Placement depth0_placements[MAX_PLACEMENTS * 2];
    static __thread int depth0_is_hold[MAX_PLACEMENTS * 2];
    int depth0_count = 0;

    // Per-root best descendant score, tracked across ALL depths (pre-prune) so
    // every root placement keeps a score even after the beam prunes its subtree -
    // gives a full candidate distribution despite the beam converging to one root.
    // Observe-only: does not affect the beam/pruning, so move quality is unchanged.
    const bool want_roots = (out_num_roots != NULL && max_roots > 0);
    static __thread float root_best[MAX_PLACEMENTS * 2];
    if (want_roots) {
        for (int i = 0; i < MAX_PLACEMENTS * 2; i++) root_best[i] = -1e30f;
    }

    // Treat any pending garbage as imminent at depth 0 (timer = 0).  In the
    // env, the front-of-queue garbage has already ticked once between the last
    // move and this one, so by the time the search runs it's effectively about
    // to push.  Modeling it as imminent both matches the env state and lets
    // the depth-0 garbage_prevented reward fire on the move actually executed.
    (void)garbage_push_delay;
    int init_garbage_timer = 0;

    // Precompute the root board's column heights once so every depth-0
    // placement can incrementally patch from it.
    int8_t root_col_heights[BOARD_COLS];
    compute_col_heights_full(board_rows, board_height, root_col_heights);

    // ---- Depth 0: enumerate placements for active piece and hold piece ----

    // Fallback: save the first placement found at depth 0 (even if dead)
    // so we always have a move to return if the beam goes empty.
    Placement fallback_pl = {0};
    int fallback_is_hold = 0;
    int fallback_piece = active_piece;
    BFSStateMeta* fallback_meta = depth0_meta_active;
    bool have_fallback = false;

    Placement placements[MAX_PLACEMENTS];
    int np;

    // Active piece placements
    np = find_placements(board_rows, board_height, active_piece, placements, MAX_PLACEMENTS,
                         depth0_meta_active);
    if (np > 0 && !have_fallback) {
        fallback_pl = placements[0]; fallback_is_hold = 0;
        fallback_piece = active_piece; fallback_meta = depth0_meta_active;
        have_fallback = true;
    }

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
            int gr_initial = gr;
            bool was_imminent = (gt <= 0 && gr_initial > 0);
            int gc = 0;
            int prevented = 0;
            if (ar.attack > 0 && gr > 0) {
                gc = ((int)ar.attack > gr) ? gr : (int)ar.attack;
                gr -= gc;
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
            if (was_imminent) {
                prevented = pushed_garbage ? gc : gr_initial;
            }
            s->garbage_remaining = gr;
            s->garbage_timer = gt;
            s->garbage_prevented = (float)prevented;
        }

        if (clears == 0 && !pushed_garbage) {
            patch_col_heights_after_place(root_col_heights, active_piece, pl->rot,
                                          pl->landing_row, pl->col, board_height,
                                          s->col_heights);
        } else {
            compute_col_heights_full(s->board, board_height, s->col_heights);
        }

                s->score = evaluate_state(s, board_height, queue, queue_len);
                s->sort_hash = state_hash(s, board_height);
                root_best[depth0_count] = s->score;  // seed own eval: every legal root emitted

        depth0_placements[depth0_count] = *pl;
        depth0_is_hold[depth0_count] = 0;
        depth0_count++;
        // Near-death roots are emitted as (very-low-scored) candidates but NOT expanded:
        // skipping next_beam_size++ reuses this slot, keeping the search beam identical to
        // the death-pruned version (no play regression) while widening the label set.
        if (placement_is_dead(s, board_height)) { continue; }
        next_beam_size++;
    }

    // Hold piece placements
    if (hold_piece != PIECE_N) {
        // Swap: play hold piece, hold becomes active
        np = find_placements(board_rows, board_height, hold_piece, placements, MAX_PLACEMENTS,
                             depth0_meta_hold);
        if (np > 0 && !have_fallback) {
            fallback_pl = placements[0]; fallback_is_hold = 1;
            fallback_piece = hold_piece; fallback_meta = depth0_meta_hold;
            have_fallback = true;
        }

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
                int gr_initial = gr;
                bool was_imminent = (gt <= 0 && gr_initial > 0);
                int gc = 0;
                int prevented = 0;
                if (ar.attack > 0 && gr > 0) {
                    gc = ((int)ar.attack > gr) ? gr : (int)ar.attack;
                    gr -= gc;
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
                if (was_imminent) {
                    prevented = pushed_garbage ? gc : gr_initial;
                }
                s->garbage_remaining = gr;
                s->garbage_timer = gt;
                s->garbage_prevented = (float)prevented;
            }

            if (clears == 0 && !pushed_garbage) {
                patch_col_heights_after_place(root_col_heights, hold_piece, pl->rot,
                                              pl->landing_row, pl->col, board_height,
                                              s->col_heights);
            } else {
                compute_col_heights_full(s->board, board_height, s->col_heights);
            }

                s->score = evaluate_state(s, board_height, queue, queue_len);
                s->sort_hash = state_hash(s, board_height);
                root_best[depth0_count] = s->score;  // seed own eval: every legal root emitted

            depth0_placements[depth0_count] = *pl;
            depth0_is_hold[depth0_count] = 1;
            depth0_count++;
            // Emit near-death roots as candidates but don't expand (see no-hold branch).
            if (placement_is_dead(s, board_height)) { continue; }
            next_beam_size++;
        }
    } else if (queue_len > 0) {
        // No hold piece yet - hold swaps active with first queue piece
        int swap_piece = queue[0];
        np = find_placements(board_rows, board_height, swap_piece, placements, MAX_PLACEMENTS,
                             depth0_meta_hold);
        if (np > 0 && !have_fallback) {
            fallback_pl = placements[0]; fallback_is_hold = 1;
            fallback_piece = swap_piece; fallback_meta = depth0_meta_hold;
            have_fallback = true;
        }

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
                int gr_initial = gr;
                bool was_imminent = (gt <= 0 && gr_initial > 0);
                int gc = 0;
                int prevented = 0;
                if (ar.attack > 0 && gr > 0) {
                    gc = ((int)ar.attack > gr) ? gr : (int)ar.attack;
                    gr -= gc;
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
                if (was_imminent) {
                    prevented = pushed_garbage ? gc : gr_initial;
                }
                s->garbage_remaining = gr;
                s->garbage_timer = gt;
                s->garbage_prevented = (float)prevented;
            }

            if (clears == 0 && !pushed_garbage) {
                patch_col_heights_after_place(root_col_heights, swap_piece, pl->rot,
                                              pl->landing_row, pl->col, board_height,
                                              s->col_heights);
            } else {
                compute_col_heights_full(s->board, board_height, s->col_heights);
            }

                s->score = evaluate_state(s, board_height, queue, queue_len);
                s->sort_hash = state_hash(s, board_height);
                root_best[depth0_count] = s->score;  // seed own eval: every legal root emitted

            depth0_placements[depth0_count] = *pl;
            depth0_is_hold[depth0_count] = 1;
            depth0_count++;
            // Emit near-death roots as candidates but don't expand (see no-hold branch).
            if (placement_is_dead(s, board_height)) { continue; }
            next_beam_size++;
        }
    }

    // Record every root's depth-0 score before pruning (so roots pruned here
    // still get a candidate score).
    if (want_roots) {
        for (int i = 0; i < next_beam_size; i++) {
            int ri = next_beam[i].depth0_placement_idx;
            if (ri >= 0 && ri < depth0_count && next_beam[i].score > root_best[ri])
                root_best[ri] = next_beam[i].score;
        }
    }

    // Deterministic dedupe + top-K select.  (Dedupe at depth 0 is safe because two
    // placements that collapse to the same post-state have identical futures -
    // picking either depth0 choice produces the same end state; the total-order
    // tiebreak keeps the canonical one.)
    next_beam_size = finalize_beam(next_beam, next_beam_size, beam_width);

    // Swap beams
    {
        SearchState* tmp = curr_beam;
        curr_beam = next_beam;
        curr_beam_size = next_beam_size;
        next_beam = tmp;
        next_beam_size = 0;
    }

    // ---- Depths 1..search_depth-1 (OpenMP parallel beam expansion) ----
    for (int depth = 1; depth < search_depth; depth++) {
        int next_count = 0;

        #pragma omp parallel
        {
            Placement local_pl[MAX_PLACEMENTS];

            #pragma omp for schedule(dynamic)
            for (int bi = 0; bi < curr_beam_size; bi++) {
                SearchState* parent = &curr_beam[bi];
                int qi = parent->next_queue_idx;
                int np_local;

                if (qi >= queue_len) {
                    // ---- Speculative: branch on remaining bag pieces ----
                    int remaining[7];
                    int n_remaining = bag_get_remaining(parent->bag_seen, remaining);

                    if (n_remaining == 0) {
                        int slot = __atomic_fetch_add(&next_count, 1, __ATOMIC_RELAXED);
                        if (slot < max_next) next_beam[slot] = *parent;
                        continue;
                    }

                    for (int pi = 0; pi < n_remaining; pi++) {
                        int spec_piece = remaining[pi];
                        uint8_t new_bag = bag_consume_piece(parent->bag_seen, spec_piece);

                        np_local = find_placements(parent->board, board_height, spec_piece,
                                                   local_pl, MAX_PLACEMENTS, NULL);
                        for (int i = 0; i < np_local; i++)
                            expand_and_insert(next_beam, &next_count, max_next,
                                parent, board_height, spec_piece, &local_pl[i],
                                parent->hold_piece, qi + 1, new_bag, queue, queue_len);

                        if (parent->hold_piece != PIECE_N) {
                            int held = parent->hold_piece;
                            np_local = find_placements(parent->board, board_height, held,
                                                       local_pl, MAX_PLACEMENTS, NULL);
                            for (int i = 0; i < np_local; i++)
                                expand_and_insert(next_beam, &next_count, max_next,
                                    parent, board_height, held, &local_pl[i],
                                    spec_piece, qi + 1, new_bag, queue, queue_len);
                        }
                    }
                    continue;
                }

                int piece = queue[qi];

                // Place current piece (no hold)
                np_local = find_placements(parent->board, board_height, piece,
                                           local_pl, MAX_PLACEMENTS, NULL);
                for (int i = 0; i < np_local; i++)
                    expand_and_insert(next_beam, &next_count, max_next,
                        parent, board_height, piece, &local_pl[i],
                        parent->hold_piece, qi + 1, parent->bag_seen, queue, queue_len);

                // Hold swap
                if (parent->hold_piece != PIECE_N) {
                    int held = parent->hold_piece;
                    np_local = find_placements(parent->board, board_height, held,
                                               local_pl, MAX_PLACEMENTS, NULL);
                    for (int i = 0; i < np_local; i++)
                        expand_and_insert(next_beam, &next_count, max_next,
                            parent, board_height, held, &local_pl[i],
                            piece, qi + 1, parent->bag_seen, queue, queue_len);
                } else if (qi + 1 < queue_len) {
                    int play_piece = queue[qi + 1];
                    np_local = find_placements(parent->board, board_height, play_piece,
                                               local_pl, MAX_PLACEMENTS, NULL);
                    for (int i = 0; i < np_local; i++)
                        expand_and_insert(next_beam, &next_count, max_next,
                            parent, board_height, play_piece, &local_pl[i],
                            piece, qi + 2, parent->bag_seen, queue, queue_len);
                }
            }
        }

        next_beam_size = (next_count < max_next) ? next_count : max_next;

        // Record each root's best descendant score at this depth before pruning.
        if (want_roots) {
            for (int i = 0; i < next_beam_size; i++) {
                int ri = next_beam[i].depth0_placement_idx;
                if (ri >= 0 && ri < depth0_count && next_beam[i].score > root_best[ri])
                    root_best[ri] = next_beam[i].score;
            }
        }

        next_beam_size = finalize_beam(next_beam, next_beam_size, beam_width);

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
        if (have_fallback) {
            // All placements were pruned as dead - use the first available
            // placement as a last-resort move so the caller always gets a
            // valid action (prevents crashes from an all-PAD sequence).
            int fp = fallback_piece;
            int norm_col = fallback_pl.col + B2B_PIECES[fp].orientations[fallback_pl.rot].min_col;
            *out_action_index = fallback_is_hold * 160 + fallback_pl.rot * 40
                              + norm_col * 4 + fallback_pl.spin_type;
            b2b_last_placement = fallback_pl;
            b2b_write_sequence(fallback_meta, fallback_pl.bfs_state,
                               fallback_is_hold, max_len, out_best_sequence);
        } else {
            // Spawn completely blocked - no placement exists at all.
            *out_action_index = -1;
            for (int i = 0; i < max_len; i++) out_best_sequence[i] = KEY_PAD;
        }
        free(curr_beam);
        free(next_beam);
        return;
    }

    // Find best state by the SAME total order finalize_beam uses, so the chosen
    // move is deterministic on score ties (curr_beam is already sorted, so this
    // is index 0, but scan explicitly for safety).
    int best_idx = 0;
    for (int i = 1; i < curr_beam_size; i++) {
        if (beam_cmp(&curr_beam[i], &curr_beam[best_idx]) < 0) best_idx = i;
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

    // --- Per-root candidate output: every legal root placement is emitted (score =
    //     its own eval, raised to its best surviving descendant; near-death roots stay
    //     as very-low-scored candidates rather than being dropped).
    if (want_roots) {
        int n = 0;
        for (int ri = 0; ri < depth0_count && n < max_roots; ri++) {
            Placement* rpl = &depth0_placements[ri];
            int rih = depth0_is_hold[ri];
            int rpiece = !rih ? active_piece
                              : ((hold_piece != PIECE_N) ? hold_piece : queue[0]);
            int rcol = rpl->col + B2B_PIECES[rpiece].orientations[rpl->rot].min_col;
            out_root_action_indices[n] = rih * 160 + rpl->rot * 40 + rcol * 4 + rpl->spin_type;
            out_root_scores[n] = root_best[ri];
            if (out_root_landing_rows) out_root_landing_rows[n] = rpl->landing_row;
            BFSStateMeta* rmeta = rih ? depth0_meta_hold : depth0_meta_active;
            b2b_write_sequence(rmeta, rpl->bfs_state, rih, max_len,
                               &out_root_sequences[(size_t)n * max_len]);
            n++;
        }
        *out_num_roots = n;
    }

    free(curr_beam);
    free(next_beam);
}

// ============================================================
// Full Game Loop in C (for optimizer - no Python overhead)
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
    int max_combo;
    float avg_combo;   // mean combo over placements that cleared >=1 line
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
        int bh = 40; // board height (20 visible + 20 buffer)

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
        int max_combo_val = 0;
        long combo_sum = 0;     // sum of combo over clearing placements
        int combo_clears = 0;   // number of clearing placements
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

            // --- Beam search (reuses b2b_search_c - sequence is discarded) ---
            int action_idx;
            int64_t dummy_seq[15];
            b2b_search_c(
                board, bh, active_piece, hold_piece,
                pq.queue, pq.queue_len, b2b, combo, tg,
                cfg.garbage_push_delay,
                (int)cur_bag_seen,
                search_depth, beam_width, 15,
                &action_idx, dummy_seq,
                0, NULL, NULL, NULL, NULL, NULL   // no per-root output in the C game loop
            );

            if (action_idx < 0) { died = true; break; }

            // --- Read placement directly from b2b_search_c's result ---
            // (avoids lossy action_idx round-trip - preserves BFS landing row)
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

            // Track combo (combo counter is post-clear; >0 means a chain is live)
            if (clears > 0) {
                if (combo > max_combo_val) max_combo_val = combo;
                combo_sum += combo;
                combo_clears++;
            }

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

        // Credit banked surge potential: a survived game ending on a held b2b chain (>=4) has
        // unrealized attack the fixed-length window cut off (surge releases `b2b` on break).
        // Without this, hoarding strategies are undercounted vs continuous-cash-out ones.
        if (!died && b2b >= 4) total_attack += (float)b2b;

        res->steps_completed = steps_done;
        res->survived = died ? 0 : 1;
        res->total_attack = total_attack;
        res->max_b2b = max_b2b_val;
        res->end_height = died ? 20 : last_height;
        res->avg_height = (steps_done > 0) ? height_sum / (float)steps_done : 0.0f;
        res->max_height = peak_height;
        res->max_combo = max_combo_val;
        res->avg_combo = (combo_clears > 0) ? (float)combo_sum / (float)combo_clears : 0.0f;
    }
}

// Apply one placement to a plain-occupancy bitboard and score it - the deterministic
// core of PyTetrisEnv._lock_piece + Scorer.judge, reused by the MCTS placement step.
// `board` is in/out (length board_height, plain occupancy, no GARB markers); it is
// mutated to the post-lock, post-clear board. `norm_col` is the dense-action column
// (pl->col + min_col); `landing_row` is the BFS lock row; `spin_type` is the enumerator's
// classification. Garbage / stats / reward stay in Python.
void b2b_lock_score_c(uint16_t* board, int board_height,
                      int piece_type, int rot, int norm_col, int landing_row,
                      int spin_type, int b2b, int combo,
                      int* out_clears, float* out_attack,
                      int* out_new_b2b, int* out_new_combo) {
    b2b_init_pieces();

    int col = norm_col - B2B_PIECES[piece_type].orientations[rot].min_col;
    lock_piece_on_board(board, board_height, piece_type, rot, landing_row, col);
    int clears = clear_lines(board, board_height);

    bool perfect_clear = true;
    for (int r = 0; r < board_height; r++) {
        if (board[r] != 0) { perfect_clear = false; break; }
    }

    AttackResult ar = compute_attack(clears, spin_type, b2b, combo, perfect_clear);
    *out_clears = clears;
    *out_attack = ar.attack;
    *out_new_b2b = ar.new_b2b;
    *out_new_combo = ar.new_combo;
}


// ============================================================
// Fully-C MCTS sim engine (OpenMP-threaded across games/trees)
// ------------------------------------------------------------
// The entire PUCT simulation loop runs in C on a compact bitboard+scalars node;
// only the TF policy/value net stays in Python. The engine keeps a persistent tree
// per game across all sims of a move and ping-pongs to Python once per round for the
// batched net eval (collect_leaves -> net -> apply_leaves). Reward = per-edge
// w_attack*attack (clipped) with an unclipped -w_death on terminal edges; the leaf
// bootstrap is the net value directly. Dirichlet noise + final sampling stay in Python.
// ============================================================

// Reentrant env-pathfinder enumeration (pathfinder.c, linked into this extension).
void find_placement_candidates_c(const uint16_t* board_rows, int board_height,
                                 int piece_type, int start_row, int start_col, int start_rot,
                                 int max_len, int is_hold,
                                 int64_t* out_sequences, int32_t* out_landing_rows);

#define MCAP 128          // candidate capacity (slots)
#define MBRANCH 64        // per-branch cap (no-hold 0..63, hold 64..127)
#define MBH 40            // board height (20 visible + 20 buffer)
#define MAXVQ 16          // visible-queue storage
#define MCLIP 10.0f       // reward clip
#define MAX_PATH 1024
#define MAX_LPR 16        // max leaves collected per tree per round (intra-tree batching)

typedef struct {
    uint16_t board[MBH];
    int active, hold;             // piece-type enums (0=N..7)
    int queue[MAXVQ]; int qlen;
    TetrioRNG rng;                // queue-refill PRNG (mirrors env _tetrio_rng)
    int pending[7]; int pending_pos, pending_len;  // _next_bag remainder
    GarbEntry gq[MAX_GARB_ENTRIES]; int gcnt;
    int b2b, combo;
} MState;

typedef struct {
    int board_height, queue_size, max_holes, garbage_push_delay;
    int auto_push_garbage, auto_fill_queue;
    float c_puct, gamma, w_attack, w_death, return_scale, w_b2b;
    int max_len;
    int leaves_per_round;        // L: leaves collected per tree per net round (>=1)
    float vloss;                 // virtual-loss magnitude (scaled-Q units)
} MConfig;

typedef struct MNode {
    MState st;
    bool terminal, expanded;
    bool awaiting_eval;          // expanded this round, priors/value not set yet (collision marker)
    float value;
    int legal[MCAP]; int n_legal;
    int desc[MCAP][5];            // (is_hold, rot, norm_col, landing_row, spin) per legal slot
    float prior[MCAP], N[MCAP], W[MCAP], Q[MCAP], edge_reward[MCAP];
    struct MNode* child[MCAP];
} MNode;

typedef struct { MNode* node; int slot; } PathEntry;

typedef struct {
    MNode* root;
    bool alive;                  // a live game with a spawnable (enumerable) root
    bool dead;                   // root had no legal placement
    // per-tree bump arena (sized to root + one node per simulation). Pre-allocated
    // single-threaded so node alloc inside the OpenMP region never hits the malloc lock.
    MNode* pool; int pool_used, pool_cap;
    // per-round pending leaves awaiting net eval (intra-tree batching: up to L per round).
    // path[p]/path_len[p] is the descent path of pending[p], kept so apply_leaves can revert
    // its virtual loss and back up the real value.
    PathEntry path[MAX_LPR][MAX_PATH]; int path_len[MAX_LPR];
    MNode* pending[MAX_LPR]; int n_pending;
} MTree;

typedef struct {
    int num_trees;
    int max_nodes;
    int n_threads;               // capped OpenMP thread count (see mcts_create)
    int prev_omp_threads;        // process OMP default, restored on destroy
    MConfig cfg;
    MTree* trees;
} MEngine;

// --- node arena (bump allocator; no malloc in the parallel region) ---
static MNode* mtree_alloc(MTree* t) {
    if (t->pool_used >= t->pool_cap) return NULL;  // budget exhausted (sized to num_sims+1)
    MNode* n = &t->pool[t->pool_used++];
    memset(n, 0, sizeof(MNode));
    return n;
}

// --- piece queue (mirror env _fill_queue: pending FIFO then fresh TetrioRNG bag) ---
static int mstate_draw(MState* s) {
    if (s->pending_len <= 0) {
        rng_next_bag(&s->rng, s->pending);
        s->pending_pos = 0; s->pending_len = 7;
    }
    int p = s->pending[s->pending_pos++]; s->pending_len--;
    return p;
}
static int mstate_pop(MState* s) {
    int p = s->queue[0];
    for (int i = 0; i < s->qlen - 1; i++) s->queue[i] = s->queue[i + 1];
    s->qlen--;
    return p;
}
static void mstate_fill(MState* s, const MConfig* cfg) {
    while (s->qlen < cfg->queue_size) s->queue[s->qlen++] = mstate_draw(s);
}

// --- enclosed-hole count (replica of hole_finder.c count_enclosed_holes; terminal check) ---
static int mcts_count_holes(const uint16_t* board, int bh) {
    bool visited[MBH * BOARD_COLS];
    for (int i = 0; i < bh * BOARD_COLS; i++) visited[i] = false;
    int q[MBH * BOARD_COLS]; int head = 0, tail = 0;
    for (int c = 0; c < BOARD_COLS; c++) {
        if ((board[0] & (1 << c)) == 0) { visited[c] = true; q[tail++] = c; }
    }
    int dr[4] = {1, 0, 0, -1}, dc[4] = {0, -1, 1, 0};
    while (head != tail) {
        int cur = q[head++]; int r = cur / BOARD_COLS, c = cur % BOARD_COLS;
        for (int i = 0; i < 4; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr < 0 || nr >= bh || nc < 0 || nc >= BOARD_COLS) continue;
            int nidx = nr * BOARD_COLS + nc;
            if (visited[nidx]) continue;
            if ((board[nr] & (1 << nc)) == 0) { visited[nidx] = true; q[tail++] = nidx; }
        }
    }
    int holes = 0;
    for (int r = 0; r < bh; r++)
        for (int c = 0; c < BOARD_COLS; c++)
            if ((board[r] & (1 << c)) == 0 && !visited[r * BOARD_COLS + c]) holes++;
    return holes;
}

// --- one placement step (mirror placement_step / the b2b game-loop body); returns raw attack ---
static float mcts_apply_step(MState* s, const MConfig* cfg, const int* d, bool* out_terminal) {
    int is_hold = d[0], rot = d[1], norm_col = d[2], landing_row = d[3], spin = d[4];
    int played;
    if (is_hold) {
        played = (s->hold == PIECE_N) ? mstate_pop(s) : s->hold;
        s->hold = s->active;
    } else {
        played = s->active;
    }
    int col = norm_col - B2B_PIECES[played].orientations[rot].min_col;
    lock_piece_on_board(s->board, cfg->board_height, played, rot, landing_row, col);
    int clears = clear_lines(s->board, cfg->board_height);
    bool pc = true;
    for (int r = 0; r < cfg->board_height; r++) if (s->board[r] != 0) { pc = false; break; }
    AttackResult ar = compute_attack(clears, spin, s->b2b, s->combo, pc);
    s->b2b = ar.new_b2b; s->combo = ar.new_combo;
    float attack = ar.attack;
    s->active = mstate_pop(s);

    bool top_out = board_topped_out(s->board, cfg->board_height);

    if (attack > 0) garb_cancel(s->gq, &s->gcnt, (int)attack);
    if (cfg->auto_push_garbage && clears == 0) {
        garb_tick(s->gq, s->gcnt);
        garb_push_one(s->board, cfg->board_height, s->gq, &s->gcnt);
    }
    // _add_to_garbage_queue is a no-op in sims (garbage_chance=0).
    if (cfg->auto_push_garbage && cfg->garbage_push_delay == 0)
        garb_push_all(s->board, cfg->board_height, s->gq, &s->gcnt);

    bool garbage_top_out = board_topped_out(s->board, cfg->board_height);
    bool exceeded_holes = false;
    if (cfg->max_holes >= 0)
        exceeded_holes = mcts_count_holes(s->board, cfg->board_height) > cfg->max_holes;

    *out_terminal = top_out || exceeded_holes || garbage_top_out;
    if (cfg->auto_fill_queue) mstate_fill(s, cfg);
    return attack;
}

// --- enumerate candidates into node (reuse env pathfinder); false if dead (no legal) ---
static bool mcts_enumerate(MNode* node, const MConfig* cfg) {
    static __thread int32_t lr_nh[160], lr_h[160];
    static __thread int64_t seq_scratch[160 * 32];   // discarded; max_len<=32
    const uint16_t* board = node->st.board;
    int ml = cfg->max_len;
    find_placement_candidates_c(board, cfg->board_height, node->st.active, SPAWN_ROW, 3, 0,
                                ml, 0, seq_scratch, lr_nh);
    int holdpiece = node->st.hold != PIECE_N ? node->st.hold : node->st.queue[0];
    find_placement_candidates_c(board, cfg->board_height, holdpiece, SPAWN_ROW, 3, 0,
                                ml, 1, seq_scratch, lr_h);
    node->n_legal = 0;
    int cnt = 0;
    for (int i = 0; i < 160 && cnt < MBRANCH; i++) {
        if (lr_nh[i] < 0) continue;
        int slot = cnt++;
        node->legal[node->n_legal++] = slot;
        node->desc[slot][0] = 0; node->desc[slot][1] = i / 40;
        node->desc[slot][2] = (i % 40) / 4; node->desc[slot][3] = lr_nh[i];
        node->desc[slot][4] = i % 4;
    }
    cnt = 0;
    for (int i = 0; i < 160 && cnt < MBRANCH; i++) {
        if (lr_h[i] < 0) continue;
        int slot = MBRANCH + cnt++;
        node->legal[node->n_legal++] = slot;
        node->desc[slot][0] = 1; node->desc[slot][1] = i / 40;
        node->desc[slot][2] = (i % 40) / 4; node->desc[slot][3] = lr_h[i];
        node->desc[slot][4] = i % 4;
    }
    return node->n_legal > 0;
}

// --- build the net-input request row for a node ---
static void mcts_fill_request(const MNode* node, const MConfig* cfg, float* board_out,
                              int64_t* pieces_out, float* bcg_out, float* pls_out, uint8_t* mask_out) {
    const MState* s = &node->st;
    // Emit only the model-visible slice (bottom NET_ROWS rows) so the net contract stays (24,10,1).
    for (int r = 0; r < NET_ROWS; r++)
        for (int c = 0; c < BOARD_COLS; c++)
            board_out[r * BOARD_COLS + c] =
                (float)((s->board[(cfg->board_height - NET_ROWS) + r] >> c) & 1);
    pieces_out[0] = s->active; pieces_out[1] = s->hold;
    for (int i = 0; i < cfg->queue_size; i++) pieces_out[2 + i] = s->queue[i];
    bcg_out[0] = (float)s->b2b; bcg_out[1] = (float)s->combo;
    bcg_out[2] = (float)garb_total(s->gq, s->gcnt);
    int W = 18;
    for (int j = 0; j < MCAP * W; j++) pls_out[j] = 0.0f;
    for (int j = 0; j < MCAP; j++) mask_out[j] = 0;
    int queue0 = s->qlen > 0 ? s->queue[0] : 0;
    for (int k = 0; k < node->n_legal; k++) {
        int slot = node->legal[k];
        const int* d = node->desc[slot];
        int is_hold = d[0], rot = d[1], norm_col = d[2], landing = d[3], spin = d[4];
        int piece = is_hold == 0 ? s->active : (s->hold != PIECE_N ? s->hold : queue0);
        float* f = &pls_out[(size_t)slot * W];
        if (piece >= 1 && piece <= 7) f[piece - 1] = 1.0f;
        f[7 + rot] = 1.0f;
        f[11] = norm_col / 9.0f;
        // landing row relative to the visible window (matches training)
        float rn = (landing - (cfg->board_height - NET_ROWS)) / (float)(NET_ROWS - 1);
        f[12] = rn < 0.0f ? 0.0f : (rn > 1.0f ? 1.0f : rn);
        f[13 + spin] = 1.0f;
        f[17] = (float)is_hold;
        mask_out[slot] = 1;
    }
}

// --- PUCT ---
// Q is used in return_scale units directly (no per-tree min-max): min-max pins the worst
// edge to 0 and the best to 1, which caps the death penalty's pull at one normalized unit
// regardless of w_death. Raw return_scale units let a large w_death actually dominate.
static int mcts_select(const MNode* node, const MConfig* cfg) {
    float total = 0.0f;
    for (int k = 0; k < node->n_legal; k++) total += node->N[node->legal[k]];
    float best = -1e30f; int best_slot = node->legal[0];
    float sq = sqrtf(total + 1e-8f);
    for (int k = 0; k < node->n_legal; k++) {
        int slot = node->legal[k];
        float n = node->N[slot];
        float q = n > 0 ? node->Q[slot] : 0.0f;
        float u = cfg->c_puct * node->prior[slot] * sq / (1.0f + n);
        float score = q + u;
        if (score > best) { best = score; best_slot = slot; }
    }
    return best_slot;
}
static void mtree_backup(const MConfig* cfg, const PathEntry* path, int len,
                         float leaf_value) {
    float g = leaf_value;
    for (int i = len - 1; i >= 0; i--) {
        MNode* node = path[i].node; int slot = path[i].slot;
        g = node->edge_reward[slot] + cfg->gamma * g;
        node->N[slot] += 1.0f;
        node->W[slot] += g;
        node->Q[slot] = node->W[slot] / node->N[slot];
    }
}

// Virtual loss: pessimize each traversed edge so the next descent in the same round diverges.
// The inverse revert in apply_leaves restores W/N exactly before the real backup.
static void mtree_apply_vloss(const PathEntry* path, int len, float vloss) {
    for (int i = 0; i < len; i++) {
        MNode* node = path[i].node; int slot = path[i].slot;
        node->N[slot] += 1.0f;
        node->W[slot] -= vloss;
        node->Q[slot] = node->W[slot] / node->N[slot];
    }
}
static void mtree_revert_vloss(const PathEntry* path, int len, float vloss) {
    for (int i = 0; i < len; i++) {
        MNode* node = path[i].node; int slot = path[i].slot;
        node->N[slot] -= 1.0f;
        node->W[slot] += vloss;
        node->Q[slot] = node->N[slot] > 0.0f ? node->W[slot] / node->N[slot] : 0.0f;
    }
}

// b2b potential for the w_b2b shaping: Phi = min(max(0, b2b), CAP).
#define B2B_POTENTIAL_CAP 12
static inline float b2b_phi(int b2b) {
    int p = b2b > 0 ? b2b : 0;
    return (float)(p < B2B_POTENTIAL_CAP ? p : B2B_POTENTIAL_CAP);
}

static float mcts_scale_reward(const MConfig* cfg, float reward) {
    float r = reward / (cfg->return_scale + 1e-8f);
    if (r > MCLIP) r = MCLIP; else if (r < -MCLIP) r = -MCLIP;
    return r;
}

// --- one round for a tree: collect up to L leaves via virtual loss. Fills t->pending[0..n_pending)
//     and their paths; terminal/dead leaves back up in-place; stops on collision or arena-full. ---
static void mcts_collect_round(MTree* t, const MConfig* cfg) {
    t->n_pending = 0;
    int L = cfg->leaves_per_round;
    for (int li = 0; li < L && t->n_pending < MAX_LPR; li++) {
        PathEntry* path = t->path[t->n_pending];   // build into the next pending slot's buffer
        int plen = 0;
        MNode* node = t->root;
        while (1) {
            if (node->terminal || node->n_legal == 0) {  // dead end: real backup of 0 in-place
                mtree_backup(cfg, path, plen, 0.0f);
                break;
            }
            if (node->awaiting_eval) break;              // collision: another descent owns this leaf
            int slot = mcts_select(node, cfg);
            path[plen].node = node; path[plen].slot = slot; plen++;
            MNode* child = node->child[slot];
            if (child == NULL) {
                MNode* leaf = mtree_alloc(t);
                if (leaf == NULL) { mtree_backup(cfg, path, plen, 0.0f); break; }  // arena full
                leaf->st = node->st;
                bool terminal = false;
                float attack = mcts_apply_step(&leaf->st, cfg, node->desc[slot], &terminal);
                node->child[slot] = leaf;
                // Death (top-out/holes, or a resulting no-legal position) is the loss signal.
                // The attack part goes through the reward clip; the w_death penalty is applied
                // UNCLIPPED (return_scale units only) so raising w_death actually deepens the
                // terminal Q instead of saturating at -MCLIP and being tuned away.
                bool dead = terminal || !mcts_enumerate(leaf, cfg);
                if (dead) {
                    leaf->terminal = true;
                    // Terminal Phi=0: refund the parent's b2b potential.
                    node->edge_reward[slot] =
                        mcts_scale_reward(cfg, cfg->w_attack * attack)
                        - cfg->w_death / (cfg->return_scale + 1e-8f)
                        - cfg->w_b2b * b2b_phi(node->st.b2b)
                              / (cfg->return_scale + 1e-8f);
                    mtree_backup(cfg, path, plen, 0.0f);
                    break;
                }
                // Potential-based b2b shaping: w_b2b*(gamma*Phi(child) - Phi(parent)).
                node->edge_reward[slot] =
                    mcts_scale_reward(cfg, cfg->w_attack * attack)
                    + cfg->w_b2b
                          * (cfg->gamma * b2b_phi(leaf->st.b2b) - b2b_phi(node->st.b2b))
                          / (cfg->return_scale + 1e-8f);
                leaf->awaiting_eval = true;
                t->pending[t->n_pending] = leaf;
                t->path_len[t->n_pending] = plen;
                t->n_pending++;
                mtree_apply_vloss(path, plen, cfg->vloss);  // steer the next descent away
                break;
            }
            node = child;
        }
    }
}

static void msoftmax_into_prior(MNode* node, const float* logits) {
    float mx = -1e30f;
    for (int k = 0; k < node->n_legal; k++) { float l = logits[node->legal[k]]; if (l > mx) mx = l; }
    float sum = 0.0f;
    for (int k = 0; k < node->n_legal; k++) { float e = expf(logits[node->legal[k]] - mx); node->prior[node->legal[k]] = e; sum += e; }
    for (int k = 0; k < node->n_legal; k++) node->prior[node->legal[k]] /= sum;
}

// ============================================================
// Exported protocol
// ============================================================

void* mcts_create(int num_trees, int board_height, int queue_size,
                  int max_holes, int garbage_push_delay, int auto_push_garbage, int auto_fill_queue,
                  float c_puct, float gamma, float w_attack, float w_death,
                  float return_scale, int max_len, int max_nodes,
                  int leaves_per_round, float vloss, float w_b2b) {
    b2b_init_pieces();
    // Prime the pathfinder's init_pieces() single-threaded before any parallel enumerate.
    { uint16_t b[MBH]; memset(b, 0, sizeof(b)); int32_t lr[160]; int64_t sq[160 * 32];
      find_placement_candidates_c(b, board_height, PIECE_I, SPAWN_ROW, 3, 0, max_len, 0, sq, lr); }
    MEngine* e = (MEngine*)calloc(1, sizeof(MEngine));
    e->num_trees = num_trees;
    e->max_nodes = max_nodes;
    // Per-round C work is tiny (one small tree per game), so the net dominates and the
    // search is net-bound. A few threads beat both serial and all-cores: using every
    // logical core oversubscribes this small work and is slower than serial. Cap low
    // (~quarter of logical cores), <= num_trees, and let OMP_NUM_THREADS lower it further.
    int cap = omp_get_num_procs() / 4; if (cap < 1) cap = 1;
    int envmax = omp_get_max_threads();
    e->n_threads = num_trees;
    if (e->n_threads > cap) e->n_threads = cap;
    if (e->n_threads > envmax) e->n_threads = envmax;
    // Size the pool itself (not just the per-region count): libgomp's default pool spans all
    // logical cores and its idle threads busy-wait, stealing CPU from the TF net threads.
    // Save/restore the process default (mcts_destroy) so we don't shrink other OpenMP users
    // (e.g. the b2b beam search) if they share the process.
    e->prev_omp_threads = omp_get_max_threads();
    omp_set_num_threads(e->n_threads);
    e->cfg.board_height = board_height; e->cfg.queue_size = queue_size;
    e->cfg.max_holes = max_holes;
    e->cfg.garbage_push_delay = garbage_push_delay;
    e->cfg.auto_push_garbage = auto_push_garbage; e->cfg.auto_fill_queue = auto_fill_queue;
    e->cfg.c_puct = c_puct; e->cfg.gamma = gamma;
    e->cfg.w_attack = w_attack; e->cfg.w_death = w_death;
    e->cfg.return_scale = return_scale; e->cfg.w_b2b = w_b2b;
    e->cfg.max_len = max_len;
    if (leaves_per_round < 1) leaves_per_round = 1;
    if (leaves_per_round > MAX_LPR) leaves_per_round = MAX_LPR;
    e->cfg.leaves_per_round = leaves_per_round;
    e->cfg.vloss = vloss;
    e->trees = (MTree*)calloc(num_trees, sizeof(MTree));
    for (int i = 0; i < num_trees; i++) {
        e->trees[i].pool = (MNode*)calloc((size_t)max_nodes, sizeof(MNode));
        e->trees[i].pool_cap = max_nodes;
    }
    return e;
}

// Set one tree's root state from a live env snapshot.
void mcts_set_root(void* h, int tree, const uint16_t* board, int active, int hold,
                   const int* queue, int qlen, int b2b, int combo,
                   int64_t rng_t, const int* pending, int pending_len,
                   const int* garb_rows, const int* garb_col, const int* garb_timer, int gcnt) {
    MEngine* e = (MEngine*)h;
    MTree* t = &e->trees[tree];
    t->root = NULL; t->alive = false; t->dead = false;
    t->pool_used = 0; t->n_pending = 0;
    MNode* root = mtree_alloc(t);
    MState* s = &root->st;
    memset(s, 0, sizeof(*s));
    for (int r = 0; r < e->cfg.board_height; r++) s->board[r] = board[r];
    s->active = active; s->hold = hold;
    s->qlen = qlen; for (int i = 0; i < qlen; i++) s->queue[i] = queue[i];
    s->b2b = b2b; s->combo = combo;
    s->rng.t = rng_t;
    s->pending_len = pending_len; s->pending_pos = 0;
    for (int i = 0; i < pending_len; i++) s->pending[i] = pending[i];
    s->gcnt = gcnt;
    for (int i = 0; i < gcnt; i++) { s->gq[i].rows = garb_rows[i]; s->gq[i].col = garb_col[i]; s->gq[i].timer = garb_timer[i]; }
    t->root = root;
}

// Enumerate all roots (parallel). Emits net-input rows for live roots; dead roots flagged.
// Returns nv (#rows); tree_ids[k] = tree index for row k.
int mcts_collect_roots(void* h, float* boards, int64_t* pieces, float* bcg,
                       float* pls, uint8_t* masks, int* tree_ids) {
    MEngine* e = (MEngine*)h;
    const MConfig* cfg = &e->cfg;
    int pw = 2 + cfg->queue_size;
    #pragma omp parallel for schedule(dynamic) num_threads(e->n_threads)
    for (int i = 0; i < e->num_trees; i++) {
        MTree* t = &e->trees[i];
        if (t->root == NULL) { t->dead = true; continue; }
        if (!mcts_enumerate(t->root, cfg)) { t->dead = true; t->alive = false; }
        else { t->alive = true; }
    }
    int nv = 0;
    for (int i = 0; i < e->num_trees; i++) {
        MTree* t = &e->trees[i];
        if (!t->alive) continue;
        mcts_fill_request(t->root, cfg, &boards[(size_t)nv * NET_ROWS * BOARD_COLS],
                          &pieces[(size_t)nv * pw], &bcg[(size_t)nv * 3],
                          &pls[(size_t)nv * MCAP * 18], &masks[(size_t)nv * MCAP]);
        tree_ids[nv] = i; nv++;
    }
    return nv;
}

// Apply root net eval + injected Dirichlet noise (Python-generated, one row per live tree
// in the same order as collect_roots emitted). dir_noise is [nv * MCAP] (only legal slots used).
void mcts_apply_roots(void* h, const float* logits, const float* values,
                      const float* dir_noise, float dir_eps) {
    MEngine* e = (MEngine*)h;
    int row = 0;
    for (int i = 0; i < e->num_trees; i++) {
        MTree* t = &e->trees[i];
        if (!t->alive) continue;
        MNode* root = t->root;
        root->value = values[row];
        root->expanded = true;
        msoftmax_into_prior(root, &logits[(size_t)row * MCAP]);
        const float* noise = &dir_noise[(size_t)row * MCAP];
        for (int k = 0; k < root->n_legal; k++) {
            int slot = root->legal[k];
            root->prior[slot] = (1.0f - dir_eps) * root->prior[slot] + dir_eps * noise[slot];
        }
        row++;
    }
}

// One simulation round: collect up to L leaves per live tree (parallel), emit leaf net-inputs.
// Emission order is tree-major then pending-within-tree; apply_leaves must consume it identically.
int mcts_collect_leaves(void* h, float* boards, int64_t* pieces, float* bcg,
                        float* pls, uint8_t* masks, int* tree_ids) {
    MEngine* e = (MEngine*)h;
    const MConfig* cfg = &e->cfg;
    int pw = 2 + cfg->queue_size;
    #pragma omp parallel for schedule(dynamic) num_threads(e->n_threads)
    for (int i = 0; i < e->num_trees; i++) {
        MTree* t = &e->trees[i];
        if (!t->alive) continue;
        mcts_collect_round(t, cfg);   // fills t->pending[0..n_pending) (+ their vloss paths)
    }
    int nv = 0;
    for (int i = 0; i < e->num_trees; i++) {
        MTree* t = &e->trees[i];
        if (!t->alive) continue;
        for (int p = 0; p < t->n_pending; p++) {
            mcts_fill_request(t->pending[p], cfg, &boards[(size_t)nv * NET_ROWS * BOARD_COLS],
                              &pieces[(size_t)nv * pw], &bcg[(size_t)nv * 3],
                              &pls[(size_t)nv * MCAP * 18], &masks[(size_t)nv * MCAP]);
            tree_ids[nv] = i; nv++;
        }
    }
    return nv;
}

// Apply leaf net eval + backup (rows match collect_leaves' emitted order). For each pending leaf:
// set priors+value, revert its virtual loss, then back up the real bootstrap along its path.
void mcts_apply_leaves(void* h, const float* logits, const float* values) {
    MEngine* e = (MEngine*)h;
    const MConfig* cfg = &e->cfg;
    int row = 0;
    for (int i = 0; i < e->num_trees; i++) {
        MTree* t = &e->trees[i];
        if (!t->alive) continue;
        for (int p = 0; p < t->n_pending; p++) {
            MNode* leaf = t->pending[p];
            leaf->value = values[row];
            leaf->expanded = true;
            leaf->awaiting_eval = false;
            msoftmax_into_prior(leaf, &logits[(size_t)row * MCAP]);
            // Bootstrap is the net value directly; the net learns V_shaped = V_true - Phi
            // (b2b's worth is carried by the per-edge potential shaping, not added here).
            float boot = leaf->value;
            mtree_revert_vloss(t->path[p], t->path_len[p], cfg->vloss);
            mtree_backup(cfg, t->path[p], t->path_len[p], boot);
            row++;
        }
    }
}

// Read per-tree root visit counts + descriptors. pi/counts are [num_trees*MCAP];
// root_desc is [num_trees*MCAP*5]; dead[num_trees] (1 if no move).
void mcts_result(void* h, float* pi, float* counts, int* root_desc, int* dead) {
    MEngine* e = (MEngine*)h;
    for (int i = 0; i < e->num_trees; i++) {
        MTree* t = &e->trees[i];
        for (int j = 0; j < MCAP; j++) { pi[(size_t)i * MCAP + j] = 0.0f; counts[(size_t)i * MCAP + j] = 0.0f; }
        for (int j = 0; j < MCAP * 5; j++) root_desc[(size_t)i * MCAP * 5 + j] = -1;
        if (!t->alive || t->root == NULL) { dead[i] = 1; continue; }
        dead[i] = 0;
        MNode* root = t->root;
        float total = 0.0f;
        for (int k = 0; k < root->n_legal; k++) total += root->N[root->legal[k]];
        for (int k = 0; k < root->n_legal; k++) {
            int slot = root->legal[k];
            counts[(size_t)i * MCAP + slot] = root->N[slot];
            pi[(size_t)i * MCAP + slot] = total > 0 ? root->N[slot] / total : root->prior[slot];
            for (int d = 0; d < 5; d++) root_desc[((size_t)i * MCAP + slot) * 5 + d] = root->desc[slot][d];
        }
    }
}

void mcts_destroy(void* h) {
    MEngine* e = (MEngine*)h;
    if (!e) return;
    omp_set_num_threads(e->prev_omp_threads);
    for (int i = 0; i < e->num_trees; i++) free(e->trees[i].pool);
    free(e->trees);
    free(e);
}

// --- parity hooks (single-state enumerate / step; drive the /tmp gates that re-verify the
//     deterministic core after a subtree re-sync re-applies these edits) ---
// Enumerate one state; out_desc[MCAP*5] (-1 empty), returns n_legal.
int mcts_debug_enum(const uint16_t* board, int board_height, int active, int hold, int queue0,
                    int max_len, int* out_desc) {
    MConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.board_height = board_height; cfg.max_len = max_len;
    b2b_init_pieces();
    MNode node; memset(&node, 0, sizeof(node));
    for (int r = 0; r < board_height; r++) node.st.board[r] = board[r];
    node.st.active = active; node.st.hold = hold; node.st.queue[0] = queue0; node.st.qlen = 1;
    mcts_enumerate(&node, &cfg);
    for (int j = 0; j < MCAP * 5; j++) out_desc[j] = -1;
    for (int k = 0; k < node.n_legal; k++) {
        int slot = node.legal[k];
        for (int d = 0; d < 5; d++) out_desc[slot * 5 + d] = node.desc[slot][d];
    }
    return node.n_legal;
}

// Apply one step to a serialized state; returns raw attack, writes post-step state out.
float mcts_debug_step(uint16_t* board, int board_height, int max_holes,
                      int garbage_push_delay, int queue_size,
                      int* active, int* hold, int* queue, int* qlen,
                      int64_t* rng_t, int* pending, int* pending_len,
                      int* garb_rows, int* garb_col, int* garb_timer, int* gcnt,
                      int* b2b, int* combo, const int* desc, int* out_terminal) {
    MConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.board_height = board_height; cfg.max_holes = max_holes;
    cfg.garbage_push_delay = garbage_push_delay; cfg.queue_size = queue_size;
    cfg.auto_push_garbage = 1; cfg.auto_fill_queue = 1;
    b2b_init_pieces();
    MState s; memset(&s, 0, sizeof(s));
    for (int r = 0; r < board_height; r++) s.board[r] = board[r];
    s.active = *active; s.hold = *hold; s.qlen = *qlen;
    for (int i = 0; i < *qlen; i++) s.queue[i] = queue[i];
    s.rng.t = *rng_t; s.pending_len = *pending_len; s.pending_pos = 0;
    for (int i = 0; i < *pending_len; i++) s.pending[i] = pending[i];
    s.gcnt = *gcnt;
    for (int i = 0; i < *gcnt; i++) { s.gq[i].rows = garb_rows[i]; s.gq[i].col = garb_col[i]; s.gq[i].timer = garb_timer[i]; }
    s.b2b = *b2b; s.combo = *combo;
    bool term = false;
    float attack = mcts_apply_step(&s, &cfg, desc, &term);
    for (int r = 0; r < board_height; r++) board[r] = s.board[r];
    *active = s.active; *hold = s.hold; *qlen = s.qlen;
    for (int i = 0; i < s.qlen; i++) queue[i] = s.queue[i];
    *rng_t = s.rng.t; *pending_len = s.pending_len;
    for (int i = 0; i < s.pending_len; i++) pending[i] = s.pending[s.pending_pos + i];
    *gcnt = s.gcnt;
    for (int i = 0; i < s.gcnt; i++) { garb_rows[i] = s.gq[i].rows; garb_col[i] = s.gq[i].col; garb_timer[i] = s.gq[i].timer; }
    *b2b = s.b2b; *combo = s.combo;
    *out_terminal = term ? 1 : 0;
    return attack;
}
