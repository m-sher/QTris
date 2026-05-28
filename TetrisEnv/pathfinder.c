#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

#define BOARD_ROWS 40
#define VISIBLE_ROWS 20
#define BOARD_COLS 10
#define ROTATIONS 4
#define SPIN_STATES 2

// Keys
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

// Max Queue Size
#define QUEUE_CAPACITY 8192 // Generous buffer
#define STATE_SPACE (BOARD_ROWS * BOARD_COLS * ROTATIONS)
#define BASE_POSITIONS 80 // 4 rot * 10 cols * 2 spins
#define MAX_LANDINGS 24  // Max distinct landing rows per base slot

typedef struct {
    uint16_t row_masks[4]; 
    int min_col;
    int max_col;
    int min_row;
    int max_row;
    int row_offsets[4]; // Offset of each row in the 4x4 grid relative to top-left
} PieceOrientation;

typedef struct {
    PieceOrientation orientations[4];
} PieceDef;

// State tracking
typedef struct {
    int16_t parent;      // Index of parent state in visited/queue logic? 
                         // No, we need parent state index.
    int8_t last_move;
    int16_t depth;
    int8_t delta_r;
    int8_t delta_row;
    int8_t delta_col;
} StateMeta;

// Globals
static PieceDef PIECES[8];
static bool initialized = false;

// Kicks
// Flattened kick tables: [from_rot][to_rot][kick_index][0=row, 1=col]
// 4 rotations -> 4x4 transitions. Each has 5 kicks (SRS). Each kick is (dr, dc).
static int8_t KICKS[4][4][5][2]; 
static int8_t I_KICKS[4][4][5][2];

// -- Initialization --

void init_pieces() {
    if (initialized) return;

    // Zero out everything first
    memset(PIECES, 0, sizeof(PIECES));
    memset(KICKS, 0, sizeof(KICKS));
    memset(I_KICKS, 0, sizeof(I_KICKS));

    // Helper to set orientations
    // I Piece
    // 0: [[1,0],[1,1],[1,2],[1,3]] -> Row 1: 1111(bin)=15. 
    PIECES[PIECE_I].orientations[0] = (PieceOrientation){ .row_masks={0, 15, 0, 0}, .min_col=0, .max_col=3, .min_row=1, .max_row=1, .row_offsets={0,1,2,3} };
    // 1: [[0,2],[1,2],[2,2],[3,2]] -> Cols at 2. 1<<2=4.
    PIECES[PIECE_I].orientations[1] = (PieceOrientation){ .row_masks={4, 4, 4, 4}, .min_col=2, .max_col=2, .min_row=0, .max_row=3, .row_offsets={0,1,2,3} };
    // 2: [[2,0],[2,1],[2,2],[2,3]] -> Row 2: 15.
    PIECES[PIECE_I].orientations[2] = (PieceOrientation){ .row_masks={0, 0, 15, 0}, .min_col=0, .max_col=3, .min_row=2, .max_row=2, .row_offsets={0,1,2,3} };
    // 3: [[0,1],[1,1],[2,1],[3,1]] -> Cols at 1. 1<<1=2.
    PIECES[PIECE_I].orientations[3] = (PieceOrientation){ .row_masks={2, 2, 2, 2}, .min_col=1, .max_col=1, .min_row=0, .max_row=3, .row_offsets={0,1,2,3} };

    // J Piece
    // 0: [[0,0],[1,0],[1,1],[1,2]] -> R0:1(1), R1:7(111)
    PIECES[PIECE_J].orientations[0] = (PieceOrientation){ .row_masks={1, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    // 1: [[0,1],[0,2],[1,1],[2,1]] -> R0:6(110), R1:2, R2:2
    PIECES[PIECE_J].orientations[1] = (PieceOrientation){ .row_masks={6, 2, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    // 2: [[1,0],[1,1],[1,2],[2,2]] -> R1:7, R2:4(100)
    PIECES[PIECE_J].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 4, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    // 3: [[0,1],[1,1],[2,0],[2,1]] -> R0:2, R1:2, R2:3(011)
    PIECES[PIECE_J].orientations[3] = (PieceOrientation){ .row_masks={2, 2, 3, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // L Piece
    // 0: [[0,2],[1,0],[1,1],[1,2]] -> R0:4(100), R1:7
    PIECES[PIECE_L].orientations[0] = (PieceOrientation){ .row_masks={4, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    // 1: [[0,1],[1,1],[2,1],[2,2]] -> R0:2, R1:2, R2:6(110)
    PIECES[PIECE_L].orientations[1] = (PieceOrientation){ .row_masks={2, 2, 6, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    // 2: [[1,0],[1,1],[1,2],[2,0]] -> R1:7, R2:1
    PIECES[PIECE_L].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 1, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    // 3: [[0,0],[0,1],[1,1],[2,1]] -> R0:3, R1:2, R2:2
    PIECES[PIECE_L].orientations[3] = (PieceOrientation){ .row_masks={3, 2, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // O Piece
    // 0: [[0,1],[0,2],[1,1],[1,2]] -> R0:6, R1:6
    // All rotations same
    PieceOrientation o_orient = (PieceOrientation){ .row_masks={6, 6, 0, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    for(int i=0; i<4; i++) PIECES[PIECE_O].orientations[i] = o_orient;

    // S Piece
    // 0: [[0,1],[0,2],[1,0],[1,1]] -> R0:6, R1:3
    PIECES[PIECE_S].orientations[0] = (PieceOrientation){ .row_masks={6, 3, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    // 1: [[0,1],[1,1],[1,2],[2,2]] -> R0:2, R1:6, R2:4
    PIECES[PIECE_S].orientations[1] = (PieceOrientation){ .row_masks={2, 6, 4, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    // 2: [[1,1],[1,2],[2,0],[2,1]] -> R1:6, R2:3
    PIECES[PIECE_S].orientations[2] = (PieceOrientation){ .row_masks={0, 6, 3, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    // 3: [[0,0],[1,0],[1,1],[2,1]] -> R0:1, R1:3, R2:2
    PIECES[PIECE_S].orientations[3] = (PieceOrientation){ .row_masks={1, 3, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // T Piece
    // 0: [[0,1],[1,0],[1,1],[1,2]] -> R0:2, R1:7
    PIECES[PIECE_T].orientations[0] = (PieceOrientation){ .row_masks={2, 7, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    // 1: [[0,1],[1,1],[1,2],[2,1]] -> R0:2, R1:6, R2:2
    PIECES[PIECE_T].orientations[1] = (PieceOrientation){ .row_masks={2, 6, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    // 2: [[1,0],[1,1],[1,2],[2,1]] -> R1:7, R2:2
    PIECES[PIECE_T].orientations[2] = (PieceOrientation){ .row_masks={0, 7, 2, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    // 3: [[0,1],[1,0],[1,1],[2,1]] -> R0:2, R1:3, R2:2
    PIECES[PIECE_T].orientations[3] = (PieceOrientation){ .row_masks={2, 3, 2, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // Z Piece
    // 0: [[0,0],[0,1],[1,1],[1,2]] -> R0:3, R1:6
    PIECES[PIECE_Z].orientations[0] = (PieceOrientation){ .row_masks={3, 6, 0, 0}, .min_col=0, .max_col=2, .min_row=0, .max_row=1, .row_offsets={0,1,2,3} };
    // 1: [[0,2],[1,1],[1,2],[2,1]] -> R0:4, R1:6, R2:2
    PIECES[PIECE_Z].orientations[1] = (PieceOrientation){ .row_masks={4, 6, 2, 0}, .min_col=1, .max_col=2, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };
    // 2: [[1,0],[1,1],[2,1],[2,2]] -> R1:3, R2:6
    PIECES[PIECE_Z].orientations[2] = (PieceOrientation){ .row_masks={0, 3, 6, 0}, .min_col=0, .max_col=2, .min_row=1, .max_row=2, .row_offsets={0,1,2,3} };
    // 3: [[0,1],[1,0],[1,1],[2,0]] -> R0:2, R1:3, R2:1
    PIECES[PIECE_Z].orientations[3] = (PieceOrientation){ .row_masks={2, 3, 1, 0}, .min_col=0, .max_col=1, .min_row=0, .max_row=2, .row_offsets={0,1,2,3} };

    // --- Kicks ---
    // Standard Kicks (J, L, S, T, Z)
    // 0->1 (0->R): (0,0), (-1,0), (-1,+1), (0,-2), (-1,-2)  => NOTE: Offsets in code are (row, col).
    // RotationSystem.py kicks are:
    // (0,1): [[0,0], [0,-1], [-1,-1], [+2,0], [+2,-1]] -> NOTE: Py code uses (row, col) but let's check values.
    // Py: (0, 1): [[+0, -1], [-1, -1], [+2, +0], [+2, -1]] -> These are 4 values? No, standard SRS has 5 tests including (0,0).
    // Wait, RotationSystem.py:
    // (0, 1): [[+0, -1], [-1, -1], [+2, +0], [+2, -1]]
    // It seems to omit (0,0) because that's checked implicitly before kicks?
    // "kick_piece" loops over kick_table.
    // "if (piece.r, new_r) not in kick_table.keys()":
    // "for delta_loc in kick_table[(piece.r, new_r)]:"
    // It seems (0,0) IS NOT in the table in Python. It's handled by basic rotation check first.
    // If basic rotation fails, THEN it checks kicks.
    // Standard SRS usually lists 5 tests, 1st is (0,0).
    // I will follow Python's logic: Basic check first, then list.
    
    // Copying values from RotationSystem.py exactly.
    // (0, 1) -> 0->R
    int8_t k01[4][2] = {{0,-1}, {-1,-1}, {2,0}, {2,-1}};
    for(int i=0; i<4; i++) { KICKS[0][1][i][0] = k01[i][0]; KICKS[0][1][i][1] = k01[i][1]; }
    
    // (0, 3) -> 0->L
    int8_t k03[4][2] = {{0,1}, {-1,1}, {2,0}, {2,1}};
    for(int i=0; i<4; i++) { KICKS[0][3][i][0] = k03[i][0]; KICKS[0][3][i][1] = k03[i][1]; }
    
    // (1, 0) -> R->0
    int8_t k10[4][2] = {{0,1}, {1,1}, {-2,0}, {-2,1}};
    for(int i=0; i<4; i++) { KICKS[1][0][i][0] = k10[i][0]; KICKS[1][0][i][1] = k10[i][1]; }

    // (1, 2) -> R->2
    int8_t k12[4][2] = {{0,1}, {1,1}, {-2,0}, {-2,1}};
    for(int i=0; i<4; i++) { KICKS[1][2][i][0] = k12[i][0]; KICKS[1][2][i][1] = k12[i][1]; }
    
    // (2, 1) -> 2->R
    int8_t k21[4][2] = {{0,-1}, {-1,-1}, {2,0}, {2,-1}};
    for(int i=0; i<4; i++) { KICKS[2][1][i][0] = k21[i][0]; KICKS[2][1][i][1] = k21[i][1]; }

    // (2, 3) -> 2->L
    int8_t k23[4][2] = {{0,1}, {-1,1}, {2,0}, {2,1}};
    for(int i=0; i<4; i++) { KICKS[2][3][i][0] = k23[i][0]; KICKS[2][3][i][1] = k23[i][1]; }

    // (3, 0) -> L->0
    int8_t k30[4][2] = {{0,-1}, {1,-1}, {-2,0}, {-2,-1}};
    for(int i=0; i<4; i++) { KICKS[3][0][i][0] = k30[i][0]; KICKS[3][0][i][1] = k30[i][1]; }

    // (3, 2) -> L->2
    int8_t k32[4][2] = {{0,-1}, {1,-1}, {-2,0}, {-2,-1}};
    for(int i=0; i<4; i++) { KICKS[3][2][i][0] = k32[i][0]; KICKS[3][2][i][1] = k32[i][1]; }

    // 180 Kicks (Standard)
    // (0, 2)
    int8_t k02[5][2] = {{-1,0}, {-1,1}, {-1,-1}, {0,1}, {0,-1}};
    for(int i=0; i<5; i++) { KICKS[0][2][i][0] = k02[i][0]; KICKS[0][2][i][1] = k02[i][1]; }
    
    // (1, 3)
    int8_t k13[5][2] = {{0,1}, {-2,1}, {-1,1}, {-2,0}, {-1,0}};
    for(int i=0; i<5; i++) { KICKS[1][3][i][0] = k13[i][0]; KICKS[1][3][i][1] = k13[i][1]; }

    // (2, 0)
    int8_t k20[5][2] = {{1,0}, {1,-1}, {1,1}, {0,-1}, {0,1}};
    for(int i=0; i<5; i++) { KICKS[2][0][i][0] = k20[i][0]; KICKS[2][0][i][1] = k20[i][1]; }

    // (3, 1)
    int8_t k31[5][2] = {{0,-1}, {-2,-1}, {-1,-1}, {-2,0}, {-1,0}};
    for(int i=0; i<5; i++) { KICKS[3][1][i][0] = k31[i][0]; KICKS[3][1][i][1] = k31[i][1]; }

    // I Kicks
    // (0, 1)
    int8_t ik01[4][2] = {{0,1}, {0,-2}, {1,-2}, {-2,1}};
    for(int i=0; i<4; i++) { I_KICKS[0][1][i][0] = ik01[i][0]; I_KICKS[0][1][i][1] = ik01[i][1]; }
    
    // (0, 3)
    int8_t ik03[4][2] = {{0,-1}, {0,2}, {1,2}, {-2,-1}};
    for(int i=0; i<4; i++) { I_KICKS[0][3][i][0] = ik03[i][0]; I_KICKS[0][3][i][1] = ik03[i][1]; }
    
    // (1, 0)
    int8_t ik10[4][2] = {{0,-1}, {0,2}, {2,-1}, {-1,2}};
    for(int i=0; i<4; i++) { I_KICKS[1][0][i][0] = ik10[i][0]; I_KICKS[1][0][i][1] = ik10[i][1]; }
    
    // (1, 2)
    int8_t ik12[4][2] = {{0,-1}, {0,2}, {-2,-1}, {1,2}};
    for(int i=0; i<4; i++) { I_KICKS[1][2][i][0] = ik12[i][0]; I_KICKS[1][2][i][1] = ik12[i][1]; }
    
    // (2, 1)
    int8_t ik21[4][2] = {{0,-2}, {0,1}, {-1,-2}, {2,1}};
    for(int i=0; i<4; i++) { I_KICKS[2][1][i][0] = ik21[i][0]; I_KICKS[2][1][i][1] = ik21[i][1]; }
    
    // (2, 3)
    int8_t ik23[4][2] = {{0,2}, {0,-1}, {-1,2}, {2,-1}};
    for(int i=0; i<4; i++) { I_KICKS[2][3][i][0] = ik23[i][0]; I_KICKS[2][3][i][1] = ik23[i][1]; }
    
    // (3, 0)
    int8_t ik30[4][2] = {{0,1}, {0,-2}, {2,1}, {-1,-2}};
    for(int i=0; i<4; i++) { I_KICKS[3][0][i][0] = ik30[i][0]; I_KICKS[3][0][i][1] = ik30[i][1]; }
    
    // (3, 2)
    int8_t ik32[4][2] = {{0,1}, {0,-2}, {-2,1}, {1,-2}};
    for(int i=0; i<4; i++) { I_KICKS[3][2][i][0] = ik32[i][0]; I_KICKS[3][2][i][1] = ik32[i][1]; }

    // 180 I Kicks
    // (0, 2)
    int8_t ik02[5][2] = {{-1,0}, {-1,1}, {-1,-1}, {0,1}, {0,-1}};
    for(int i=0; i<5; i++) { I_KICKS[0][2][i][0] = ik02[i][0]; I_KICKS[0][2][i][1] = ik02[i][1]; }

    // (1, 3)
    int8_t ik13[5][2] = {{0,1}, {-2,1}, {-1,1}, {-2,0}, {-1,0}};
    for(int i=0; i<5; i++) { I_KICKS[1][3][i][0] = ik13[i][0]; I_KICKS[1][3][i][1] = ik13[i][1]; }

    // (2, 0)
    int8_t ik20[5][2] = {{1,0}, {1,-1}, {1,1}, {0,-1}, {0,1}};
    for(int i=0; i<5; i++) { I_KICKS[2][0][i][0] = ik20[i][0]; I_KICKS[2][0][i][1] = ik20[i][1]; }

    // (3, 1)
    int8_t ik31[5][2] = {{0,-1}, {-2,-1}, {-1,-1}, {-2,0}, {-1,0}};
    for(int i=0; i<5; i++) { I_KICKS[3][1][i][0] = ik31[i][0]; I_KICKS[3][1][i][1] = ik31[i][1]; }

    initialized = true;
}

// -- Helpers --

int check_collision(const uint16_t* board_rows, int board_height, int piece_type, int rot, int r, int c) {
    PieceOrientation* ori = &PIECES[piece_type].orientations[rot];
    
    // Bounds: disallow negative rows (matches Python overlaps/_can_occupy) and right/left walls.
    if (c + ori->min_col < 0 || c + ori->max_col >= BOARD_COLS) return 1;
    if (r + ori->min_row < 0) return 1;
    if (r + ori->max_row >= board_height) return 1;

    for (int i = 0; i < 4; i++) {
        int board_row = r + i;
        if (board_row < 0 || board_row >= board_height) continue;
        
        uint16_t mask = ori->row_masks[i];
        uint16_t shifted;
        if (c >= 0) {
            shifted = mask << c;
        } else {
            shifted = mask >> (-c);
        }
        
        if (board_rows[board_row] & shifted) return 1;
    }
    return 0;
}

int hard_drop_row(const uint16_t* board_rows, int board_height, int piece_type, int rot, int r, int c) {
    int curr = r;
    while (!check_collision(board_rows, board_height, piece_type, rot, curr + 1, c)) {
        curr++;
    }
    return curr;
}

// Encode state: (row * 10 + norm_col) * 4 + rot
// norm_col = col + min_col
// This ensures we index by [0..cols)
int encode_state(int r, int c, int rot, int piece_type) {
    int min_col = PIECES[piece_type].orientations[rot].min_col;
    int norm_col = c + min_col;
    if (norm_col < 0 || norm_col >= BOARD_COLS) return -1;
    if (r < 0 || r >= BOARD_ROWS) return -1; // Should check bound
    
    return ((r * BOARD_COLS) + norm_col) * 4 + rot;
}

// Decode not strictly needed if we store (r,c,rot) in Queue, but we store ID to save space?
// Actually, Queue storing int16 is fine.
void decode_state(int state, int* r, int* c, int* rot, int piece_type) {
    *rot = state % 4;
    int base = state / 4;
    int norm_col = base % BOARD_COLS;
    *r = base / BOARD_COLS;
    
    int min_col = PIECES[piece_type].orientations[*rot].min_col;
    *c = norm_col - min_col;
}

bool is_t_spin(const uint16_t* board, int board_height, int r, int c, int rot, int last_move_key, int delta_loc_sum) {
    // 3-corner rule
    // T Piece corners: (0,0), (0,2), (2,0), (2,2)
    // Board bounds check included
    
    // Front corners depend on rotation.
    // 0 (Up): Front=(0,0),(0,2), Back=(2,0),(2,2) ? No.
    // "Front" means the side the T is pointing to.
    // T points: 0=Up, 1=Right, 2=Down, 3=Left.
    // 0: Pointing Up (row 0, col 1). Front corners are (0,0) and (0,2). Back are (2,0), (2,2).
    // 1: Pointing Right (row 1, col 2). Front: (0,2), (2,2).
    // 2: Pointing Down (row 2, col 1). Front: (2,0), (2,2).
    // 3: Pointing Left (row 1, col 0). Front: (0,0), (2,0).
    
    int corners_filled = 0;
    int front_corners_filled = 0;
    
    int corners[4][2] = {{0,0}, {0,2}, {2,2}, {2,0}}; // TL, TR, BR, BL
    
    // Define front corner indices for each rotation (indices into corners array)
    // Rot 0: TL, TR (0, 1)
    // Rot 1: TR, BR (1, 2)
    // Rot 2: BR, BL (2, 3)
    // Rot 3: BL, TL (3, 0)
    
    for (int i=0; i<4; i++) {
        int cr = r + corners[i][0];
        int cc = c + corners[i][1];
        
        bool filled = false;
        if (cr >= board_height || cc < 0 || cc >= BOARD_COLS) {
            filled = true; // Wall/Floor counts as filled
        } else if (cr < 0) {
            filled = true; // Ceiling counts as filled (matches Scorer.py logic)
        } else {
            if (board[cr] & (1 << cc)) filled = true; // wait, board[cr] bit order?
            // Collision logic: 1<<c. So bit 0 is col 0.
            // Correct.
        }
        
        if (filled) {
            corners_filled++;
            // Check if it's a front corner
            if ((rot == 0 && (i==0 || i==1)) ||
                (rot == 1 && (i==1 || i==2)) ||
                (rot == 2 && (i==2 || i==3)) ||
                (rot == 3 && (i==3 || i==0))) {
                front_corners_filled++;
            }
        }
    }
    
    // T-Spin or Mini: 3 (or 4) corners filled counts as spin (matches Scorer spin != NO_SPIN).
    if (corners_filled < 3) return false;
    return true;
}

bool check_immobility(const uint16_t* board, int board_height, int piece_type, int rot, int r, int c) {
     // Check Left, Right, Up, Down (approx for mobility)
     // Python: `for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]: if not overlaps...`
     // If ANY move is possible, NO_SPIN (unless T).
     
     int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
     for(int i=0; i<4; i++) {
         if (!check_collision(board, board_height, piece_type, rot, r + dirs[i][0], c + dirs[i][1])) {
             return false;
         }
     }
     return true;
}

// -- Candidate collection for row tiers --

typedef struct {
    int count;
    int landing_rows[MAX_LANDINGS];
    int bfs_states[MAX_LANDINGS];
} SlotCandidates;

static void write_sequence(
    const StateMeta* meta, int bfs_state, int is_hold,
    int max_len, int64_t* out_row
) {
    int len = 0;
    int path[STATE_SPACE];
    int curr = bfs_state;

    while(meta[curr].parent != -1) {
        path[len++] = meta[curr].last_move;
        curr = meta[curr].parent;
    }

    int p = 0;
    out_row[p++] = KEY_START;
    if (is_hold) out_row[p++] = KEY_HOLD;

    for(int i=len-1; i>=0; i--) {
        if (p < max_len) out_row[p++] = path[i];
    }

    if (p < max_len) out_row[p++] = KEY_HARD_DROP;
    while(p < max_len) out_row[p++] = KEY_PAD;
}

static void sort_candidates(SlotCandidates* slot) {
    int n = slot->count;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (slot->landing_rows[j] < slot->landing_rows[i]) {
                int tmp_r = slot->landing_rows[i];
                slot->landing_rows[i] = slot->landing_rows[j];
                slot->landing_rows[j] = tmp_r;
                int tmp_s = slot->bfs_states[i];
                slot->bfs_states[i] = slot->bfs_states[j];
                slot->bfs_states[j] = tmp_s;
            }
        }
    }
}

// -- Main BFS --

void find_sequences_c(
    const uint16_t* board_rows,
    const int board_height,
    const int piece_type,
    const int start_row,
    const int start_col,
    const int start_rot,
    const int max_len,
    const int is_hold,
    const int num_row_tiers,
    int64_t* output_buffer // Array of size [80 * num_row_tiers * max_len]
) {
    if (!initialized) init_pieces();

    static StateMeta meta[STATE_SPACE];
    static bool visited[STATE_SPACE];
    static int queue[QUEUE_CAPACITY];
    static SlotCandidates candidates[BASE_POSITIONS];

    for(int i=0; i<STATE_SPACE; i++) {
        visited[i] = false;
        meta[i].parent = -1;
    }
    for(int i=0; i<BASE_POSITIONS; i++) {
        candidates[i].count = 0;
    }
    
    int start_state = encode_state(start_row, start_col, start_rot, piece_type);
    if (start_state == -1 || check_collision(board_rows, board_height, piece_type, start_rot, start_row, start_col)) {
        return;
    }
    
    int head = 0;
    int tail = 0;
    
    queue[tail++] = start_state;
    visited[start_state] = true;
    meta[start_state].depth = 0;
    meta[start_state].last_move = KEY_START;
    meta[start_state].delta_r = 0;
    
    int max_seq = is_hold ? max_len : max_len - 1;
    int max_depth = max_seq - 2;
    
    int visible_start = board_height - VISIBLE_ROWS;
    
    while(head != tail) {
        int curr_state = queue[head++];
        head %= QUEUE_CAPACITY;
        
        int r, c, rot;
        decode_state(curr_state, &r, &c, &rot, piece_type);
        int depth = meta[curr_state].depth;
        
        int land_r = hard_drop_row(board_rows, board_height, piece_type, rot, r, c);

        if (r == land_r && land_r >= visible_start && land_r < visible_start + VISIBLE_ROWS) {
            bool is_spin = false;

            if (meta[curr_state].delta_r != 0) {
                 if (piece_type == PIECE_T) {
                     int delta_sum = abs(meta[curr_state].delta_row) + abs(meta[curr_state].delta_col);
                     is_spin = is_t_spin(board_rows, board_height, land_r, c, rot, 
                                         meta[curr_state].last_move, delta_sum);
                 } else {
                     if (check_immobility(board_rows, board_height, piece_type, rot, land_r, c)) {
                         is_spin = true;
                     }
                 }
            }
            
            int norm_col = c + PIECES[piece_type].orientations[rot].min_col;
            int base_idx = rot * BOARD_COLS * SPIN_STATES + norm_col * SPIN_STATES + (is_spin ? 1 : 0);
            
            SlotCandidates* slot = &candidates[base_idx];
            bool dup = false;
            for (int i = 0; i < slot->count; i++) {
                if (slot->landing_rows[i] == land_r) { dup = true; break; }
            }
            if (!dup && slot->count < MAX_LANDINGS) {
                slot->landing_rows[slot->count] = land_r;
                slot->bfs_states[slot->count] = curr_state;
                slot->count++;
            }
        }
        
        if (depth >= max_depth) continue;
        
        int moves[] = {KEY_TAP_LEFT, KEY_TAP_RIGHT, KEY_DAS_LEFT, KEY_DAS_RIGHT, 
                       KEY_CLOCKWISE, KEY_ANTICLOCKWISE, KEY_ROTATE_180, KEY_SOFT_DROP};
                       
        for(int m=0; m<8; m++) {
            int key = moves[m];
            int nr = r, nc = c, nrot = rot;
            int dr=0, drow=0, dcol=0;
            
            bool valid = false;
            
            if (key == KEY_TAP_LEFT) {
                if (!check_collision(board_rows, board_height, piece_type, rot, r, c-1)) {
                    nc--; valid = true; dcol=-1;
                }
            } else if (key == KEY_TAP_RIGHT) {
                if (!check_collision(board_rows, board_height, piece_type, rot, r, c+1)) {
                    nc++; valid = true; dcol=1;
                }
            } else if (key == KEY_DAS_LEFT) {
                int tmp = c;
                while(!check_collision(board_rows, board_height, piece_type, rot, r, tmp-1)) {
                    tmp--;
                }
                if(tmp != c) { nc=tmp; valid=true; dcol=nc-c; }
            } else if (key == KEY_DAS_RIGHT) {
                int tmp = c;
                while(!check_collision(board_rows, board_height, piece_type, rot, r, tmp+1)) {
                    tmp++;
                }
                if(tmp != c) { nc=tmp; valid=true; dcol=nc-c; }
            } else if (key == KEY_SOFT_DROP) {
                 int tmp = r;
                 int max_row = PIECES[piece_type].orientations[rot].max_row;
                 while(!check_collision(board_rows, board_height, piece_type, rot, tmp+1, c)) {
                     tmp++;
                     if (tmp + max_row >= board_height - 1) break;
                 }
                 if(tmp != r) { nr=tmp; valid=true; drow=nr-r; }
            } else {
                int delta = 0;
                if (key == KEY_CLOCKWISE) delta = 1;
                else if (key == KEY_ANTICLOCKWISE) delta = 3;
                else delta = 2;
                
                int next_rot = (rot + delta) % 4;
                
                if (!check_collision(board_rows, board_height, piece_type, next_rot, r, c)) {
                    nrot = next_rot; valid = true; dr = (delta==3)?-1:delta;
                } else {
                    int8_t (*table)[2];
                    if (piece_type == PIECE_I) table = I_KICKS[rot][next_rot];
                    else table = KICKS[rot][next_rot];
                    
                    int count = 4;
                    if (key == KEY_ROTATE_180) count = 5;
                    
                    for(int k=0; k<count; k++) {
                        int kdr = table[k][0];
                        int kdc = table[k][1];
                        if (kdr == 0 && kdc == 0 && count == 5) continue;
                        
                        if (!check_collision(board_rows, board_height, piece_type, next_rot, r+kdr, c+kdc)) {
                            nr = r+kdr; nc = c+kdc; nrot = next_rot;
                            valid = true;
                            dr = (delta==3)?-1:delta;
                            drow = kdr; dcol = kdc;
                            break;
                        }
                    }
                }
            }
            
            if (valid) {
                int next_s = encode_state(nr, nc, nrot, piece_type);
                if (next_s != -1 && !visited[next_s]) {
                    visited[next_s] = true;
                    meta[next_s].parent = curr_state;
                    meta[next_s].last_move = key;
                    meta[next_s].depth = depth + 1;
                    meta[next_s].delta_r = dr;
                    meta[next_s].delta_row = drow;
                    meta[next_s].delta_col = dcol;
                    
                    queue[tail++] = next_s;
                    tail %= QUEUE_CAPACITY;
                }
            }
        }
    }

    // Post-process: select tier representatives and write sequences
    int N = num_row_tiers;
    for (int base = 0; base < BASE_POSITIONS; base++) {
        SlotCandidates* slot = &candidates[base];
        int K = slot->count;
        if (K == 0) continue;

        sort_candidates(slot);

        int selected_indices[MAX_LANDINGS];
        int num_selected;

        if (K <= N) {
            num_selected = K;
            for (int i = 0; i < K; i++) selected_indices[i] = i;
        } else {
            num_selected = N;
            for (int t = 0; t < N; t++) {
                if (N == 1) {
                    selected_indices[t] = 0;
                } else {
                    selected_indices[t] = (int)(0.5 + (double)t * (double)(K - 1) / (double)(N - 1));
                }
            }
        }

        for (int t = 0; t < num_selected; t++) {
            int ci = selected_indices[t];
            int out_slot = base * N + t;
            write_sequence(meta, slot->bfs_states[ci], is_hold,
                           max_len, &output_buffer[out_slot * max_len]);
        }
    }
}
