#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define BOARD_COLS 10
#define QUEUE_CAPACITY 4096 

typedef struct {
    int r;
    int c;
} Point;

// Finds the number of enclosed holes on a board.
// An enclosed hole is a cell that does not have a valid path to the top of the board.
// board_rows: array of uint16_t where each element is a bitmask of the row.
// board_height: number of rows in the board.
int count_enclosed_holes(const uint16_t* board_rows, int board_height) {
    if (board_height <= 0) return 0;

    // Init visited
    bool* visited = (bool*)calloc(board_height * BOARD_COLS, sizeof(bool));
    if (!visited) return -1;

    // Queue for BFS
    Point queue[QUEUE_CAPACITY];
    int head = 0;
    int tail = 0;

    // Add valid starting points (empty cells in top row) to queue
    for (int c = 0; c < BOARD_COLS; c++) {
        if ((board_rows[0] & (1 << c)) == 0) {
            int idx = 0 * BOARD_COLS + c;
            visited[idx] = true;
            queue[tail++] = (Point){0, c};
        }
    }

    // BFS
    // Directions: Down, Left, Right, Up
    int dr[] = {1, 0, 0, -1};
    int dc[] = {0, -1, 1, 0};

    while (head != tail) {
        Point curr = queue[head++];
        head %= QUEUE_CAPACITY;

        for (int i = 0; i < 4; i++) {
            int nr = curr.r + dr[i];
            int nc = curr.c + dc[i];

            // Bounds check
            if (nr < 0 || nr >= board_height || nc < 0 || nc >= BOARD_COLS) {
                continue;
            }

            // Check if visited
            int nidx = nr * BOARD_COLS + nc;
            if (visited[nidx]) {
                continue;
            }

            // Check if empty
            if ((board_rows[nr] & (1 << nc)) == 0) {
                // Found reachable empty cell
                visited[nidx] = true;
                queue[tail++] = (Point){nr, nc};
                tail %= QUEUE_CAPACITY;
            }
        }
    }

    // Count enclosed holes
    // Enclosed hole = Empty AND Not Visited
    int enclosed_holes = 0;
    for (int r = 0; r < board_height; r++) {
        for (int c = 0; c < BOARD_COLS; c++) {
            bool is_empty = (board_rows[r] & (1 << c)) == 0;
            if (is_empty) {
                int idx = r * BOARD_COLS + c;
                if (!visited[idx]) {
                    enclosed_holes++;
                }
            }
        }
    }

    free(visited);
    return enclosed_holes;
}
