"""Beam oracle backed by the GPU teacher, shaped like CB2BSearch for the demo."""

import numpy as np
from TetrisEnv.Pieces import PieceType

from qtris.data.placement_features import PAD

BRANCH_STRIDE = 160


class GpuOracle:
    """CB2BSearch.search_with_scores over teacher.GpuTeacher, with key sequences.

    The teacher returns placement descriptors, so each root is matched back to a key
    sequence through the env's own pathfinder, one bulk call per hold branch.
    """

    def __init__(self, env, search_depth, beam_width, queue_size, max_len):
        from teacher.api import GpuTeacher

        self._env = env
        self._max_len = int(max_len)
        self._teacher = GpuTeacher(
            max_batch=1,
            width=int(beam_width),
            depth=int(search_depth),
            queue_len=int(queue_size),
        )

    def _branch_sequences(self, board, is_hold):
        """Key sequence of every placement of one branch, keyed by descriptor."""
        env = self._env
        if is_hold:
            held = env._hold_piece
            piece = env._spawn_piece(held if held != PieceType.N else env._queue[0])
        else:
            piece = env._active_piece
        rots, ncols, rows, spins, seqs = (
            env._key_sequence_finder.find_unique_placements(
                board=board,
                piece=piece,
                max_len=self._max_len,
                is_hold=bool(is_hold),
                with_sequences=True,
            )
        )
        return {
            (int(r), int(c), int(w), int(s)): seqs[i]
            for i, (r, c, w, s) in enumerate(zip(rots, ncols, rows, spins, strict=True))
        }

    def _pad(self):
        return np.full(self._max_len, PAD, dtype=np.int64)

    def search_with_scores(
        self,
        board,
        active_piece,
        hold_piece,
        queue,
        b2b,
        combo,
        total_garbage,
        garbage_push_delay=1,
        bag_seen=0,
        search_depth=10,
        beam_width=128,
        max_len=15,
        max_roots=512,
    ):
        """Best action and every root candidate, matching CB2BSearch's return tuple."""
        queue = np.asarray(queue, dtype=np.int32)
        action, placement, root_actions, root_scores, root_pl, root_rows, best_score = (
            self._teacher.search_with_scores(
                board,
                active_piece,
                hold_piece,
                queue,
                b2b,
                combo,
                total_garbage,
                garbage_push_delay=garbage_push_delay,
                bag_seen=bag_seen,
                search_depth=search_depth,
                beam_width=beam_width,
                max_roots=max_roots,
            )
        )

        lookup = {}
        for is_hold in (0, 1):
            lookup[is_hold] = self._branch_sequences(board, is_hold)

        n = len(root_actions)
        sequences = np.full((n, self._max_len), PAD, dtype=np.int64)
        for i in range(n):
            a = int(root_actions[i])
            key = (
                (a % BRANCH_STRIDE) // 40,
                (a % 40) // 4,
                int(root_rows[i]),
                a % 4,
            )
            seq = lookup[a // BRANCH_STRIDE].get(key)
            if seq is not None:
                sequences[i] = seq

        best_sequence = self._pad()
        if int(action) >= 0:
            # Keyed on the full descriptor: the action index alone is shared by every
            # landing row of one shape and column.
            key = (
                (int(action) % BRANCH_STRIDE) // 40,
                (int(action) % 40) // 4,
                int(placement[3]),
                int(action) % 4,
            )
            seq = lookup[int(action) // BRANCH_STRIDE].get(key)
            if seq is not None:
                best_sequence = seq

        return (
            int(action),
            best_sequence,
            np.asarray(root_actions, np.int32),
            np.asarray(root_scores, np.float32),
            sequences,
            np.asarray(root_rows, np.int32),
            float(best_score),
        )
