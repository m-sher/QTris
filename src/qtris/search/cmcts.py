"""Thin ctypes wrapper over the fully-C MCTS engine in `b2b_search.c`.

The engine holds one PUCT tree per game in C and runs the whole sim loop (descend / step /
enumerate / backup) GIL-free, OpenMP-threaded across games. Only the TF policy/value net stays
in Python: per round the driver calls `collect_leaves` (C emits up to `leaves_per_round` leaves
per live tree, diverged by virtual loss), runs the net once on the batch, then `apply_leaves`
(C sets priors + bootstrap, reverts the virtual loss, and backs up). Intra-tree batching cuts the
sequential net calls per move from `num_simulations` to `ceil(num_simulations / leaves_per_round)`.
Dirichlet root noise and final action sampling are generated in Python and passed in.
"""

import ctypes
import glob
import os

import numpy as np

import TetrisEnv

CANDIDATE_CAPACITY = 128
FEATURE_DIM = 18
_COL_BITS = (np.uint16(1) << np.arange(10, dtype=np.uint16)).astype(np.uint16)

_F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_I32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_I64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
_U16 = np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS")
_U8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")


def _load_lib():
    pkg = os.path.dirname(os.path.abspath(TetrisEnv.__file__))
    cands = glob.glob(os.path.join(pkg, "b2b_search*.so"))
    if not cands:
        raise RuntimeError(f"b2b_search .so not found in {pkg}")
    lib = ctypes.CDLL(cands[0])
    lib.mcts_create.argtypes = (
        [ctypes.c_int] * 8
        + [ctypes.c_float] * 6
        + [ctypes.c_int] * 2
        + [ctypes.c_int, ctypes.c_float]  # leaves_per_round, vloss
    )
    lib.mcts_create.restype = ctypes.c_void_p
    lib.mcts_set_root.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        _U16,
        ctypes.c_int,
        ctypes.c_int,
        _I32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        _I32,
        ctypes.c_int,
        _I32,
        _I32,
        _I32,
        ctypes.c_int,
    ]
    lib.mcts_set_root.restype = None
    for name in ("mcts_collect_roots", "mcts_collect_leaves"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, _F32, _I64, _F32, _F32, _U8, _I32]
        fn.restype = ctypes.c_int
    lib.mcts_apply_roots.argtypes = [ctypes.c_void_p, _F32, _F32, _F32, ctypes.c_float]
    lib.mcts_apply_roots.restype = None
    lib.mcts_apply_leaves.argtypes = [ctypes.c_void_p, _F32, _F32]
    lib.mcts_apply_leaves.restype = None
    lib.mcts_result.argtypes = [ctypes.c_void_p, _F32, _F32, _I32, _I32]
    lib.mcts_result.restype = None
    lib.mcts_destroy.argtypes = [ctypes.c_void_p]
    lib.mcts_destroy.restype = None
    return lib


_LIB = None


class CMCTS:
    def __init__(
        self,
        num_trees,
        *,
        board_height=24,
        queue_size=5,
        max_height=18,
        max_holes=50,
        garbage_push_delay=0,
        auto_push_garbage=1,
        auto_fill_queue=1,
        c_puct=1.5,
        gamma=0.99,
        w_attack=1.0,
        w_b2b=1.0,
        w_death=0.0,
        return_scale=1.0,
        max_len=15,
        num_simulations=64,
        leaves_per_round=4,
        vloss=1.0,
    ):
        global _LIB
        if _LIB is None:
            _LIB = _load_lib()
        self.lib = _LIB
        self.n = num_trees
        self.bh = board_height
        self.qsize = queue_size
        self.pw = 2 + queue_size
        self.cap = CANDIDATE_CAPACITY
        self.lpr = max(1, int(leaves_per_round))
        self.h = self.lib.mcts_create(
            num_trees,
            board_height,
            queue_size,
            max_height,
            max_holes if max_holes is not None else -1,
            garbage_push_delay,
            auto_push_garbage,
            auto_fill_queue,
            c_puct,
            gamma,
            w_attack,
            w_b2b,
            w_death,
            return_scale,
            max_len,
            num_simulations
            + self.lpr
            + 1,  # arena: root + ~one node per simulation (+L headroom)
            self.lpr,
            vloss,
        )
        # request buffers: a round emits up to num_trees * lpr leaves; sliced to nv per round
        rows = num_trees * self.lpr
        self._boards = np.zeros(rows * board_height * 10, np.float32)
        self._pieces = np.zeros(rows * self.pw, np.int64)
        self._bcg = np.zeros(rows * 3, np.float32)
        self._pls = np.zeros(rows * self.cap * FEATURE_DIM, np.float32)
        self._masks = np.zeros(rows * self.cap, np.uint8)
        self._tree_ids = np.zeros(rows, np.int32)
        # result buffers
        self._pi = np.zeros(num_trees * self.cap, np.float32)
        self._counts = np.zeros(num_trees * self.cap, np.float32)
        self._desc = np.zeros(num_trees * self.cap * 5, np.int32)
        self._dead = np.zeros(num_trees, np.int32)

    def set_root(self, tree, env):
        occ = (env._board != 0).astype(np.uint16)
        board = np.ascontiguousarray((occ * _COL_BITS).sum(axis=1, dtype=np.uint16))
        queue = np.array([p.value for p in env._queue], np.int32)
        pending = np.array([p.value for p in env._next_bag], np.int32)
        gq = env._garbage_queue
        gr = np.array([g[0] for g in gq], np.int32) if gq else np.zeros(0, np.int32)
        gc = np.array([g[1] for g in gq], np.int32) if gq else np.zeros(0, np.int32)
        gt = np.array([g[2] for g in gq], np.int32) if gq else np.zeros(0, np.int32)
        # ndpointer rejects size-0 arrays; pad to length 1 (gcnt=0 means C ignores them)
        ensure = lambda a: a if a.size else np.zeros(1, np.int32)  # noqa: E731
        self.lib.mcts_set_root(
            self.h,
            tree,
            board,
            env._active_piece.piece_type.value,
            env._hold_piece.value,
            ensure(queue),
            len(queue),
            env._scorer._b2b,
            env._scorer._combo,
            int(env._tetrio_rng._t),
            ensure(pending),
            len(pending),
            ensure(gr),
            ensure(gc),
            ensure(gt),
            len(gq),
        )

    def _collect(self, fn):
        nv = fn(
            self.h,
            self._boards,
            self._pieces,
            self._bcg,
            self._pls,
            self._masks,
            self._tree_ids,
        )
        if nv == 0:
            return 0, None
        boards = self._boards[: nv * self.bh * 10].reshape(nv, self.bh, 10, 1)
        pieces = self._pieces[: nv * self.pw].reshape(nv, self.pw)
        bcg = self._bcg[: nv * 3].reshape(nv, 3)
        pls = self._pls[: nv * self.cap * FEATURE_DIM].reshape(
            nv, self.cap, FEATURE_DIM
        )
        masks = self._masks[: nv * self.cap].reshape(nv, self.cap).astype(bool)
        tree_ids = self._tree_ids[:nv].copy()
        return nv, (boards, pieces, bcg, pls, masks, tree_ids)

    def collect_roots(self):
        return self._collect(self.lib.mcts_collect_roots)

    def collect_leaves(self):
        return self._collect(self.lib.mcts_collect_leaves)

    def apply_roots(self, logits, values, dir_noise, dir_eps):
        self.lib.mcts_apply_roots(
            self.h,
            np.ascontiguousarray(logits, np.float32).ravel(),
            np.ascontiguousarray(values, np.float32).ravel(),
            np.ascontiguousarray(dir_noise, np.float32).ravel(),
            float(dir_eps),
        )

    def apply_leaves(self, logits, values):
        self.lib.mcts_apply_leaves(
            self.h,
            np.ascontiguousarray(logits, np.float32).ravel(),
            np.ascontiguousarray(values, np.float32).ravel(),
        )

    def result(self):
        self.lib.mcts_result(self.h, self._pi, self._counts, self._desc, self._dead)
        pi = self._pi.reshape(self.n, self.cap).copy()
        counts = self._counts.reshape(self.n, self.cap).copy()
        desc = self._desc.reshape(self.n, self.cap, 5).copy()
        dead = self._dead.astype(bool).copy()
        return pi, counts, desc, dead

    def destroy(self):
        if self.h:
            self.lib.mcts_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
