"""TF-tensor bridge over a Gymnasium vector env.

Marshals batched numpy <-> tf and rebuilds the per-channel reward dict from the batched
info (gym batches each reward channel to shape (num_envs,) plus a "_<channel>" mask).
"""

import numpy as np
import tensorflow as tf


class VecStep:
    """Batched result: tf observation dict, reward-channel dict, done bool tensor (num_envs,)."""

    __slots__ = ("observation", "reward", "done")

    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done


def _to_tf(d):
    return {k: tf.convert_to_tensor(v) for k, v in d.items()}


class TFVecEnv:
    def __init__(self, venv):
        self._venv = venv
        self.num_envs = venv.num_envs

    def reset(self):
        obs, _info = self._venv.reset()
        return VecStep(_to_tf(obs), {}, None)

    def step(self, actions):
        a = actions.numpy() if hasattr(actions, "numpy") else np.asarray(actions)
        obs, _reward, terminated, truncated, info = self._venv.step(a)
        reward = _to_tf({k: v for k, v in info.items() if not k.startswith("_")})
        done = tf.convert_to_tensor(np.logical_or(terminated, truncated))
        return VecStep(_to_tf(obs), reward, done)

    def close(self):
        self._venv.close()


def make_tf_vec_env(constructors):
    """Wrap env constructors in a NEXT_STEP-autoreset AsyncVectorEnv + TFVecEnv."""
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode

    venv = AsyncVectorEnv(list(constructors), autoreset_mode=AutoresetMode.NEXT_STEP)
    return TFVecEnv(venv)
