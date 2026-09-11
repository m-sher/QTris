"""Productive-attack targets, censored death hazards, and checkpoint profiles."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from qtris.search.cmcts import RISK_HORIZON
from qtris.search.placement_mcts import MCTSConfig

OBJECTIVE = "productive_attack_risk_v1"
PROFILE_NAME = "objective.json"


def attack_risk_config(**kwargs):
    return MCTSConfig(
        **{
            "gamma": 0.97,
            "w_attack": 0.006,
            "w_death": 0.0,
            "w_b2b": 0.0,
            "w_height": 0.0,
            "w_bumpiness": 0.0,
            "w_holes": 0.0,
            "w_plain": 0.0,
            "w_residual": 0.0,
            "risk_gate": True,
            **kwargs,
        }
    )


def n_step_attack(rewards, values, n, gamma, tail_value):
    """Discount observed rewards and bootstrap the state after the last included action."""
    if n < 1 or len(rewards) != len(values) or not 0 <= gamma <= 1:
        raise ValueError("invalid attack target horizon, values, or discount")
    length = len(rewards)
    targets = np.zeros(length, np.float32)
    for t in range(length):
        end = min(length, t + n)
        g = tail_value if end == length else values[end]
        for j in range(end - 1, t - 1, -1):
            g = rewards[j] + gamma * g
        targets[t] = g
    return targets


def death_targets(length, own_death, horizon=RISK_HORIZON):
    """Return chosen-action hazard labels/masks and observed cumulative-risk labels/masks."""
    hazards = np.zeros((length, horizon), np.float32)
    hazard_mask = np.zeros_like(hazards)
    cumulative = np.zeros_like(hazards)
    observed = np.zeros_like(hazards)
    for t in range(length):
        remaining = length - t
        hazard_mask[t, : min(remaining, horizon)] = 1.0
        observed[t, : min(remaining, horizon)] = 1.0
        if own_death and remaining <= horizon:
            hazards[t, remaining - 1] = 1.0
            cumulative[t, remaining - 1 :] = 1.0
            observed[t, :] = 1.0
    return hazards, hazard_mask, cumulative, observed


def state_observation(env):
    """Copy the post-exchange learner-visible state before any reset."""
    return (
        (env._board[-24:] != 0).astype(np.float32)[..., None],
        np.array(
            [
                env._active_piece.piece_type.value,
                env._hold_piece.value,
                *[p.value for p in env._queue],
            ],
            np.int64,
        ),
        np.array(
            [env._scorer._b2b, env._scorer._combo, env._get_total_garbage()], np.float32
        ),
    )


def learner_values(net, observations, batch_size):
    """Evaluate all target bootstrap states with one learner and fixed-size batches."""
    out = []
    for start in range(0, len(observations), batch_size):
        chunk = observations[start : start + batch_size]
        arrays = [np.stack([obs[k] for obs in chunk]) for k in range(3)]
        if len(chunk) < batch_size:
            arrays = [
                np.concatenate(
                    [a, np.zeros((batch_size - len(chunk), *a.shape[1:]), a.dtype)]
                )
                for a in arrays
            ]
        values = net.state_value(*[tf.constant(a) for a in arrays]).numpy()[:, 0]
        out.extend(values[: len(chunk)])
    return np.asarray(out, np.float32)


@tf.function
def train_step(net, batch, value_coef, risk_coef):
    with tf.GradientTape() as tape:
        logits, values, hazards = net.attack_risk(
            (
                batch["boards"],
                batch["pieces"],
                batch["bcg"],
                batch["cand_placements"],
                batch["cand_mask"],
            ),
            training=True,
        )
        masked = tf.where(batch["cand_mask"], logits, tf.constant(-1e9, tf.float32))
        log_probs = tf.nn.log_softmax(masked, axis=-1)
        tgt = batch["pi_target"]
        pm = batch["policy_mask"]
        pnorm = tf.reduce_sum(pm)
        ce = -tf.reduce_sum(tgt * log_probs, axis=-1)
        policy_loss = tf.math.divide_no_nan(tf.reduce_sum(pm * ce), pnorm)
        value_loss = tf.reduce_mean((values[:, 0] - batch["value_target"]) ** 2)
        chosen = tf.gather(hazards, batch["slot"], batch_dims=1)
        mask = batch["hazard_mask"]
        risk_ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=batch["hazard_target"], logits=chosen
        )
        risk_loss = tf.math.divide_no_nan(
            tf.reduce_sum(risk_ce * mask), tf.reduce_sum(mask)
        )
        loss = policy_loss + value_coef * value_loss + risk_coef * risk_loss
    grads = tape.gradient(loss, net.trainable_variables)
    grad_norm = tf.linalg.global_norm(grads)
    net.optimizer.apply_gradients(zip(grads, net.trainable_variables))
    entropy = -tf.reduce_sum(tf.exp(log_probs) * log_probs, axis=-1)
    target_var = tf.math.reduce_variance(batch["value_target"])
    residual_var = tf.math.reduce_variance(batch["value_target"] - values[:, 0])
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "risk_loss": risk_loss,
        "entropy": tf.math.divide_no_nan(tf.reduce_sum(pm * entropy), pnorm),
        "explained_var": 1.0 - tf.math.divide_no_nan(residual_var, target_var),
        "grad_norm": grad_norm,
    }


def resolve_checkpoint(path):
    path = Path(path)
    prefix = tf.train.latest_checkpoint(str(path)) if path.is_dir() else str(path)
    if prefix is None or not Path(prefix + ".index").is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    return prefix


def load_profile(path):
    path = Path(path)
    candidates = [Path(str(path) + ".objective.json")]
    if path.is_dir():
        candidates.append(path / PROFILE_NAME)
    else:
        candidates.append(path.parent / PROFILE_NAME)
    for candidate in candidates:
        if candidate.is_file():
            data = json.loads(candidate.read_text())
            if (
                data.get("objective") != OBJECTIVE
                or data.get("risk_horizon") != RISK_HORIZON
            ):
                raise ValueError(f"Unsupported objective profile: {candidate}")
            return data
    return None


def save_profile(path, cfg, n_step=14):
    data = {
        "objective": OBJECTIVE,
        "risk_horizon": RISK_HORIZON,
        "n_step": n_step,
        "search": asdict(cfg),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def save_checkpoint(manager, cfg, n_step):
    prefix = manager.save()
    save_profile(prefix + ".objective.json", cfg, n_step)
    for sidecar in Path(manager.directory).glob("ckpt-*.objective.json"):
        if not Path(str(sidecar).removesuffix(".objective.json") + ".index").exists():
            sidecar.unlink()
    return prefix


def prepare_destination(directory, init_checkpoint, cfg, n_step):
    """Validate initialization/resume before creating the destination profile."""
    destination = Path(directory)
    populated = destination.exists() and any(destination.iterdir())
    if init_checkpoint and populated:
        raise ValueError("--init-checkpoint requires an empty destination directory")
    source = resolve_checkpoint(init_checkpoint) if init_checkpoint else None
    if populated:
        profile = load_profile(destination)
        if profile is None:
            raise ValueError(
                "Legacy checkpoint: use --init-checkpoint with a fresh destination"
            )
        if profile["search"] != asdict(cfg) or profile["n_step"] != n_step:
            raise ValueError(
                "Resume settings differ from the saved objective/search profile"
            )
    else:
        save_profile(destination / PROFILE_NAME, cfg, n_step)
    return source


def risk_calibration(predictions, targets, observed, horizons=(RISK_HORIZON,)):
    """Score cumulative death probabilities only where outcomes were observed."""
    result = {}
    for h in horizons:
        mask = observed[:, h - 1].astype(bool)
        p, y = predictions[mask, h - 1], targets[mask, h - 1]
        result[f"observed_h{h}"] = int(mask.sum())
        result[f"censored_fraction_h{h}"] = float(1 - mask.mean())
        result[f"predicted_h{h}"] = float(p.mean()) if len(p) else None
        result[f"death_rate_h{h}"] = float(y.mean()) if len(y) else None
        result[f"brier_h{h}"] = float(np.mean((p - y) ** 2)) if len(p) else None
    return result
