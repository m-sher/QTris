"""Register AZ placement MCP tools."""

from __future__ import annotations

from typing import Any

from qtris.mcp.protocol import Tool


def _args(schema_props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": schema_props,
        "required": required or [],
        "additionalProperties": False,
    }


def build_tools() -> list[Tool]:
    from qtris.mcp.az_placement import artifacts as ck
    from qtris.mcp.az_placement import observability as obs
    from qtris.mcp.az_placement import probe

    def az_defaults(_: dict[str, Any]) -> Any:
        return ck.defaults()

    def az_list_checkpoints(args: dict[str, Any]) -> Any:
        return ck.list_az_checkpoints(args.get("pattern"))

    def az_inspect_checkpoint(args: dict[str, Any]) -> Any:
        return ck.inspect_checkpoint(args.get("checkpoint_dir"), args.get("mode"))

    def az_list_pool(args: dict[str, Any]) -> Any:
        return ck.list_pool(args.get("checkpoint_dir"))

    def az_get_elo(args: dict[str, Any]) -> Any:
        return ck.get_elo(args.get("checkpoint_dir"))

    def az_list_states(args: dict[str, Any]) -> Any:
        return ck.list_az_states(args.get("states_dir"))

    def az_list_garbage_traces(args: dict[str, Any]) -> Any:
        return ck.list_garbage_traces(args.get("traces_dir"))

    def az_list_tb_runs(args: dict[str, Any]) -> Any:
        return obs.list_tb_runs(
            args.get("tb_root"),
            name_filter=args.get("name_filter"),
            limit=int(args.get("limit", 30)),
        )

    def az_describe_metrics(_: dict[str, Any]) -> Any:
        return obs.describe_az_metrics()

    def az_read_metrics(args: dict[str, Any]) -> Any:
        return obs.read_tb_scalars(
            args.get("tb_run"),
            tags=args.get("tags"),
            tb_root=args.get("tb_root"),
            last_n=int(args.get("last_n", 20)),
        )

    def az_play_games(args: dict[str, Any]) -> Any:
        return probe.play_games(
            args.get("checkpoint_dir"),
            mode=args.get("mode", "single"),
            num_games=int(args.get("num_games", 8)),
            max_steps=int(args.get("max_steps", 200)),
            num_simulations=int(args.get("num_simulations", 16)),
            seed=int(args.get("seed", 0)),
            opponent=args.get("opponent"),
        )

    def az_probe_policy(args: dict[str, Any]) -> Any:
        return probe.probe_policy(
            args.get("checkpoint_dir"),
            mode=args.get("mode", "1v1"),
            seed=int(args.get("seed", 0)),
            top_k=int(args.get("top_k", 8)),
            num_simulations=int(args.get("num_simulations", 0)),
        )

    def az_benchmark_inference(args: dict[str, Any]) -> Any:
        return probe.benchmark_inference(
            args.get("checkpoint_dir"),
            mode=args.get("mode", "1v1"),
            batch_size=int(args.get("batch_size", 16)),
            warmup=int(args.get("warmup", 2)),
            repeats=int(args.get("repeats", 5)),
            num_simulations=int(args.get("num_simulations", 0)),
        )

    return [
        Tool(
            name="az_defaults",
            description=(
                "Return default AZ placement paths, train/demo commands, and MCP scope. "
                "Call first when orienting to this repo's placement AlphaZero setup."
            ),
            input_schema=_args({}),
            handler=az_defaults,
        ),
        Tool(
            name="az_list_checkpoints",
            description=(
                "List AZ-related checkpoint directories (placement_az, 1v1_placement_az, phases, "
                "pretrain) with latest step, pool presence, and on-disk size."
            ),
            input_schema=_args(
                {
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob under repo root, e.g. checkpoints/1v1_placement_az_phase*",
                    },
                }
            ),
            handler=az_list_checkpoints,
        ),
        Tool(
            name="az_inspect_checkpoint",
            description=(
                "Inspect one checkpoint dir: TF pointer, opponent pool, return_scale (solo AZ), "
                "and weight health (NaN/Inf + norms). Loads TF on first use."
            ),
            input_schema=_args(
                {
                    "checkpoint_dir": {
                        "type": "string",
                        "description": "Checkpoint dir or prefix. Default: checkpoints/1v1_placement_az",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["single", "1v1"],
                        "description": "Architecture/value-head mode. Inferred from path when omitted.",
                    },
                }
            ),
            handler=az_inspect_checkpoint,
        ),
        Tool(
            name="az_list_pool",
            description=(
                "List 1v1 opponent-pool snapshots under <ckpt>/pool/gen_* with Elo ratings when available."
            ),
            input_schema=_args(
                {
                    "checkpoint_dir": {
                        "type": "string",
                        "description": "1v1 AZ checkpoint dir. Default: checkpoints/1v1_placement_az",
                    },
                }
            ),
            handler=az_list_pool,
        ),
        Tool(
            name="az_get_elo",
            description=(
                "Read the anchored opponent-pool Elo book (pool/elo.json): ratings, games, "
                "learner vs ref/pool gaps."
            ),
            input_schema=_args(
                {
                    "checkpoint_dir": {
                        "type": "string",
                        "description": "1v1 AZ checkpoint dir. Default: checkpoints/1v1_placement_az",
                    },
                }
            ),
            handler=az_get_elo,
        ),
        Tool(
            name="az_list_states",
            description="Inventory harvested AZ training states under data/az_states/run*.",
            input_schema=_args(
                {
                    "states_dir": {
                        "type": "string",
                        "description": "Override states root. Default: data/az_states",
                    },
                }
            ),
            handler=az_list_states,
        ),
        Tool(
            name="az_list_garbage_traces",
            description=(
                "Summarize garbage_traces tier libraries (solo AZ trace-replay garbage)."
            ),
            input_schema=_args(
                {
                    "traces_dir": {
                        "type": "string",
                        "description": "Override traces root. Default: garbage_traces",
                    },
                }
            ),
            handler=az_list_garbage_traces,
        ),
        Tool(
            name="az_list_tb_runs",
            description=(
                "List recent TensorBoard runs under tb_logs/Tetris (optional name filter)."
            ),
            input_schema=_args(
                {
                    "tb_root": {
                        "type": "string",
                        "description": "TB root. Default: tb_logs/Tetris",
                    },
                    "name_filter": {
                        "type": "string",
                        "description": "Substring filter on run dir name (e.g. az, 1v1, placement).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max runs to return (default 30).",
                    },
                }
            ),
            handler=az_list_tb_runs,
        ),
        Tool(
            name="az_describe_metrics",
            description=(
                "Cheat-sheet of SingleAgentAZLog / OneVsOneAZLog metrics, value targets, "
                "search shaping, and common red flags when debugging training."
            ),
            input_schema=_args({}),
            handler=az_describe_metrics,
        ),
        Tool(
            name="az_read_metrics",
            description=(
                "Read scalar metric series (loss/kl/win_rate/elo/app/...) from a TB run's "
                "event files, the values az_list_tb_runs omits. Returns per-tag count, "
                "first/last/min/max, mean over last_n, and the last_n points. Pair with "
                "az_describe_metrics red flags to diagnose divergence."
            ),
            input_schema=_args(
                {
                    "tb_run": {
                        "type": "string",
                        "description": "Run dir name under tb_root, an absolute path, or omit for the latest run.",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated substring filters on tag name (e.g. 'kl,loss,elo'). Omit for all.",
                    },
                    "tb_root": {
                        "type": "string",
                        "description": "TB root. Default: tb_logs/Tetris",
                    },
                    "last_n": {
                        "type": "integer",
                        "description": "Trailing points returned per tag (default 20).",
                    },
                }
            ),
            handler=az_read_metrics,
        ),
        Tool(
            name="az_play_games",
            description=(
                "Play full games headless and report outcome stats a single-root probe "
                "cannot. mode=single: solo rollout -> app (attack/piece, the b2b-APP target), "
                "b2b, clears, death_rate. mode=1v1: learner vs opponent (pool gen_0 by "
                "default) -> win/loss/draw rates. Loads TF + runs MCTS per move (slow)."
            ),
            input_schema=_args(
                {
                    "checkpoint_dir": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["single", "1v1"],
                        "default": "single",
                    },
                    "num_games": {"type": "integer", "default": 8},
                    "max_steps": {
                        "type": "integer",
                        "default": 200,
                        "description": "Per-game move cap; reaching it counts as a timeout/draw.",
                    },
                    "num_simulations": {"type": "integer", "default": 16},
                    "seed": {"type": "integer", "default": 0},
                    "opponent": {
                        "type": "string",
                        "description": "1v1 only: pool snapshot id (e.g. gen_0), a checkpoint path, or omit for pool gen_0 / self.",
                    },
                }
            ),
            handler=az_play_games,
        ),
        Tool(
            name="az_probe_policy",
            description=(
                "Run the placement net on a fresh board: root value + top candidate priors. "
                "Set num_simulations>0 for full MCTS visit readout (slower). mode=1v1 uses "
                "tanh value + 1v1 search shaping; mode=single uses solo AZ settings."
            ),
            input_schema=_args(
                {
                    "checkpoint_dir": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["single", "1v1"],
                        "default": "1v1",
                    },
                    "seed": {"type": "integer", "default": 0},
                    "top_k": {"type": "integer", "default": 8},
                    "num_simulations": {
                        "type": "integer",
                        "default": 0,
                        "description": "0 = net prior focus; >0 = MCTS sims at probe root.",
                    },
                }
            ),
            handler=az_probe_policy,
        ),
        Tool(
            name="az_benchmark_inference",
            description=(
                "Benchmark batched policy_value forward latency; optionally include batched "
                "MCTS search timing (num_simulations>0)."
            ),
            input_schema=_args(
                {
                    "checkpoint_dir": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["single", "1v1"],
                        "default": "1v1",
                    },
                    "batch_size": {"type": "integer", "default": 16},
                    "warmup": {"type": "integer", "default": 2},
                    "repeats": {"type": "integer", "default": 5},
                    "num_simulations": {
                        "type": "integer",
                        "default": 0,
                        "description": "If >0, also time PlacementMCTS.search over batch_size envs.",
                    },
                }
            ),
            handler=az_benchmark_inference,
        ),
    ]
