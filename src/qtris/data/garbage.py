"""Garbage trace selection shared by the datagen and demo entry points."""

from qtris.training.placement_az import _load_trace_pools


def resolve_traces(args):
    """(trace pool, tier name) named by --garbage-traces, or (None, None).

    The tier defaults to the last sorted name, which the library orders weakest first.
    """
    traces_dir = getattr(args, "garbage_traces", None)
    if not traces_dir:
        return None, None
    pools = _load_trace_pools(traces_dir)
    tier = getattr(args, "trace_tier", None) or (list(pools)[-1] if pools else None)
    if tier not in pools:
        raise SystemExit(
            f"trace tier {tier!r} not found in {traces_dir} (have {list(pools)})"
        )
    return pools[tier], tier
