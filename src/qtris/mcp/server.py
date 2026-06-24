"""qtris-mcp entrypoint: AZ placement debug tools over stdio MCP."""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Ensure repo root is on path when launched outside `uv run`.
    _maybe_fix_path()

    # Reduce TF log spam on tool calls (agents parse tool stdout as MCP).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("QTRIS_MCP_GPU", ""))

    from qtris.mcp.az_placement.tools import build_tools
    from qtris.mcp.protocol import McpServer

    tools = build_tools()
    server = McpServer(name="qtris-az-placement", version="0.1.0", tools=tools)
    # Banner must not go to stdout (MCP uses stdout for JSON-RPC).
    print(
        f"qtris-az-placement MCP: {len(tools)} tools (placement AZ only). Waiting on stdin.",
        file=sys.stderr,
        flush=True,
    )
    server.run()


def _maybe_fix_path() -> None:
    from pathlib import Path

    here = Path(__file__).resolve()
    for parent in here.parents:
        src = parent / "src"
        if (parent / "pyproject.toml").exists() and (src / "qtris").is_dir():
            src_s = str(src)
            if src_s not in sys.path:
                sys.path.insert(0, src_s)
            break


if __name__ == "__main__":
    main()
