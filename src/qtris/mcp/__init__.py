"""MCP tool suite for agent-assisted debugging of QTris models.

Currently scoped to the placement AlphaZero (AZ) family only.
"""

__all__ = ["main"]


def main() -> None:
    from qtris.mcp.server import main as _main

    _main()
