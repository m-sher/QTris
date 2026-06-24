"""Minimal MCP stdio server (JSON-RPC 2.0 over newline-delimited stdin/stdout).

Avoids the `mcp` PyPI package, which conflicts with tf-agents' typing-extensions pin.
Implements only what Grok/Claude need: initialize, tools/list, tools/call.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


class McpServer:
    """stdio MCP server with a fixed tool registry."""

    def __init__(self, name: str, version: str, tools: list[Tool]):
        self.name = name
        self.version = version
        self._tools = {t.name: t for t in tools}

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                self._write(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {e}",
                        },
                    }
                )
                continue
            if not isinstance(msg, dict):
                continue
            # Notifications have no id and get no response.
            if "id" not in msg and msg.get("method", "").endswith("initialized"):
                continue
            if "id" not in msg and msg.get("method") == "notifications/initialized":
                continue
            resp = self._dispatch(msg)
            if resp is not None:
                self._write(resp)

    def _write(self, obj: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(obj, default=_json_default) + "\n")
        sys.stdout.flush()

    def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        req_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": self.version},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        }
                        for t in self._tools.values()
                    ]
                }
            elif method == "tools/call":
                name = params.get("name", "")
                args = params.get("arguments") or {}
                tool = self._tools.get(name)
                if tool is None:
                    return self._error(req_id, -32601, f"Unknown tool: {name}")
                payload = tool.handler(args if isinstance(args, dict) else {})
                text = (
                    payload
                    if isinstance(payload, str)
                    else json.dumps(payload, indent=2, default=_json_default)
                )
                result = {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                }
            elif method == "ping":
                result = {}
            elif method.startswith("notifications/"):
                return None
            else:
                return self._error(req_id, -32601, f"Method not found: {method}")
        except Exception as e:
            tb = traceback.format_exc()
            err_text = f"{type(e).__name__}: {e}\n\n{tb}"
            if method == "tools/call":
                result = {
                    "content": [{"type": "text", "text": err_text}],
                    "isError": True,
                }
            else:
                return self._error(req_id, -32000, str(e), data=tb)

        if req_id is None:
            return None
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id, code: int, message: str, data: Any = None) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)
