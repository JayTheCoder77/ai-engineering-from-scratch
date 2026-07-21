#!/usr/bin/env python3
"""Real (tiny) stdio MCP-style server for Exercise 1.

Run as a CHILD PROCESS by ex1_client.py. Reads one JSON-RPC-ish request
per line from stdin, writes one JSON response per line to stdout. This is
the process you will SIGTERM so the client can detect its death via EOF.

It does NOT need to be perfect MCP — it just needs to stay alive, answer
a couple of methods, and die cleanly when killed so stdout hits EOF.
"""
import json
import sys

TOOLS = [
    {"name": "ping", "description": "reply pong",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "search", "description": "search (dummy)",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
]


def handle(method: str, params: dict) -> dict:
    if method == "initialize":
        return {"protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "proc-server"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        return {"content": [{"type": "text",
                             "text": f"[proc-server] {params.get('name')} ran"}],
                "isError": False}
    raise ValueError(method)


def main() -> None:
    # `for line in sys.stdin` blocks until a full line OR stdin EOF.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method")
        params = req.get("params", {})
        try:
            result = handle(method, params)
            resp = {"jsonrpc": "2.0", "id": req.get("id"), "result": result}
        except Exception as e:  # never let the loop die on a bad request
            resp = {"jsonrpc": "2.0", "id": req.get("id"),
                    "error": {"message": str(e)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
