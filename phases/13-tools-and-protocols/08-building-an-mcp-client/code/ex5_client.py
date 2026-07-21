#!/usr/bin/env python3
"""Exercise 5 - port the multi-server client to the official MCP SDK.

The SDK's stdio_client + ClientSession replace ALL the hand-rolled
subprocess / reader-thread / JSON plumbing from ex1-ex4. The CLIENT logic
you already wrote (namespace merge + routing) stays; only the TRANSPORT
changes. The whole thing shrinks to ~40 lines.

Run (from the lesson dir):  uv run python code/ex5_client.py
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "ex5_server.py")

# Two logical servers, same script -> both expose `search` (collision!).
SERVERS = {"notes": SERVER, "github": SERVER}


async def main() -> None:
    # --- transport: keep ONE persistent stdio connection per server ----
    # stdio_client and ClientSession are both async context managers, but a
    # plain `async with` PER server would close each before the next opened.
    # An AsyncExitStack lets us enter them all and keep them alive for the
    # whole run, then close them together at the end.
    registry: dict[str, tuple[str, str]] = {}
    sessions: dict[str, ClientSession] = {}

    async with contextlib.AsyncExitStack() as stack:
        for name, path in SERVERS.items():
            params = StdioServerParameters(command=sys.executable, args=[path])
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            sessions[name] = session

        # discover + merge: registry[canonical] -> (server_name, local_name).
        # Prefix-on-collision: a name already taken gets prefixed by its server.
        for name, session in sessions.items():
            result = await session.list_tools()          # -> result.tools: list[Tool]
            for tool in result.tools:
                local = tool.name
                canonical = local if local not in registry else f"{name}/{local}"
                registry[canonical] = (name, local)
                print(f"  {canonical:18s} -> {name}:{local}")

        # route + call: resolve a canonical name, then invoke the real tool.
        target = "github/search"
        srv_name, local_name = registry[target]
        res = await sessions[srv_name].call_tool(local_name, {"query": "hello"})
        print(f"\ncall {target}:")
        for block in res.content:
            print("  ", getattr(block, "text", block))


if __name__ == "__main__":
    asyncio.run(main())
