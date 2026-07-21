#!/usr/bin/env python3
"""Real MCP server (FastMCP) for Exercise 5.

Exposes `search` and `ping`. The client launches this script as TWO
processes (named "notes" and "github" on the client side), so you can
watch cross-server tool merging + collision prefixing in the SDK client.

Run by the client via stdio; do NOT run this by hand.
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ex5-demo")


@mcp.tool()
def search(query: str) -> str:
    """Search the demo store for a query string."""
    return f"[demo-server] searched for {query!r}"


@mcp.tool()
def ping() -> str:
    """Health-check ping."""
    return "pong"


if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport
