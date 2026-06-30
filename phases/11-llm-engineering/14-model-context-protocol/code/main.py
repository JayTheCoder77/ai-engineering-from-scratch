from mcp.server.fastmcp import FastMCP
from pathlib import Path
from collections import deque

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """subtract two integers."""
    return a - b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'


# Only files under this directory are allowed.
ALLOWED_ROOT = Path("/var/log").resolve()


@mcp.resource("log://app")
def app_log() -> str:
    """
    Return the last 100 lines of /var/log/app.log.
    """

    requested_path = (ALLOWED_ROOT / "app.log").resolve()

    # Prevent directory traversal.
    if ALLOWED_ROOT not in requested_path.parents and requested_path != ALLOWED_ROOT:
        raise PermissionError("Access outside allowed root.")

    if not requested_path.exists():
        return "Log file does not exist."

    with requested_path.open("r", encoding="utf-8", errors="ignore") as f:
        last_lines = deque(f, maxlen=100)

    return "".join(last_lines)


@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")


# "mcpServers": {
# "demo-server": {
#     "command": "wsl",
#     "args": [
#     "bash",
#     "-lc",
#     "cd /home/jayant/projects/ai-engineering-from-scratch/phases/11-llm-engineering/14-model-context-protocol/code && uv run python main.py"
#     ]
# }
# },



# from mcp.client.stdio import StdioServerParameters, stdio_client
# from mcp import ClientSession

# params = StdioServerParameters(command="python", args=["server.py"])

# async def call_add(a: int, b: int) -> int:
#     async with stdio_client(params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             tools = await session.list_tools()
#             result = await session.call_tool("add", {"a": a, "b": b})
#             return int(result.content[0].text)

# Inside the server entrypoint
# mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)

# {
#   "mcpServers": {
#     "demo": {
#       "type": "http",
#       "url": "https://tools.example.com/mcp"
#     }
#   }
# }