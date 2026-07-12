import subprocess
import json

def main():
    # Spawn the child process
    child = subprocess.Popen(
        ["npx", "@wkalidev/multichain-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # line buffered
    )

    # Payloads to send
    payloads = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "my-client", "version": "1.0" }
            }
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {
                "cursor": "optional-cursor-value"
            }
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_balance",
                "arguments": {
                    "address": "SP2C2YFP12AJZB4MABJBAJ55XECVS7E4PMMZ89YZR",
                    "chain": "stacks"
                }
            }
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "get_prices",
                "arguments": {
                    "symbols": ["STX", "CELO", "ETH"]
                }
            }
        }
    ]

    # Write payloads to child.stdin
    for payload in payloads:
        child.stdin.write(json.dumps(payload) + "\n")
    child.stdin.flush()
    child.stdin.close()

    # Read stdout line by line
    lines = []
    for line in child.stdout:
        lines.append(line.strip())

    # Wait for the child process to terminate
    child.wait()

    req = 0
    res = 0
    notif = 0
    lifecycle = 0
    operation = 0

    for line in lines:
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        # classify kind — check which keys exist in msg
        if "id" in msg and "method" in msg:
            req += 1
        elif "id" in msg and ("result" in msg or "error" in msg):
            res += 1
        else:
            notif += 1

        # categorize phase (for every message)
        msg_id = msg.get("id")
        if msg_id == 1:
            lifecycle += 1
        elif msg_id in (2, 3, 4):
            operation += 1

    # print totals
    print(f"requests: {req}, responses: {res}, notifications: {notif}")
    print(f"lifecycle: {lifecycle}, operation: {operation}")
    
    total_lines = len(lines)
    if total_lines > 0:
        print(f"fraction lifecycle: {(lifecycle / total_lines):.2f}")
        print(f"fraction operation: {(operation / total_lines):.2f}")
    else:
        print("fraction lifecycle: 0.00")
        print("fraction operation: 0.00")

if __name__ == "__main__":
    main()
