import { spawn } from "child_process";
import fs from "fs";
import { createInterface } from "readline";

const child = spawn("npx", ["@wkalidev/multichain-mcp"], { stdio: ["pipe", "pipe", "pipe"] });

const rl = createInterface({ input: child.stdout });
const lines: string[] = [];
rl.on("line", (line) => lines.push(line));

child.stdin.write(JSON.stringify({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": { "name": "my-client", "version": "1.0" }
    },
}) + "\n");

child.stdin.write(JSON.stringify(
    {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
) + "\n");

child.stdin.write(JSON.stringify(
    {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {
            "cursor": "optional-cursor-value"
        }
    }
) + "\n");

child.stdin.write(JSON.stringify(
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
    }
) + "\n");

child.stdin.write(JSON.stringify(
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
) + "\n");

child.stdin.end();
child.on("close", () => {
    let req = 0, res = 0, notif = 0;
    let lifecycle = 0, operation = 0;

    for (const line of lines) {
        const msg = JSON.parse(line);

        // classify kind — check which keys exist in msg

        if ("id" in msg && "method" in msg) req++;
        else if ("id" in msg && ("result" in msg || "error" in msg)) res++;
        else notif++;

        // categorize phase (for every message)
        if (msg.id === 1) {
            lifecycle++;
        } else if (msg.id === 2 || msg.id === 3 || msg.id === 4) {
            operation++;
        }
    }

    // print totals
    console.log(`requests: ${req}, responses: ${res}, notifications: ${notif}`);
    console.log(`lifecycle: ${lifecycle}, operation: ${operation}`);
    console.log(`fraction lifecycle: ${(lifecycle / lines.length).toFixed(2)}`);
    console.log(`fraction operation: ${(operation / lines.length).toFixed(2)}`);
});
