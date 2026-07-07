"""Phase 13 Lesson 01 - the tool interface, four-step loop, no LLM.

Implements the describe -> decide -> execute -> observe cycle used by every
2026 tool-calling stack (OpenAI, Anthropic, Gemini, MCP, A2A). The "decide"
step is faked with a keyword router so the loop runs offline; replace it with
any real provider in Lesson 02.

The harness:
  - registers three tools (add, get_time, get_weather)
  - validates tool-call arguments against a minimal JSON Schema subset
  - prints each step so you can read the choreography
  - bounds iteration at MAX_TURNS to prevent runaway loops

Run: python code/main.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


MAX_TURNS = 5

load_dotenv()
unsloth_api_key = os.environ.get("UNSLOTH_STUDIO_AUTH_TOKEN")

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    executor: Callable[[dict], Any]
    consequential: bool = False


def tool_add(args: dict) -> dict:
    return {"sum": args["a"] + args["b"]}


def tool_get_time(args: dict) -> dict:
    tz = args.get("timezone", "UTC")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return {"now": now, "timezone": tz}


def tool_get_weather(args: dict) -> dict:
    fake = {"Bengaluru": 28, "Tokyo": 12, "Zurich": 4, "Lagos": 31}
    city = args["city"]
    units = args.get("units", "celsius")
    temp = fake.get(city, 20)
    return {"city": city, "temp": temp, "units": units}

def tool_stock_price(args: dict) -> dict:
    ticker = args.get("ticker")
    fake_prices = {
        "AAPL": 290.50,
        "GOOGL": 3200.75,
        "MSFT": 285.33,
    }
    price = fake_prices.get(ticker.upper(), 150.00)
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}

def tool_execute_trade(args:dict) -> dict:
    ticker = args.get("ticker")
    action = args.get("action")
    qty = args.get("quantity")

    fake_prices = {
        "AAPL": 290.50,
        "GOOGL": 3200.75,
        "MSFT": 285.33,
    }
    price = fake_prices.get(ticker.upper(), 150.00)
    total_cost = qty * price
    return {"status": "filled", "ticker": ticker, "action": action, "quantity": qty, "total_cost": total_cost, "currency": "USD"}

REGISTRY: list[Tool] = [
    Tool(
        name="add",
        description=(
            "Use when the user asks for the sum of two numbers. "
            "Do not use for subtraction, product, or symbolic algebra."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        executor=tool_add,
    ),
    Tool(
        name="get_time",
        description=(
            "Use when the user asks what time it is. "
            "Do not use for historical dates or future scheduling."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {"type": "string"},
            },
            "required": [],
        },
        executor=tool_get_time,
    ),
    Tool(
        name="get_weather",
        description=(
            "Use when the user asks about current conditions in a named city. "
            "Do not use for forecasts or historical weather data."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
        executor=tool_get_weather,
    ),
    Tool(
        name="get_stock_price",
        description="Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries.",
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
        executor=tool_stock_price
    ),
    Tool(
        name="execute_trade",
        description="Use when the user wants to buy or sell a stock. Do not use for checking stock prices.",
        consequential=True,
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "action" : {"type" : "string"},
                "quantity" : {"type" : "number"}
            },
            "required": ["ticker" , "action" , "quantity"],
        },
        executor=tool_execute_trade
    ),
]


def validate(schema: dict, value: Any) -> list[str]:
    errors: list[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"expected object, got {type(value).__name__}"]
        for key , _ in value.items():
            if key not in schema.get("properties" , {}):
                errors.append("Unexpected arg")
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"missing required field '{field}'")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate(sub, value[key]))
        return errors
    if t == "number" and not isinstance(value, (int, float)):
        errors.append(f"expected number, got {type(value).__name__}")
    if t == "string" and not isinstance(value, str):
        errors.append(f"expected string, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"value {value!r} not in enum {schema['enum']}")
    return errors


def fake_decide(user_msg: str, history: list[dict]) -> dict:
    """Stand-in for the model. Routes by keyword so the loop runs offline.

    Production substitute: swap this for provider.chat.completions.create with
    tools=[t.input_schema for t in REGISTRY]. Same return shape.
    """
    last = history[-1] if history else {}
    if last.get("role") == "tool":
        return {"content": f"Final answer built from tool output: {last.get('content')}"}
    msg = user_msg.lower()
    
    if "test missing field" in msg:
        return {
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "get_weather",
                    "arguments": {},
                }
            ]
        }
    if "test extra field" in msg:
        return {
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "get_weather",
                    "arguments": {"city": "Tokyo", "extra_arg": "some_value"},
                }
            ]
        }

    if re.search(r"\b(add|sum|plus)\b", msg):
        nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", msg)]
        if len(nums) >= 2:
            return {
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": "add",
                        "arguments": {"a": nums[0], "b": nums[1]},
                    }
                ]
            }
    if "time" in msg:
        return {
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "get_time",
                    "arguments": {"timezone": "UTC"},
                }
            ]
        }
    if "stock" in msg or "price" in msg:
        ticker_expr = re.search(r"\b([A-Z]{3,5})\b", user_msg)
        if ticker_expr:
            return {
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": "get_stock_price",
                        "arguments": {"ticker": ticker_expr.group(1)},
                    }
                ]
            }
    if "buy" in msg or "sell" in msg:
        ticker_expr = re.search(r"\b([A-Z]{3,5})\b", user_msg)
        action_expr = "buy" if "buy" in msg else "sell"
        amount_match = re.search(r"amount:? (\d+\.?\d*)",msg) 
        quantity_val = float(amount_match.group(1)) if amount_match else None
        
        return{
            "tool_calls": [
                {
                    "id" : f"call_{uuid.uuid4().hex[:8]}",
                    "name" : "execute_trade",
                    "arguments":{
                        "ticker" : ticker_expr.group(1),
                        "action" : action_expr,
                        "quantity" : quantity_val
                    }
                }
            ]
        }
    match = re.search(r"weather in (\w+)", msg)
    if match:
        city = match.group(1).title()
        return {
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "get_weather",
                    "arguments": {"city": city, "units": "celsius"},
                }
            ]
        }
    return {"content": "I cannot route that query to any registered tool."}

def decide(user_msg: str, history: list[dict]) -> dict:
    """Queries your local server running the quantized Gemma model."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in REGISTRY
    ]

    client = OpenAI(
        base_url="http://localhost:8888/v1",              
        api_key=unsloth_api_key
    )

    response = client.chat.completions.create(
        model="unsloth/gemma-4-12B-it-qat-GGUF",                               
        messages=[  
            {"role": "user", "content": user_msg}
        ],
        tools=tools,
    )
    message = response.choices[0].message

    if message.tool_calls:
        return {
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]
        }

    return {"content": message.content}

def run_loop(user_msg: str) -> None:
    print("=" * 72)
    print(f"USER : {user_msg}")
    print("-" * 72)
    tools_by_name = {t.name: t for t in REGISTRY}
    history: list[dict] = [{"role": "user", "content": user_msg}]
    for turn in range(1, MAX_TURNS + 1):
        # decision = fake_decide(user_msg, history)
        decision = decide(user_msg, history)
        if "content" in decision:
            print(f"TURN {turn} DECIDE : final answer")
            print(f"MODEL : {decision['content']}")
            return
        for call in decision["tool_calls"]:
            tool = tools_by_name.get(call["name"])
            print(f"TURN {turn} DECIDE : call {call['name']} id={call['id']}")
            print(f"           args = {json.dumps(call['arguments'])}")
            if tool is None:
                print(f"           ERROR : unknown tool {call['name']}")
                return
            errs = validate(tool.input_schema, call["arguments"])
            if errs:
                print(f"           VALIDATION ERRORS : {errs}")
                return
            if tool.consequential:
                print("           GATE : tool is consequential, would confirm")
                choice = input("confirm execution (y/n)")
                if choice.lower() == "n":
                    print("   (aborted by user)")
                    return
                print("   (approved)")
            start = time.perf_counter()
            result = tool.executor(call["arguments"])
            ms = (time.perf_counter() - start) * 1000
            print(f"TURN {turn} EXECUTE: {tool.name} -> {json.dumps(result)}"
                  f" [{ms:.2f} ms]")
            history.append({
                "role": "tool", "id": call["id"],
                "name": tool.name, "content": json.dumps(result),
            })
        print(f"TURN {turn} OBSERVE: history length = {len(history)}")
    print("LOOP TERMINATED : hit MAX_TURNS circuit breaker")


def describe_registry() -> None:
    print("TOOL REGISTRY")
    print("-" * 72)
    for t in REGISTRY:
        kind = "consequential" if t.consequential else "pure"
        print(f"  {t.name:14s} [{kind}] - {t.description}")
    print()


def main() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 01 - THE TOOL INTERFACE")
    print("=" * 72)
    describe_registry()
    for query in (
        "please add 7 and 35",
        "what time is it?",
        "tell me the weather in Bengaluru",
        "write me a haiku about tea",
        "what is the current stock price of AAPL?",
        "test missing field",
        "test extra field",
        "buy AAPL amount 10",
        "sell MSFT amount 10",
    ):
        run_loop(query)
        print()


if __name__ == "__main__":
    main()
