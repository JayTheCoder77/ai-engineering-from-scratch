"""Phase 13 Lesson 03 - parallel and streaming tool calls.

Two demos, stdlib only:
  1. Three-city weather run, sequential vs parallel (thread pool).
     Measures wall-clock and shows the max vs sum pattern.
  2. Stream accumulator for out-of-order argument chunks.
     Replays a fake OpenAI-shaped stream of three interleaved parallel calls
     and reassembles each per-id before executing.

Run: python code/main.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import asyncio

# ------------------------------------------------------------------
# demo 1: sequential vs parallel weather lookup
# ------------------------------------------------------------------

SIMULATED_LATENCY_MS = {"Bengaluru": 100, "Tokyo": 100, "Zurich": 200}


def executor_weather(city: str) -> dict:
    latency = SIMULATED_LATENCY_MS.get(city, 500)
    time.sleep(latency / 1000.0)
    return {"city": city, "temp_c": hash(city) % 35}

async def executor_weather_async(city: str) -> dict:
    latency = SIMULATED_LATENCY_MS.get(city, 500)
    await asyncio.sleep(latency / 1000.0)
    return {"city": city, "temp_c": hash(city) % 35}


def run_sequential(cities: list[str]) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    results = [executor_weather(c) for c in cities]
    dt_ms = (time.perf_counter() - start) * 1000
    return dt_ms, results


def run_parallel(cities: list[str]) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(cities)) as pool:
        results = list(pool.map(executor_weather, cities))
    dt_ms = (time.perf_counter() - start) * 1000
    return dt_ms, results

async def run_parallel_async(cities: list[str]) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    results = await asyncio.gather(*[executor_weather_async(city) for city in cities])
    dt_ms = (time.perf_counter() - start) * 1000
    return dt_ms, results


# ------------------------------------------------------------------
# demo 2: stream accumulator
# ------------------------------------------------------------------

@dataclass
class CallBuffer:
    id: str
    name: str = ""
    args_buf: str = ""
    done: bool = False

    def try_parse(self) -> dict | None:
        if not self.done:
            return None
        return json.loads(self.args_buf)


@dataclass
class StreamAccumulator:
    buffers: dict[str, CallBuffer] = field(default_factory=dict)

    def on_event(self, event: dict) -> tuple[list[CallBuffer] , list[CallBuffer]]:
        kind = event["type"]
        idx = event.get("id")
        completed: list[CallBuffer] = []
        cancelled: list[CallBuffer] = []
        if kind == "call_start":
            self.buffers[idx] = CallBuffer(id=idx, name=event["name"])
        elif kind == "args_delta":
            buf = self.buffers[idx]
            buf.args_buf += event["chunk"]
        elif kind == "call_cancelled":
            buf = self.buffers[idx]
            buf.done = False
            cancelled.append(buf)
        elif kind == "call_stop":
            buf = self.buffers[idx]
            buf.done = True
            completed.append(buf)
        return (completed , cancelled)

DEPENDENCY_GRAPH = {
    "write_file" : ["create_file"],
    "append_file" : ["create_file"]
}

def has_dependency_conflict(calls : list[str]) -> bool:
    tool_names = set(calls)
    for dependent , prerequisites in DEPENDENCY_GRAPH.items():
        if dependent in tool_names and any(p in tool_names for p in prerequisites):
            return True
    return False

def mock_executor(tool_name : str) -> dict:
    time.sleep(5)
    return "execution done"
    
def run_dependency_aware(tool_calls : list[dict]) -> None:
    batches = dependency_batches(tool_calls)
    print(f"  batches: {[[c['tool_name'] for c in b] for b in batches if b]}")
    for i, batch in enumerate(batches):
        if not batch:
            continue
        names = [c["tool_name"] for c in batch]
        print(f"Executing batch {i}: {names}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(mock_executor, names))

def dependency_batches(tool_calls : list[dict]) -> list[list[dict]]:
    names = {c["tool_name"] for c in tool_calls}
    batch0 = []
    batch1 = []
    for c in tool_calls:
        prereqs = DEPENDENCY_GRAPH.get(c["tool_name"] , [])
        if prereqs and any(p in names for p in prereqs):
            batch1.append(c)
        else:
            batch0.append(c)
    return [batch0, batch1] 

def fake_openai_stream():
    """Three interleaved parallel calls. Real streams look like this."""
    yield {"type": "call_start", "id": "call_A", "name": "get_weather"}
    yield {"type": "call_start", "id": "call_B", "name": "get_weather"}
    yield {"type": "call_start", "id": "call_C", "name": "get_weather"}
    yield {"type": "args_delta", "id": "call_A", "chunk": '{"city"'}
    yield {"type": "args_delta", "id": "call_B", "chunk": '{"city'}
    yield {"type": "args_delta", "id": "call_A", "chunk": ':"Beng'}
    yield {"type": "args_delta", "id": "call_C", "chunk": '{"city":"Zu'}
    yield {"type": "call_cancelled", "id": "call_C"}
    yield {"type": "args_delta", "id": "call_A", "chunk": 'aluru"}'}
    yield {"type": "call_stop", "id": "call_A"}
    yield {"type": "args_delta", "id": "call_B", "chunk": '":"Tokyo"}'}
    yield {"type": "call_stop", "id": "call_B"}


def replay_and_execute() -> dict[str, dict]:
    acc = StreamAccumulator()
    results: dict[str, dict] = {}
    in_flight: dict[str, "Future"] = {}  # type: ignore
    with ThreadPoolExecutor(max_workers=4) as pool:
        for event in fake_openai_stream():
            completed , cancelled = acc.on_event(event)
            for buf in completed:
                args = buf.try_parse()
                print(f"  call {buf.id} args complete -> {args}")
                in_flight[buf.id] = pool.submit(executor_weather, args["city"])
            for buf in cancelled:
                print(f" call {buf.id} CANCELLED (partial args: {buf.args_buf})")
                del acc.buffers[buf.id]
        for cid, fut in in_flight.items():
            results[cid] = fut.result()
    return results




# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 03 - PARALLEL AND STREAMING TOOL CALLS")
    print("=" * 72)

    cities = ["Bengaluru", "Tokyo", "Zurich"]
    sum_lat = sum(SIMULATED_LATENCY_MS.values())
    max_lat = max(SIMULATED_LATENCY_MS.values())

    print("\n--- demo 1: three-city weather (simulated) ---")
    print(f"per-city simulated latency : {SIMULATED_LATENCY_MS}")
    print(f"theoretical sequential     : {sum_lat} ms  (sum)")
    print(f"theoretical (async and threadpool) parallel       : {max_lat} ms  (max)")

    seq_ms, seq_res = run_sequential(cities)
    par_ms, par_res = run_parallel(cities)
    async_par_ms, async_par_res = asyncio.run(run_parallel_async(cities))
    print(f"\nactual sequential : {seq_ms:.0f} ms")
    print(f"actual parallel   : {par_ms:.0f} ms")
    print(f"actual async parallel   : {async_par_ms:.0f} ms")
    speedup = seq_ms / par_ms if par_ms else 0
    async_speedup = seq_ms / async_par_ms if async_par_ms else 0
    print(f"speedup           : {speedup:.2f}x (thread pool)")
    print(f"speedup           : {async_speedup:.2f}x (async)")

    print("\n--- demo 2: stream accumulator ---")
    print("replaying fake interleaved stream of three parallel calls ...")
    results = replay_and_execute()
    print("\nfinal results (keyed by tool_call_id):")
    for cid, r in results.items():
        print(f"  {cid} -> {r}")


    print("\n--- demo 3: dependency-aware tool calls ---")
    run_dependency_aware([{"tool_name": "write_file"} , {"tool_name": "create_file"}])
    run_dependency_aware([{"tool_name": "write_file"}, {"tool_name": "executor_weather"}])
    
if __name__ == "__main__":
    main()
