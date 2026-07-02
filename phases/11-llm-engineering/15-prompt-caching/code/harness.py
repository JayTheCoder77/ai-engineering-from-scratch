# phases/11-llm-engineering/15-prompt-caching/code/harness.py
"""Test harness for prompt caching.

Given a list of RequestLogEntry elements, computes the hit rate and cost
for Anthropic (5m), Anthropic (1h), OpenAI (automatic), and Gemini (explicit).
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class RequestLogEntry:
    timestamp: float  # seconds from start
    prefix_key: str  # cache key (e.g. hash of system prompt)
    prefix_tokens: int
    suffix_tokens: int


# Pricing (USD per 1K tokens)
PRICES = {
    "anthropic": {
        "base": 0.015,
        "write_5m": 0.01875,
        "write_1h": 0.030,
        "read": 0.0015,
    },
    "openai": {"base": 0.005, "write": 0.005, "read": 0.0025},
    "gemini": {
        "base": 0.00125,
        "write": 0.00125,
        "read": 0.0003125,
        "storage_per_1k_per_hour": 0.0000125,
    },
}


@dataclass
class SimulationResult:
    provider_name: str
    writes: int
    reads: int
    misses: int  # cache misses due to TTL expiry or key change
    invalid_sizes: int  # requests ignored because prefix size was too small
    total_cost: float
    hit_rate: float
    savings_pct: float


# Let's generate a realistic request log
# 1. Standard fast conversations (within 5 minutes)
# 2. Re-engagement after 15 minutes (should hit 1h caches, but miss 5m)
# 3. Request with a too-small prefix (below provider floor)
# 4. A completely new prefix key (cache miss due to key change)
SAMPLE_LOG: List[RequestLogEntry] = [
    # Conversation A starts (fast turns)
    RequestLogEntry(
        timestamp=0.0, prefix_key="agent_v1", prefix_tokens=5000, suffix_tokens=200
    ),
    RequestLogEntry(
        timestamp=30.0, prefix_key="agent_v1", prefix_tokens=5000, suffix_tokens=250
    ),
    RequestLogEntry(
        timestamp=60.0, prefix_key="agent_v1", prefix_tokens=5000, suffix_tokens=300
    ),
    # Re-engagement after 8 minutes (480 seconds later)
    RequestLogEntry(
        timestamp=540.0, prefix_key="agent_v1", prefix_tokens=5000, suffix_tokens=200
    ),
    # Request with a too-small prefix (Anthropic/OpenAI limit is 1024, Gemini is 4096)
    RequestLogEntry(
        timestamp=600.0, prefix_key="small_agent", prefix_tokens=500, suffix_tokens=100
    ),
    # A new agent config is requested (different prefix key)
    RequestLogEntry(
        timestamp=700.0, prefix_key="agent_v2", prefix_tokens=6000, suffix_tokens=200
    ),
    RequestLogEntry(
        timestamp=730.0, prefix_key="agent_v2", prefix_tokens=6000, suffix_tokens=250
    ),
    # Long pause - 2 hours later (7200 seconds later)
    RequestLogEntry(
        timestamp=7930.0, prefix_key="agent_v2", prefix_tokens=6000, suffix_tokens=300
    ),
]

def simulate_anthropic(log: List[RequestLogEntry], ttl_seconds: float) -> SimulationResult:
    # Set pricing based on TTL
    prices = PRICES["anthropic"]
    write_rate = prices["write_1h"] if ttl_seconds > 300 else prices["write_5m"]
    read_rate = prices["read"]
    base_rate = prices["base"]

    writes = 0
    reads = 0
    misses = 0
    invalid_sizes = 0
    total_cost = 0.0

    # cache holds {prefix_key: last_accessed_timestamp}
    cache: Dict[str, float] = {}

    # Calculate baseline (no cache) cost for savings calculation
    baseline_cost = sum((r.prefix_tokens + r.suffix_tokens) / 1000 * base_rate for r in log)

    for r in log:
        if r.prefix_tokens < 1024:
            # Too small to cache
            invalid_sizes += 1
            total_cost += ((r.prefix_tokens + r.suffix_tokens) / 1000) * base_rate
            continue

        # Check cache state
        if r.prefix_key in cache:
            last_accessed = cache[r.prefix_key]
            # Has it expired?
            if r.timestamp - last_accessed >= ttl_seconds:
                # Expired -> Cache Miss (Write)
                misses += 1
                writes += 1
                total_cost += (r.prefix_tokens / 1000) * write_rate + (r.suffix_tokens / 1000) * base_rate
                cache[r.prefix_key] = r.timestamp
            else:
                # Active -> Cache Hit (Read)
                reads += 1
                total_cost += (r.prefix_tokens / 1000) * read_rate + (r.suffix_tokens / 1000) * base_rate
                cache[r.prefix_key] = r.timestamp
        else:
            # First time seeing this key -> Cache Miss (Write)
            writes += 1
            total_cost += (r.prefix_tokens / 1000) * write_rate + (r.suffix_tokens / 1000) * base_rate
            cache[r.prefix_key] = r.timestamp

    # Calculate metrics
    total_requests = len(log) - invalid_sizes
    hit_rate = reads / total_requests if total_requests > 0 else 0.0
    savings_pct = (1 - (total_cost / baseline_cost)) * 100 if baseline_cost > 0 else 0.0

    return SimulationResult(
        provider_name=f"Anthropic ({int(ttl_seconds/60)}m TTL)",
        writes=writes,
        reads=reads,
        misses=misses,
        invalid_sizes=invalid_sizes,
        total_cost=total_cost,
        hit_rate=hit_rate,
        savings_pct=savings_pct
    )

def simulate_openai(log: List[RequestLogEntry], ttl_seconds: float) -> SimulationResult:
    # Set pricing based on TTL
    prices = PRICES["openai"]
    ttl_seconds = 3600.0
    write_rate = prices["write"]
    read_rate = prices["read"]
    base_rate = prices["base"]

    writes = 0
    reads = 0
    misses = 0
    invalid_sizes = 0
    total_cost = 0.0

    # cache holds {prefix_key: last_accessed_timestamp}
    cache: Dict[str, float] = {}

    # Calculate baseline (no cache) cost for savings calculation
    baseline_cost = sum((r.prefix_tokens + r.suffix_tokens) / 1000 * base_rate for r in log)

    for r in log:
        if r.prefix_tokens < 1024:
            # Too small to cache
            invalid_sizes += 1
            total_cost += ((r.prefix_tokens + r.suffix_tokens) / 1000) * base_rate
            continue

        # Check cache state
        if r.prefix_key in cache:
            last_accessed = cache[r.prefix_key]
            # Has it expired?
            if r.timestamp - last_accessed >= ttl_seconds:
                # Expired -> Cache Miss (Write)
                misses += 1
                writes += 1
                total_cost += (r.prefix_tokens / 1000) * write_rate + (r.suffix_tokens / 1000) * base_rate
                cache[r.prefix_key] = r.timestamp
            else:
                # Active -> Cache Hit (Read)
                reads += 1
                total_cost += (r.prefix_tokens / 1000) * read_rate + (r.suffix_tokens / 1000) * base_rate
                cache[r.prefix_key] = r.timestamp
        else:
            # First time seeing this key -> Cache Miss (Write)
            writes += 1
            total_cost += (r.prefix_tokens / 1000) * write_rate + (r.suffix_tokens / 1000) * base_rate
            cache[r.prefix_key] = r.timestamp

    # Calculate metrics
    total_requests = len(log) - invalid_sizes
    hit_rate = reads / total_requests if total_requests > 0 else 0.0
    savings_pct = (1 - (total_cost / baseline_cost)) * 100 if baseline_cost > 0 else 0.0

    return SimulationResult(
        provider_name=f"Openai ({int(ttl_seconds/60)}m TTL)",
        writes=writes,
        reads=reads,
        misses=misses,
        invalid_sizes=invalid_sizes,
        total_cost=total_cost,
        hit_rate=hit_rate,
        savings_pct=savings_pct
    )

def simulate_gemini(log: List[RequestLogEntry], ttl_seconds: float) -> SimulationResult:
    # Set pricing based on TTL
    prices = PRICES["gemini"]
    ttl_seconds = 3600.0
    write_rate = prices["write"]
    read_rate = prices["read"]
    base_rate = prices["base"]

    writes = 0
    reads = 0
    misses = 0
    invalid_sizes = 0
    total_cost = 0.0

    # cache holds {prefix_key: last_accessed_timestamp}
    cache: Dict[str, float] = {}

    # Calculate baseline (no cache) cost for savings calculation
    baseline_cost = sum((r.prefix_tokens + r.suffix_tokens) / 1000 * base_rate for r in log)

    for r in log:
        if r.prefix_tokens < 4096:
            # Too small to cache
            invalid_sizes += 1
            total_cost += ((r.prefix_tokens + r.suffix_tokens) / 1000) * base_rate
            continue

        # Check cache state
        if r.prefix_key in cache:
            last_accessed = cache[r.prefix_key]
            # Has it expired?
            if r.timestamp - last_accessed >= ttl_seconds:
                # Expired -> Cache Miss (Write)
                misses += 1
                writes += 1
                storage_cost = (r.prefix_tokens / 1000) * prices["storage_per_1k_per_hour"] * (ttl_seconds /3600)
                total_cost += storage_cost + (r.prefix_tokens / 1000) * write_rate + (r.suffix_tokens / 1000) * base_rate 
                cache[r.prefix_key] = r.timestamp
            else:
                # Active -> Cache Hit (Read)
                reads += 1
                total_cost += (r.prefix_tokens / 1000) * read_rate + (r.suffix_tokens / 1000) * base_rate
                # cache[r.prefix_key] = r.timestamp
        else:
            # First time seeing this key -> Cache Miss (Write)
            writes += 1
            storage_cost = (r.prefix_tokens / 1000) * prices["storage_per_1k_per_hour"] * (ttl_seconds /3600)
            total_cost += storage_cost + (r.prefix_tokens / 1000) * write_rate + (r.suffix_tokens / 1000) * base_rate
            cache[r.prefix_key] = r.timestamp

    # Calculate metrics
    total_requests = len(log) - invalid_sizes
    hit_rate = reads / total_requests if total_requests > 0 else 0.0
    savings_pct = (1 - (total_cost / baseline_cost)) * 100 if baseline_cost > 0 else 0.0

    return SimulationResult(
        provider_name=f"gemini ({int(ttl_seconds/60)}m TTL)",
        writes=writes,
        reads=reads,
        misses=misses,
        invalid_sizes=invalid_sizes,
        total_cost=total_cost,
        hit_rate=hit_rate,
        savings_pct=savings_pct
    )

def main():
        print(f"Running cache simulation with {len(SAMPLE_LOG)} log entries...\n")

        results = [
            simulate_anthropic(SAMPLE_LOG, ttl_seconds=300),   # 5m TTL
            simulate_anthropic(SAMPLE_LOG, ttl_seconds=3600),  # 1h TTL
            simulate_openai(SAMPLE_LOG, ttl_seconds=3600),     # 1h automatic
            simulate_gemini(SAMPLE_LOG, ttl_seconds=3600),     # 1h explicit
        ]

        # Print a markdown table
        print(f"| {'Provider / Config':<28} | {'Writes':<6} | {'Reads':<5} | {'Misses':<6} | {'Invalid':<7} | {'Total Cost':<10} | {'Hit Rate':<8} | {'Savings':<7} |")
        print(f"| {'-'*28} | {'-'*6} | {'-'*5} | {'-'*6} | {'-'*7} | {'-'*10} | {'-'*8} | {'-'*7} |")
        for r in results:
            print(
                f"| {r.provider_name:<28} "
                f"| {r.writes:<6} "
                f"| {r.reads:<5} "
                f"| {r.misses:<6} "
                f"| {r.invalid_sizes:<7} "
                f"| ${r.total_cost:<9.5f} "
                f"| {r.hit_rate*100:<7.1f}% "
                f"| {r.savings_pct:<6.1f}% |"
            )

if __name__ == "__main__":
    main()