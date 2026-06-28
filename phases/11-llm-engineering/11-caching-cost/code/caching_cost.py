import hashlib
import time
import json
import math
from dataclasses import dataclass, field
from datetime import datetime


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
}

class CircuitBreaker:
    def __init__(self , training_data):
        self.cost_tracker = CostTracker(monthly_budget=0.1)
        self.cache = SemanticCacheLRU()
        self.model = EmbeddingRouter(training_data)
        self.warning_triggered = False
        self.throttle_triggered = False
        self.stop_triggered = False
    
    def get_status(self):
        spent = self.cost_tracker.total_cost()
        pct = spent / self.cost_tracker.monthly_budget
        stop = pct >= 0.95
        throttle = pct >= 0.85 and pct < 0.95
        warning = pct >= 0.70 and pct < 0.85
        return {"pct_spent": pct, "warning": warning, "throttle": throttle, "stop": stop}

    def process_request(self , query):
        status = self.get_status()
        cached = self.cache.get(query)
        if cached:
            return {"status": "success", "response": cached["response"], "source": "cache"}
        if status["stop"]:
            if not self.stop_triggered:
                print({"status": "rejected", "reason": f"Circuit breaker active (95% budget limit reached)"})
                self.stop_triggered = True
            return {"status": "rejected", "reason": f"Circuit breaker active (95% budget limit reached)"}
        
        # 1. Warning Alert check
        if status["warning"] and not self.warning_triggered:
            print(f"  ⚠️ ALERT [WARNING]: Budget has exceeded 70% (Spent:${self.cost_tracker.total_cost():.4f})")
            self.warning_triggered = True

        # 2. Model Selection (Independent block)
        if status["throttle"]:
            if not self.throttle_triggered:
                print(f"  ⚡ ALERT [THROTTLE]: Budget has exceeded 85%! Switching to gpt-4o-mini.")
                self.throttle_triggered = True
            model = "gpt-4o-mini"
        else:
            model = self.model.route(query)["model"]
        
        sim_res = simulate_llm_call(model, query)
        self.cost_tracker.log_call(model, sim_res["input_tokens"], sim_res["output_tokens"], latency_ms=sim_res["latency_ms"])
        self.cache.put(query, sim_res["response"])
        return {"status": "success", "response": sim_res["response"], "source": model}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    if model not in MODEL_PRICING:
        return {"error": f"Unknown model: {model}"}
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }


class ExactCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, model, messages, temperature):
        key_data = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            self.misses += 1
            return None
        key = self._hash(model, messages, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                entry["access_count"] += 1
                return entry["response"]
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, response):
        if temperature > 0:
            return
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        key = self._hash(model, messages, temperature)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        }

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.cache),
        }


def simple_embed(text):
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    norm = math.sqrt(sum(v * v for v in vocab.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vocab.items()}


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    return dot

class SemanticCacheLRU:
    # modified
    def __init__(self, similarity_threshold=0.85, max_size=10, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            best_match["last_accessed"] = time.time()
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            oldest = min(self.entries, key=lambda e: e["last_accessed"])
            self.entries.remove(oldest)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
            "last_accessed": time.time()
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }
    
class SemanticCacheFIFO:
    # original
    def __init__(self, similarity_threshold=0.85, max_size=10, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            # oldest = min(self.entries, key=lambda e: e["last_accessed"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }

class TieredSemanticCache:
    def __init__(self, similarity_threshold=0.85, max_size=500, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.high_hits = 0
        self.medium_hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match and best_sim >= 0.98:
            self.high_hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "confidence": "high","similarity": round(best_sim, 4)}
        elif best_match and best_sim >= 0.90:
            self.medium_hits += 1
            best_match["access_count"] += 1
            disclaimer_response = f"Based on a similar previous question:{best_match['response']}"
            return {"response": disclaimer_response, "confidence": "medium","similarity": round(best_sim, 4)}
        else:
            self.misses += 1
            return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.high_hits + self.medium_hits + self.misses
        return {
            "high_hits": self.high_hits,
            "medium_hits": self.medium_hits,
            "misses": self.misses,
            "hit_rate": round((self.high_hits + self.medium_hits) / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }

class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = {}
        self.tiers = {
            "free": {"capacity": 50_000, "refill_rate": 500, "max_requests_per_min": 10},
            "pro": {"capacity": 500_000, "refill_rate": 5_000, "max_requests_per_min": 60},
            "enterprise": {"capacity": 5_000_000, "refill_rate": 50_000, "max_requests_per_min": 300},
        }

    def _get_bucket(self, user_id, tier="free"):
        if user_id not in self.buckets:
            tier_config = self.tiers.get(tier, self.tiers["free"])
            self.buckets[user_id] = {
                "tokens": tier_config["capacity"],
                "capacity": tier_config["capacity"],
                "refill_rate": tier_config["refill_rate"],
                "last_refill": time.time(),
                "request_timestamps": [],
                "max_rpm": tier_config["max_requests_per_min"],
                "tier": tier,
                "total_tokens_used": 0,
            }
        return self.buckets[user_id]

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * bucket["refill_rate"])
        if refill > 0:
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill)
            bucket["last_refill"] = now

    def check(self, user_id, tokens_needed, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        self._refill(bucket)
        now = time.time()
        bucket["request_timestamps"] = [t for t in bucket["request_timestamps"] if now - t < 60]
        if len(bucket["request_timestamps"]) >= bucket["max_rpm"]:
            return {"allowed": False, "reason": "rate_limit", "retry_after_seconds": 60 - (now - bucket["request_timestamps"][0])}
        if bucket["tokens"] < tokens_needed:
            deficit = tokens_needed - bucket["tokens"]
            wait = deficit / bucket["refill_rate"]
            return {"allowed": False, "reason": "token_limit", "tokens_available": bucket["tokens"], "retry_after_seconds": round(wait, 1)}
        return {"allowed": True, "tokens_available": bucket["tokens"]}

    def consume(self, user_id, tokens_used, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        bucket["tokens"] -= tokens_used
        bucket["request_timestamps"].append(time.time())
        bucket["total_tokens_used"] += tokens_used

    def get_usage(self, user_id):
        if user_id not in self.buckets:
            return {"error": "User not found"}
        b = self.buckets[user_id]
        return {
            "user_id": user_id,
            "tier": b["tier"],
            "tokens_remaining": b["tokens"],
            "capacity": b["capacity"],
            "total_tokens_used": b["total_tokens_used"],
            "utilization": round(b["total_tokens_used"] / b["capacity"], 4) if b["capacity"] else 0,
        }

class EmbeddingRouter:
    def __init__(self , training_data):
        self.training_entries = []
        for query , label in training_data:
            self.training_entries.append({
                "query": query,
                "label": label,
                "embedding": simple_embed(query)
            })
    
    def classify(self , query):
        query_emb = simple_embed(query)
        best_label = "simple"
        best_sim = -1.0

        for entry in self.training_entries:
            sim = cosine_similarity(query_emb , entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_label = entry["label"]

        return best_label

    def route(self , query , tier="pro"):
        complexity = self.classify(query)
        routing_table = {
            "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini","enterprise": "gpt-4o-mini"},
            "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4","enterprise": "claude-sonnet-4"},
            "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o","enterprise": "claude-opus-4"},
        }
        model = routing_table[complexity].get(tier , "gpt-4o-mini")
        return {"query" : query , "model" : model , "complexity" : complexity}

class CostTracker:
    def __init__(self, monthly_budget=1000.0):
        self.logs = []
        self.monthly_budget = monthly_budget
        self.alerts = []

    def log_call(self, model, input_tokens, output_tokens, cached_input_tokens=0, latency_ms=0, user_id="anonymous", cache_status="miss"):
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": latency_ms,
            "cost": cost["total_cost"],
            "user_id": user_id,
            "cache_status": cache_status,
        }
        self.logs.append(entry)
        self._check_budget()
        return entry

    def _check_budget(self):
        total = self.total_cost()
        pct = total / self.monthly_budget if self.monthly_budget > 0 else 0
        if pct >= 0.95 and not any(a["level"] == "stop" for a in self.alerts):
            self.alerts.append({"level": "stop", "message": f"Budget 95% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.85 and not any(a["level"] == "throttle" for a in self.alerts):
            self.alerts.append({"level": "throttle", "message": f"Budget 85% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.70 and not any(a["level"] == "warning" for a in self.alerts):
            self.alerts.append({"level": "warning", "message": f"Budget 70% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})

    def total_cost(self):
        return round(sum(e["cost"] for e in self.logs), 6)

    def cost_by_model(self):
        by_model = {}
        for e in self.logs:
            m = e["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["calls"] += 1
            by_model[m]["cost"] = round(by_model[m]["cost"] + e["cost"], 6)
            by_model[m]["input_tokens"] += e["input_tokens"]
            by_model[m]["output_tokens"] += e["output_tokens"]
        return by_model

    def cache_savings(self):
        cache_hits = [e for e in self.logs if e["cache_status"] == "hit"]
        if not cache_hits:
            return {"saved": 0, "cache_hits": 0}
        saved = 0
        for e in cache_hits:
            full_cost = calculate_cost(e["model"], e["input_tokens"], e["output_tokens"])
            saved += full_cost["total_cost"]
        return {"saved": round(saved, 4), "cache_hits": len(cache_hits)}
    
    def project_monthly_cost(self):
        now = time.time()
        seven_days_sec = 7 * 86400
        
        recent_logs = [e for e in self.logs if now - e["timestamp"] <= seven_days_sec]
        if not recent_logs:
            return {"projected_monthly" : 0.0 , "alert" : False , "savings" : 0}
        
        weekday_costs = []
        weekend_costs = []

        for e in recent_logs:
            day_of_week = datetime.fromtimestamp(e["timestamp"]).weekday()
            if day_of_week < 5:
                weekday_costs.append(e["cost"])
            else:
                weekend_costs.append(e["cost"])

        avg_weekday = sum(weekday_costs) / len(weekday_costs) if weekday_costs else 0
        avg_weekend = sum(weekend_costs) / len(weekend_costs) if weekend_costs else avg_weekday

        projected_monthly = round((avg_weekday * 22) + (avg_weekend * 8), 2)
        
        budget_threshold = self.monthly_budget * 1.20
        alert_triggered = projected_monthly > budget_threshold
        return {
            "projected_monthly_cost": projected_monthly,
            "monthly_budget": self.monthly_budget,
            "budget_threshold_120": round(budget_threshold, 2),
            "alert_triggered": alert_triggered,
            "status": "DANGER: Exceeds 120% budget!" if alert_triggered else "OK"
        }

    def summary(self):
        if not self.logs:
            return {"total_calls": 0, "total_cost": 0}
        total_latency = sum(e["latency_ms"] for e in self.logs)
        cache_hits = sum(1 for e in self.logs if e["cache_status"] == "hit")
        return {
            "total_calls": len(self.logs),
            "total_cost": self.total_cost(),
            "avg_cost_per_call": round(self.total_cost() / len(self.logs), 6),
            "avg_latency_ms": round(total_latency / len(self.logs), 1),
            "cache_hit_rate": round(cache_hits / len(self.logs), 4),
            "cost_by_model": self.cost_by_model(),
            "cache_savings": self.cache_savings(),
            "budget_remaining": round(self.monthly_budget - self.total_cost(), 2),
            "budget_utilization": round(self.total_cost() / self.monthly_budget, 4) if self.monthly_budget > 0 else 0,
            "alerts": self.alerts,
        }


SIMPLE_KEYWORDS = ["what time", "hours", "address", "phone", "price", "return policy", "hello", "hi", "thanks", "yes", "no"]
COMPLEX_KEYWORDS = ["analyze", "compare", "explain why", "write code", "debug", "architect", "design", "trade-off", "evaluate"]


def classify_complexity(query):
    q = query.lower()
    if len(q.split()) <= 5 or any(kw in q for kw in SIMPLE_KEYWORDS):
        return "simple"
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "medium"


def route_model(query, tier="pro"):
    complexity = classify_complexity(query)
    routing_table = {
        "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
        "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
        "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
    }
    model = routing_table[complexity].get(tier, "gpt-4o-mini")
    return {"query": query, "complexity": complexity, "model": model, "tier": tier}


def simulate_llm_call(model, query):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku-3.5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-4", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-4", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCacheLRU(similarity_threshold=0.75, max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-4o-mini", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Exercise 1: 50-Query Benchmark (FIFO vs LRU) ---")
    fifo_cache = SemanticCacheFIFO(similarity_threshold=0.85, max_size=3)
    lru_cache = SemanticCacheLRU(similarity_threshold=0.85, max_size=3)

    core_queries = [
        "What is the return policy?",
        "What are your store hours?",
        "How do I track my order?"
    ]

    for i in range(50):
        if i % 4 != 0:
            query = core_queries[i % len(core_queries)]
        else:
            query = f"Can I get help with issue item #{i}?"

        res_fifo = fifo_cache.get(query)
        if not res_fifo:
            fifo_cache.put(query, f"Response to {query}")

        res_lru = lru_cache.get(query)
        if not res_lru:
            lru_cache.put(query, f"Response to {query}")

    print(f"  FIFO Cache Stats (max_size=3): {fifo_cache.stats()}")
    print(f"  LRU Cache Stats (max_size=3):  {lru_cache.stats()}")


    print("\n--- Exercise 2: Cost Projection Demo ---")
    test_tracker = CostTracker(monthly_budget=10.00) # $10 budget

    # Simulate 50 calls (which will spend ~$0.30-$0.40 in real-time)
    for _ in range(50):
        test_tracker.log_call("gpt-4o", 1000, 500)

    projection = test_tracker.project_monthly_cost()
    print(f"  Projected Monthly Cost: ${projection['projected_monthly_cost']}")
    print(f"  Monthly Budget: ${projection['monthly_budget']}")
    print(f"  Alert Triggered: {projection['alert_triggered']} ({projection['status']})")

    print("\n--- Exercise 3: Tiered Semantic Cache Demo ---")
    tiered_cache = TieredSemanticCache()
    tiered_cache.put("What is the return policy?", "Items can be returned within 30 days.")

    # 1. Exact match -> High confidence
    res_high = tiered_cache.get("What is the return policy?")
    print(f"  Exact match query -> Confidence: {res_high['confidence']}")

    # 2. Paraphrased match -> Medium confidence (or High depending on simple_embed)
    res_med = tiered_cache.get("What is our return policy?")
    if res_med:
        print(f"  Paraphrased query -> Confidence: {res_med['confidence']}")
        print(f"  Response text: {res_med['response']}")

    print(f"  Tiered Stats: {tiered_cache.stats()}")

    print("\n--- Exercise 4: Embedding Router Demo ---")
    training_data = [
        # Simple queries
        ("What time do you close?", "simple"),
        ("What is your phone number?", "simple"),
        ("Where is the store located?", "simple"),
        ("What are your business hours?", "simple"),
        ("Hello how are you?", "simple"),
        
        # Medium queries
        ("Summarize this document for me", "medium"),
        ("Compare product A and product B", "medium"),
        ("Explain the difference between TCP and UDP", "medium"),
        ("What are the key takeaways from the report?", "medium"),
        
        # Complex queries
        ("Write code for a binary search tree with deletion in Python","complex"),
        ("Analyze the trade-offs between microservices and monolithic architectures", "complex"),
        ("Debug this memory leak in my C++ application", "complex"),
        ("Design a distributed rate-limiting system using Redis", "complex")
    ]

    router = EmbeddingRouter(training_data)

    test_set = [
        ("When does the shop open?", "simple"),
        ("What is your contact number?", "simple"),
        ("Hi there", "simple"),
        ("Summarize the quarterly sales results", "medium"),
        ("Explain microservices architecture", "complex"),
        ("Write a python function for quicksort", "complex")
    ]

    correct = 0
    for q, true_label in test_set:
        res = router.route(q)
        pred_label = res["complexity"]
        if pred_label == true_label:
            correct += 1
        print(f"  Query: '{q}' -> Predicted: {pred_label} (Actual: {true_label})")

    print(f"  Accuracy: {correct}/{len(test_set)} ({correct/len(test_set)*100:.1f}%)")

    print("\n--- Exercise 5: Circuit Breaker 1,000-Request Simulation ---")
    cb = CircuitBreaker(training_data)
    cb.cost_tracker.monthly_budget = 0.05

    blocked_count = 0
    throttled_count = 0
    normal_count = 0

    for i in range(1000):
        # Pass a complex query so we can see throttling degrade it to gpt-4o-mini
        query = f"Unique distinct request topic number {i} code {i*37}"
        res = cb.process_request(query)
        
        if res["status"] == "rejected":
            blocked_count += 1
        elif res.get("source") == "gpt-4o-mini" and cb.get_status()["throttle"]:
            throttled_count += 1
        else:
            normal_count += 1

    print(f"\n  Simulation Results over 1,000 Requests against $1.00 Budget:")
    print(f"  Normal / Standard Calls: {normal_count}")
    print(f"  Throttled Calls (gpt-4o-mini): {throttled_count}")
    print(f"  Blocked Calls (95% Stop): {blocked_count}")
    print(f"  Final Spent: ${cb.cost_tracker.total_cost():.4f} / ${cb.cost_tracker.monthly_budget}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    print("\n  [Before: no caching, single model (gpt-4o)]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = simulate_llm_call("gpt-4o", q)
        tracker_before.log_call("gpt-4o", result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: caching + routing + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCacheLRU(similarity_threshold=0.75)
    tracker_after = CostTracker(monthly_budget=1000.0)

    for q in queries:
        messages = [{"role": "user", "content": q}]
        cached = exact_c.get("gpt-4o", messages, 0.0)
        if cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=5, cache_status="hit")
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=15, cache_status="hit")
            continue
        route = route_model(q)
        result = simulate_llm_call(route["model"], q)
        tracker_after.log_call(route["model"], result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-4o", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-4o-mini", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-4", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-4o", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-4", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)   

if __name__ == "__main__":
    run_demo()
