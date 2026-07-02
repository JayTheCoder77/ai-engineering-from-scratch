# phases/11-llm-engineering/15-prompt-caching/code/tests/test_main.py
"""Unit tests for prompt caching simulators and layout optimizer.

Ref: docs/en.md Exercise 1, 2, and 3.
"""

import unittest
import sys
import os

# Adjust path to find harness and optimizer modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from harness import RequestLogEntry, simulate_anthropic, simulate_openai, simulate_gemini
from optimizer import PromptSection, optimize_layout

class TestPromptCaching(unittest.TestCase):
    def test_optimize_layout_reordering(self):
        """Test that stable blocks are grouped first and relative order is preserved."""
        sections = [
            PromptSection("sys", "content", True, 100),
            PromptSection("time", "content", False, 10),
            PromptSection("shots", "content", True, 200),
            PromptSection("query", "content", False, 20),
        ]
        optimized, bp = optimize_layout(sections)
        self.assertEqual(len(optimized), 4)
        self.assertEqual(optimized[0].name, "sys")
        self.assertEqual(optimized[1].name, "shots")
        self.assertEqual(optimized[2].name, "time")
        self.assertEqual(optimized[3].name, "query")
        self.assertEqual(bp, 1)

    def test_optimize_layout_empty(self):
        """Test optimizer handles empty section list gracefully."""
        optimized, bp = optimize_layout([])
        self.assertEqual(optimized, [])
        self.assertEqual(bp, -1)

    def test_simulate_anthropic_hit(self):
        """Test Anthropic hits when requests are made within TTL."""
        log = [
            RequestLogEntry(0.0, "key1", 5000, 100),
            RequestLogEntry(100.0, "key1", 5000, 100),
        ]
        res = simulate_anthropic(log, ttl_seconds=300)
        self.assertEqual(res.writes, 1)
        self.assertEqual(res.reads, 1)
        self.assertEqual(res.misses, 0)

    def test_simulate_openai_hit(self):
        """Test OpenAI automatic hits within rolling 1-hour window."""
        log = [
            RequestLogEntry(0.0, "key1", 5000, 100),
            RequestLogEntry(100.0, "key1", 5000, 100),
        ]
        res = simulate_openai(log, ttl_seconds=3600)
        self.assertEqual(res.writes, 1)
        self.assertEqual(res.reads, 1)

    def test_simulate_gemini_ttl_expiry(self):
        """Test Gemini explicit cache does not refresh on read and expires on time."""
        # TTL is 3600 seconds. Cache created at 0.0.
        # Request at 3000.0 is a hit (within 3600).
        # Request at 4000.0 is expired (4000 >= 3600), so it writes again.
        log = [
            RequestLogEntry(0.0, "key1", 5000, 100),
            RequestLogEntry(3000.0, "key1", 5000, 100),
            RequestLogEntry(4000.0, "key1", 5000, 100),
        ]
        res = simulate_gemini(log, ttl_seconds=3600)
        self.assertEqual(res.writes, 2)  # Initial write + Re-creation write
        self.assertEqual(res.reads, 1)
        self.assertEqual(res.misses, 1)

if __name__ == "__main__":
    unittest.main()
