# phases/11-llm-engineering/15-prompt-caching/code/optimizer.py
"""Prompt layout optimizer.

Given a list of prompt sections marked as stable or volatile, reorders them
to maximize the prefix size that can be cached, and inserts the cache_control
marker on the last stable block.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class PromptSection:
    name: str
    content: str
    stable: bool
    tokens: int

def optimize_layout(sections: List[PromptSection]) -> Tuple[List[PromptSection], int]:
    """Reorders prompt sections to maximize cacheability.
    
    1. Reorders all stable sections to the top, preserving their relative order.
    2. Places all volatile sections at the bottom, preserving their relative order.
    3. Finds the last stable section and returns its new index (the breakpoint).
    
    Returns:
        Tuple of (optimized_sections_list, breakpoint_index)
    """
    stable_blocks = [s for s in sections if s.stable]
    volatile_blocks = [s for s in sections if not s.stable]
    
    optimized = stable_blocks + volatile_blocks
    breakpoint_idx = len(stable_blocks) - 1 if stable_blocks else -1
    
    return optimized, breakpoint_idx

# Sample unoptimized prompt layout
ORIGINAL_LAYOUT = [
    PromptSection(name="system_instructions", content="You are a translation assistant.", stable=True, tokens=1500),
    PromptSection(name="current_time", content="Current time is 2026-07-02T13:00:00Z.", stable=False, tokens=50),
    PromptSection(name="few_shot_examples", content="Input: Bonjour -> Output: Hello...", stable=True, tokens=2000),
    PromptSection(name="user_query", content="Translate: Comment ça va?", stable=False, tokens=100),
]

def main():
    print("Original Layout:")
    for i, sec in enumerate(ORIGINAL_LAYOUT):
        status = "STABLE" if sec.stable else "VOLATILE"
        print(f"  [{i}] {sec.name:<20} ({status:<8}) - {sec.tokens:>4} tokens")
        
    optimized, breakpoint_idx = optimize_layout(ORIGINAL_LAYOUT)
    if optimized is None:
        print("\nTODO: Implement optimize_layout")
        return
        
    print("\nOptimized Layout:")
    for i, sec in enumerate(optimized):
        status = "STABLE" if sec.stable else "VOLATILE"
        marker = " <-- [CACHE BREAKPOINT]" if i == breakpoint_idx else ""
        print(f"  [{i}] {sec.name:<20} ({status:<8}) - {sec.tokens:>4} tokens{marker}")
        
    # Calculate cacheable tokens
    cacheable_tokens = sum(sec.tokens for sec in optimized[:breakpoint_idx + 1])
    print(f"\nCacheable Tokens: {cacheable_tokens} out of {sum(s.tokens for s in ORIGINAL_LAYOUT)}")

if __name__ == "__main__":
    main()
