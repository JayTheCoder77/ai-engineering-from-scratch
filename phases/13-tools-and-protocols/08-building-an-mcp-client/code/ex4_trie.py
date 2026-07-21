#!/usr/bin/env python3
"""Exercise 4 - scaling to 100 concurrent servers: the trie dispatch.

The simple dict[str, MergedTool] is fine for 3 servers but can't answer
prefix questions efficiently at 100. A trie (prefix tree) stores each
namespaced tool as a PATH of nodes (e.g. ["github", "search"]), giving
O(depth) lookup AND cheap "list all tools under server X". We also track
a per-server tool-count metric for scheduling/health.

Your job: implement insert() and search() in ToolTrie.

Run: python code/ex4_trie.py
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    value: object = None  # leaf payload: the MergedTool / routing target


class ToolTrie:
    def __init__(self) -> None:
        self.root = TrieNode()
        self.tool_count: dict[str, int] = {}  # server -> #tools (the metric)

    def insert(self, path: list[str], server: str, value: object) -> None:
        # === TODO(human): insert a namespaced tool =====================
        # `path` is a list of segments, e.g. ["files", "search"] or
        # ["search"] (for a bare, un-prefixed name).
        #   1. Start at self.root.
        #   2. For each segment, walk into (creating if missing) the child
        #      node:  node = node.children.setdefault(seg, TrieNode())
        #   3. At the final node, set node.value = value.
        #   4. Bump the per-server metric: self.tool_count[server] += 1
        #      (use dict.setdefault(server, 0) first, or initialize).
        # ==============================================================
        node = self.root
        for segment in path:
            node = node.children.setdefault(segment , TrieNode())
        node.value = value
        self.tool_count[server] = self.tool_count.get(server , 0) + 1

    def search(self, path: list[str]) -> object | None:
        # === TODO(human): follow `path` from self.root ================
        # Walk child-by-child. If any segment is missing, return None.
        # If you reach the end, return node.value (may be None if the
        # path is a prefix with no exact tool at the leaf).
        # ==============================================================
        node = self.root
        for seg in path:
            if seg not in node.children:
                return None
            node = node.children[seg]
        return node.value

    def collect(self, prefix: list[str]) -> list[str]:
        """Bonus: return all tool names under a server prefix (e.g. ['github'])."""
        node = self.root
        for seg in prefix:
            if seg not in node.children:
                return []
            node = node.children[seg]
        out: list[str] = []

        def walk(n: TrieNode, acc: str) -> None:
            if n.value is not None:
                out.append(acc)
            for seg, child in n.children.items():
                walk(child, f"{acc}/{seg}" if acc else seg)

        walk(node, "")
        return out


# --- fixture: same 3 servers, all declaring `search` -----------------
SERVERS = {
    "notes":  [("search", "Search notes"), ("create", "Create a note")],
    "files":  [("read", "Read a file"), ("search", "Search files")],
    "github": [("list_issues", "List issues"), ("open_pr", "Open a PR"),
               ("search", "Search repo")],
}


def main() -> None:
    trie = ToolTrie()
    for server, tools in SERVERS.items():
        for local, desc in tools:
            # prefix-on-collision: bare name stays, duplicates get server/ prefix
            canonical = local if local != "search" or server == "notes" else f"{server}/{local}"
            path = canonical.split("/")
            trie.insert(path, server, f"{server}:{local}")

    print("lookup files/search ->", trie.search(["files", "search"]))
    print("lookup search       ->", trie.search(["search"]))
    print("lookup github/search->", trie.search(["github", "search"]))
    print("lookup bogus        ->", trie.search(["nope"]))
    print("\ntools under 'github' (prefix query):", trie.collect(["github"]))
    print("tool-count metric   ->", trie.tool_count)


if __name__ == "__main__":
    main()
