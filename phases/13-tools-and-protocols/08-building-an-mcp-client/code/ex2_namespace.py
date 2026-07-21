#!/usr/bin/env python3
"""Exercise 2 - namespace merge + collision policies + routing.

Three servers, all of which declare `search` (the collision). Your job is
to implement merge() so colliding names resolve per `policy`, producing a
dispatch table (self.registry) that call() can route through.

Run: python code/ex2_namespace.py
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    name: str
    tools: list[dict] = field(default_factory=list)


@dataclass
class MergedTool:
    canonical_name: str
    server_name: str
    local_name: str
    description: str


# Fixture: three servers. Every one of them exposes `search`.
SESSIONS = [
    Session(name="notes", tools=[
        {"name": "search", "description": "Search notes"},
        {"name": "create", "description": "Create a note"},
    ]),
    Session(name="files", tools=[
        {"name": "read", "description": "Read a file"},
        {"name": "search", "description": "Search files"},
    ]),
    Session(name="github", tools=[
        {"name": "list_issues", "description": "List issues"},
        {"name": "open_pr", "description": "Open a PR"},
        {"name": "search", "description": "Search repo"},
    ]),
]


class MultiServerClient:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {s.name: s for s in SESSIONS}
        self.registry: dict[str, MergedTool] = {}
        self.rejected: list[str] = []  # filled only under the "reject" policy

    def merge(self, policy: str) -> None:
        # === TODO(human): build self.registry from every session's tools ===
        # Iterate self.sessions in order. For each tool:
        #   * local  = tool["name"]
        #   * canonical = local, UNLESS a collision forces a rename
        # A "collision" = canonical is ALREADY a key in self.registry.
        #
        # Resolve per policy when a collision happens:
        #   "prefix-on-collision": keep the FIRST server's bare name; later
        #       duplicates become f"{server_name}/{local}"  (e.g. files/search).
        #   "first-come": the first server's tool wins; later duplicates are
        #       silently DROPPED (do not add to registry).
        #   "reject": on collision, refuse - drop the colliding tool AND append
        #       its canonical name to self.rejected so main() can warn the user.
        #
        # When there is NO collision, just register canonical = local.
        # Register via:
        #   self.registry[canonical] = MergedTool(canonical, s.name, local,
        #                                         tool["description"])
        for session in self.sessions.values():
            for tool in session.tools:
                local = tool["name"]
                if local in self.registry.keys():
                    # collision
                    if policy == "prefix-on-collision":
                        canonical = f"{session.name}/{local}"
                    elif policy == "first-come":
                        continue
                    elif policy == "reject":
                        self.rejected.append(local)
                        continue
                else:
                    canonical = local
                    
                self.registry[canonical] = MergedTool(canonical, session.name, local,
                                                    tool["description"])

    def call(self, canonical_name: str, args: dict) -> str:
        if canonical_name not in self.registry:
            return f"unknown tool {canonical_name}"
        mt = self.registry[canonical_name]
        return f"[{mt.server_name}] {mt.local_name} ran"


def main() -> None:
    for policy in ("prefix-on-collision", "first-come", "reject"):
        c = MultiServerClient()
        c.merge(policy)
        print(f"\n=== policy={policy!r}  ({len(c.registry)} tools) ===")
        for name, mt in c.registry.items():
            print(f"  {name:18s} -> {mt.server_name}:{mt.local_name}")
        if c.rejected:
            print(f"  REJECTED (warn user): {c.rejected}")
        # routing sanity check
        target = "files/search" if policy == "prefix-on-collision" else "create"
        print(f"  route {target!r} -> {c.call(target, {})}")


if __name__ == "__main__":
    main()
