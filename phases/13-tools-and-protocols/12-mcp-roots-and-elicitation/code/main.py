"""Phase 13 Lesson 12 - MCP roots and elicitation.

Demonstrates:
  - client-declared roots enforced as server boundary
  - elicitation/create for disambiguation when a tool has multiple matches
  - URL-mode elicitation sketched for OAuth-style first-run (experimental)

Fake client stand-in for the user interaction; real SDKs ship a real dialog.

Run: python code/main.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


# ---- client-declared roots ----
ROOTS = [
    {"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"},
    {"uri": "file:///Users/alice/Scratch", "name": "Scratch"},
]

MIN_SDK_VERSION = "2025-11-25"
SDK_VERSION = "2025-10-01"

def uri_in_roots(uri: str) -> bool:
    for r in ROOTS:
        if uri.startswith(r["uri"]):
            return True
    return False


# ---- fake data ----
NOTES = {
    "note-3": {"title": "TPS report 2023", "uri": "file:///Users/alice/Documents/Notes/tps-2023.md"},
    "note-7": {"title": "TPS report 2024", "uri": "file:///Users/alice/Documents/Notes/tps-2024.md"},
    "note-14": {"title": "TPS report 2025", "uri": "file:///Users/alice/Documents/Notes/tps-2025.md"},
    "note-99": {"title": "shopping list", "uri": "file:///Users/alice/Documents/Notes/shopping.md"},
    "note-89": {"title": "games list", "uri": "file:///Users/alice/Documents/Notes/games.md"},
    "note-100": {"title": "outside root", "uri": "file:///tmp/outside.md"},
}

ARCHIVE_NOTES = {}

# handle_id -> uri
OPEN_HANDLES : dict[str , dict] = {}

# ---- elicitation stand-in (fake user answers) ----
FAKE_USER_ANSWERS: dict[str, dict] = {
    "delete_tps": {"action": "accept", "content": {"note_id": "note-14", "confirm": True}},
    "delete_outside": {"action": "decline", "content": {}},
    "archive_games": {"action": "accept", "content": {"note_id": "note-89", "confirm": True}},
    "archive_tps": {"action": "accept", "content": {"note_id": "note-3", "confirm": True}},
    "archive_outside": {"action": "decline", "content": {}}
}


def elicit(key: str, message: str, schema: dict | None = None,
           url: str | None = None) -> dict:
    """Simulates elicitation/create round trip."""
    print(f"  [elicit] message={message!r}")
    if url:
        print(f"  [elicit] url-mode: open {url} in browser (SEP-1036, experimental)")
    if schema:
        print(f"  [elicit] schema: {json.dumps(schema)}")
    resp = FAKE_USER_ANSWERS.get(key, {"action": "cancel", "content": {}})
    print(f"  [elicit] <- {resp}")
    return resp


# ---- tools ----

def tool_notes_delete(args: dict) -> dict:
    title = args["title"]
    matches = [{"id": nid, **n} for nid, n in NOTES.items() if title.lower() in n["title"].lower()]
    if not matches:
        return {"content": [{"type": "text", "text": "no match"}], "isError": True}
    if len(matches) == 1:
        m = matches[0]
        if not uri_in_roots(m["uri"]):
            return {"content": [{"type": "text", "text": f"rejected: {m['uri']} outside roots"}],
                    "isError": True}
        del NOTES[m["id"]]
        return {"content": [{"type": "text", "text": f"deleted {m['id']}"}], "isError": False}
    # disambiguation via elicitation
    schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "enum": [m["id"] for m in matches]},
            "confirm": {"type": "boolean"},
        },
        "required": ["note_id", "confirm"],
    }
    elicit_key = "delete_tps" if title == "TPS report" else "delete_outside"
    resp = elicit(elicit_key,
                  f"Multiple notes match {title!r}. Pick one and confirm.",
                  schema=schema)
    if resp["action"] != "accept" or not resp["content"].get("confirm"):
        return {"content": [{"type": "text", "text": "cancelled by user"}], "isError": False}
    nid = resp["content"]["note_id"]
    if nid not in NOTES:
        return {"content": [{"type": "text", "text": "race: note missing"}], "isError": True}
    if not uri_in_roots(NOTES[nid]["uri"]):
        return {"content": [{"type": "text", "text": "rejected: outside roots"}], "isError": True}
    del NOTES[nid]
    return {"content": [{"type": "text", "text": f"deleted {nid} after user pick"}], "isError": False}

def tool_notes_archive(args:dict) -> dict:
    title = args["title"]
    matches = [{"id": nid, **n} for nid, n in NOTES.items() if title.lower() in n["title"].lower()]
    if not matches:
        return {"content": [{"type": "text", "text": "no match"}], "isError": True}
    if len(matches) == 1:
        m = matches[0]
        if not uri_in_roots(m["uri"]):
            return {"content": [{"type": "text", "text": f"rejected: {m['uri']} outside roots"}],
                    "isError": True}
        # elicitation
        schema = {
        "type": "object",
        "properties": {
            "confirm": {"type": "boolean"},
        },
        "required": ["confirm"],
        }
        resp = elicit("archive_games",
                      f"Do you want to archive {title!r}. (accept/decline).",
                      schema=schema)
        if resp["action"] == "accept":
            ARCHIVE_NOTES[m["id"]] = m
            del NOTES[m["id"]]
            return {"content": [{"type": "text", "text": f"archived {m['id']}"}], "isError": False}
        else:
            return {"content": [{"type": "text", "text": "cancelled by user"}], "isError": False}

    # disambiguation (multiple notes matched)
    schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "enum": [m["id"] for m in matches]},
            "confirm": {"type": "boolean"},
        },
        "required": ["note_id", "confirm"],
    }
    elicit_key = "archive_tps" if title == "TPS report" else "archive_outside"
    resp = elicit(elicit_key,
                  f"Multiple notes match {title!r}. Pick one and confirm.",
                  schema=schema)
    if resp["action"] != "accept" or not resp["content"].get("confirm"):
        return {"content": [{"type": "text", "text": "cancelled by user"}], "isError": False}
    nid = resp["content"]["note_id"]
    if nid not in NOTES:
        return {"content": [{"type": "text", "text": "race: note missing"}], "isError": True}
    if not uri_in_roots(NOTES[nid]["uri"]):
        return {"content": [{"type": "text", "text": "rejected: outside roots"}], "isError": True}
    ARCHIVE_NOTES[nid] = NOTES[nid]
    del NOTES[nid]
    return {"content": [{"type": "text", "text": f"archived {nid} after user pick"}], "isError": False}
    

def tool_notes_setup(args: dict) -> dict:
    if SDK_VERSION >= MIN_SDK_VERSION:
        print("  [drift-risk] using URL-mode for notes setup (experimental, SEP-1036)")
        resp = elicit("setup",
                  "Sign in to your notes provider",
                  url="https://example.com/oauth/authorize?client_id=...")
    else:
        print("  [drift-risk] SDK too old; falling back to form-mode for notes setup")
        resp = elicit("setup",
                  "Sign in to your notes provider",
                  schema={"type": "object", "properties": {"provider": {"type": "string"}}, "required": ["provider"]})    
    if resp["action"] != "accept":
        return {"content": [{"type": "text", "text": "setup cancelled"}], "isError": False}
    return {"content": [{"type": "text", "text": "setup complete"}], "isError": False}


TOOL_EXECUTORS: dict[str, Callable[[dict], dict]] = {
    "notes_delete": tool_notes_delete,
    "notes_setup": tool_notes_setup,
    "notes_archive" : tool_notes_archive
}


def call(name: str, args: dict) -> dict:
    return TOOL_EXECUTORS[name](args)

# recheck_handles_after_root_change
def on_root_change():
    print("  [server] re-reading roots after list_changed notification")
    dropped = []
    for handle_id , handle in list(OPEN_HANDLES.items()):
        if not uri_in_roots(handle["uri"]):
            dropped.append(handle_id)
            del OPEN_HANDLES[handle_id]
    if dropped:
        print(f"  [server] closed handles outside new root set: {dropped}")

def demo() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 12 - ROOTS AND ELICITATION")
    print("=" * 72)

    print("\n--- declared roots ---")
    for r in ROOTS:
        print(f"  {r['uri']:60s} ({r['name']})")

    print("\n--- scenario 1: unambiguous delete inside roots ---")
    r = call("notes_delete", {"title": "shopping"})
    print(f"  result: {r['content'][0]['text']}")

    print("\n--- scenario 2: ambiguous delete, elicitation fires ---")
    r = call("notes_delete", {"title": "TPS report"})
    print(f"  result: {r['content'][0]['text']}")

    print("\n--- scenario 3: target outside roots ---")
    NOTES["note-100"] = {"title": "outside root", "uri": "file:///tmp/outside.md"}
    r = call("notes_delete", {"title": "outside"})
    print(f"  result: {r['content'][0]['text']}")

    print("\n---scenario 4 : unambigous archive inside roots ---")
    r = call("notes_archive" , {"title" : "games list"})
    print(f"  result: {r['content'][0]['text']}")

    print("\n---scenario 5 : ambigous archive, elictiation fires ---")
    r = call("notes_archive" , {"title" : "TPS report"})
    print(f"  result: {r['content'][0]['text']}")

    print("\n---scenario 6 : archive outside roots ---")
    NOTES["note-100"] = {"title" : "outside" , "uri" : "file:///tmp/outside.md"}
    r = call("notes_archive" , {"title" : "outside"})
    print(f"  result: {r['content'][0]['text']}")

    print("\n -- UPDATED NOTES ---")
    print(NOTES)
    print("\n -- ARCHIVED NOTES ---")
    print(ARCHIVE_NOTES)

    # print("\n--- scenario 7 : URL-mode elicitation (experimental) ---")
    # FAKE_USER_ANSWERS["setup"] = {"action": "accept", "content": {"signed": True}}
    # r = call("notes_setup", {})
    # print(f"  result: {r['content'][0]['text']}")

    print("\n--- scenario 7: SDK too old — falls back to form mode ---")
    r = call("notes_setup", {})
    print(f"  result: {r['content'][0]['text']}")

    print("\n--- scenario 8: URL-mode with compatible SDK ---")
    global SDK_VERSION
    SDK_VERSION = "2026-01-01"
    FAKE_USER_ANSWERS["setup"] = {"action": "accept", "content": {"signed": True}}
    r = call("notes_setup", {})
    print(f"  result: {r['content'][0]['text']}")
    SDK_VERSION = "2025-10-01"  # restore

    OPEN_HANDLES["scratch-log"] = {"uri": "file:///Users/alice/Scratch/temp.log"}
    print(f"\n  Current OPEN_HANDLES: {list(OPEN_HANDLES.keys())}")

    print("\n--- roots/list_changed simulation ---")
    ROOTS.pop()
    on_root_change()
    print(f"  roots after user removed Scratch: {[r['uri'] for r in ROOTS]}")
    print(f"  server should drop any open handles outside the new set")

if __name__ == "__main__":
    demo()
