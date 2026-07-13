from mcp.server.fastmcp import FastMCP , Context
from typing import Optional
# from mcp import types
import json
import uuid

app = FastMCP("notes")

NOTES: dict[str, dict] = {
    "note-1": {"title": "MCP overview", "body": "Primitives, lifecycle, JSON-RPC.", "tag": "mcp"},
    "note-2": {"title": "Function calling", "body": "Provider shapes diff by envelope.", "tag": "api"},
    "note-3": {"title": "Tool schemas", "body": "Atomic beats monolithic.", "tag": "design"},
}

@app.tool(
    title="List Notes",
)
def notes_list(tag: Optional[str] = None) -> list[dict]:
    """Use when the user wants all notes or a filtered list by tag. Do not use to read a note body."""
    items = []
    for nid, note in NOTES.items():
        if tag and note.get("tag") != tag:
            continue
        items.append({"id": nid, "title": note["title"], "tag": note.get("tag", "")})
    return [{"type": "text", "text": json.dumps(items)}]

@app.tool(
    title="Search Notes",
)
def notes_search(query: str, limit: Optional[int] = 10) -> list[dict]:
    """Use when the user searches notes by content keywords. Do not use for tag filters."""
    q = query.lower()
    hits = []
    for nid, n in NOTES.items():
        if q in n["title"].lower() or q in n["body"].lower():
            hits.append({"id": nid, "title": n["title"]})
    return [{"type": "text", "text": json.dumps(hits[:limit])}]

@app.tool(
    title="Create Notes",
)
async def notes_create(ctx : Context , title: str, body: str, tag: Optional[str] = "") -> list[dict]:
    """Use when the user writes a new note. Do not use to edit existing ones."""
    nid = f"note-{uuid.uuid4().hex[:6]}"
    
    NOTES[nid] = {"title": title, "body": body, "tag": tag}
    await ctx.send_resource_updated(f"notes://{nid}")
    return [
        {"type": "text", "text": f"Created {nid}"},
        {"type": "resource", "resource": {"uri": f"notes://{nid}", "text": body}},
    ]


@app.tool(
    title="Delete Notes",
    annotations={"destructiveHint": True}
)
async def notes_delete(ctx : Context , nid: str) -> list[dict]:
    """Use when the user deletes a note. Do not use to delete existing notes."""
    if nid not in NOTES:
        raise ValueError(f"not found: {nid}")
    snapshot = NOTES[nid]
    del NOTES[nid]
    await ctx.send_resource_updated(f"notes://{nid}")
    return [
        {"type": "text", "text": f"Deleted {nid}: {snapshot['title']}"},
        {"type": "resource", "resource": {"uri": f"notes://{nid}",
                                        "text": f"# {snapshot['title']}\n\n{snapshot['body']}\n\ntag: {snapshot.get('tag', '')}"}},
    ]

@app.resource("notes://{nid}")
def get_note(nid: str) -> str:
    if nid not in NOTES:
        raise ValueError(f"not found: {nid}")
    n = NOTES[nid]
    uri = f"notes://{nid}"  
    return f"# {n['title']}\n\n{n['body']}\n\ntag: {n.get('tag', '')}"

@app.prompt()
def review_note(nid: str) -> dict:
    body = get_note(nid)
    return {
        "description": "Review the note and propose concrete improvements.",
        "messages": [
            {"role": "user", "content": {"type": "text",
                "text": f"Review this note and propose improvements:\n\n{body}"}}
        ],
    }

if __name__=="__main__":
    app.run(transport="stdio")