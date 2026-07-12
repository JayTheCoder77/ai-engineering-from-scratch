
# # ❌ Bad (STDIO)
# print("Processing request")

# # ✅ Good (STDIO)
# print("Processing request", file=sys.stderr)

# # ✅ Good (STDIO)
# logging.info("Processing request")

import sys
import logging
from typing import Annotated
import json
# import httpx
from mcp.server.fastmcp import FastMCP
from main import lint_registry , report

# Initialize FastMCP server
mcp = FastMCP("notes")

notes_store: dict[str , dict] = {}

_next_id = 0
def _gen_id() -> str:
    global _next_id
    _next_id += 1
    return f"note-{_next_id:04d}"


@mcp.tool()
async def notes_list(tag: Annotated[str, "Tag to filter notes by"] = None) -> str:
    """
    Use when the user needs to list all notes or a filtered list by tag. 
    Do not use for reading a single note's full body; use notes_get instead.
    """
    if tag:
        return json.dumps([value for key, value in notes_store.items() if value.get("tag") == tag])
    return json.dumps(list(notes_store.values()))

@mcp.tool()
async def notes_search(query: Annotated[str, "Free-text query to search in note contents"]) -> str:
    """
    Use when the user wants to search for notes containing a specific text query.
    Do not use for filtering by tag; use notes_list instead.
    """
    results = []
    for note in notes_store.values():
        if query.lower() in note.get("title", "").lower() or query.lower() in note.get("body", "").lower():
            results.append(note)
    return json.dumps(results)
    

@mcp.tool()
async def notes_create(title: Annotated[str, "Title of the note"], body: Annotated[str, "Body of the note"], tag: Annotated[str, "Tag to categorize the note"] = None) -> str:
    """
    Use when the user asks to write a new note with a title and body. 
    Do not use for editing existing notes; use notes_update instead.
    """
    note_id = _gen_id() 
    note = {
        "note_id" : note_id,
        "title" : title,
        "body" : body
    }
    if tag:
        note["tag"] = tag
    notes_store[note_id] = note
    
    return json.dumps({"note_id" : note_id})

@mcp.tool()
async def notes_update(note_id: Annotated[str, "ID of the note to update"], title: Annotated[str, "New title of the note"] = None, body: Annotated[str, "New body of the note"] = None, tag: Annotated[str, "New tag for the note"] = None) -> str:
    """
    Use when the user asks to edit an existing note by providing its ID and new content.
    Do not use for creating new notes; use notes_create instead.
    """
    # fetch note by its id
    
    if note_id in notes_store.keys():
        if title:
            notes_store[note_id]["title"] = title
        if body:
            notes_store[note_id]["body"] = body
        if tag:
            notes_store[note_id]["tag"] = tag
        return json.dumps(notes_store[note_id])
    
    logging.error("note not found")
    return json.dumps({"error" : "note not found"})

@mcp.tool()
async def notes_delete(note_id: Annotated[str, "ID of the note to delete"]) -> str:
    """
    Use when the user asks to delete one or more notes by their IDs.
    Do not use for listing or searching notes; use notes_list or notes_search instead.
    """
    try:
        del notes_store[note_id]
    except Exception as e:
        logging.error(e)
        return json.dumps({"error" : "note not found"})
    return json.dumps({"status" : "deleted"})

@mcp.prompt()
def summarize(note_id : Annotated[str , "ID of the note to be summarized"]) -> str:
    """
    Use when the user asks to summarize a note by providing its ID.
    Do not use for creating or updating notes; use notes_create or notes_update instead.
    """
    if note_id in notes_store.keys():
        note = notes_store[note_id]
        prompt = f"""
            Summarize the following note. Keep the summary concise and accurate, capture the main points and insights.
            Note:
            Title: {note['title']}
            Body: {note['body']}
            Tag: {note.get('tag', 'No tag')}
        """
        return prompt
    else:
        logging.error("note not found")
        return json.dumps({"error" : "note not found"})

NOTES_REGISTRY = [
    {
        "name": "notes_list",
        "description": (
            "Use when the user needs to list all notes or a filtered list by tag. "
            "Do not use for reading a single note's full body; use notes_get instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Optional tag filter"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "notes_search",
        "description": (
            "Use when the user wants to search for notes containing a specific text query. "
            "Do not use for filtering by tag; use notes_list instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text query"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "notes_create",
        "description": (
            "Use when the user asks to write a new note with a title and body. "
            "Do not use for editing existing notes; use notes_update instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title"},
                "body": {"type": "string", "description": "Note body"},
                "tag": {"type": "string", "description": "Optional tag"},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "notes_update",
        "description": (
            "Use when the user asks to edit an existing note by providing its ID and new content. "
            "Do not use for creating new notes; use notes_create instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "ID of the note to update"},
                "title": {"type": "string", "description": "New title"},
                "body": {"type": "string", "description": "New body"},
                "tag": {"type": "string", "description": "New tag"},
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "notes_delete",
        "description": (
            "Use when the user asks to delete one or more notes by their IDs. "
            "Do not use for listing or searching notes; use notes_list or notes_search instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "ID of the note to delete"},
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
    },
]

report("NOTES_REGISTRY", NOTES_REGISTRY)

# def main():
#     # mcp.run(transport="stdio")

# if __name__ == "__main__":
#     main()



# .github/curriculum.yml - ex4


  # tool-schema-lint:
  #   name: Tool schema lint (block findings gate)
  #   runs-on: ubuntu-latest
  #   if: github.event_name == 'pull_request'
  #   steps:
  #     - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5  # v4
  #       with:
  #         persist-credentials: false
  #     - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065  # v5
  #       with:
  #         python-version: "3.12"
  #     - name: run tool schema linter on lesson registries
  #       id: lint
  #       run: |
  #         cd phases/13-tools-and-protocols/05-tool-schema-design/code
  #         output=$(python3 main.py)
  #         echo "$output"
  #         # Count block severity findings
  #         block_count=$(echo "$output" | grep -c '\[block\]' || true)
  #         echo "block_count=$block_count" >> "$GITHUB_OUTPUT"
  #         if [ "$block_count" -gt 0 ]; then
  #           echo "FAILURE: $block_count block-level finding(s) detected"
  #           exit 1
  #         fi
  #         echo "PASS: no block-level findings"