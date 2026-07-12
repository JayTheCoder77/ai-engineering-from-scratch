"""Phase 13 Lesson 05 - tool schema design linter.

Audits a tool registry against design rules from the lesson:
  - names: snake_case, verb-noun, no arguments, no tense markers
  - descriptions: Use-when pattern, length bounds, no injection keywords
  - schemas: typed properties, required list, enum on closed sets
  - shape: atomic vs monolithic (flag `action: str` if enum size > 3)

Plus a Composio-inspired rule: string parameters with format-related
descriptions should include an inline example (e.g., "e.g. 'hello'").

Run on GOOD_REGISTRY (passes) and BAD_REGISTRY (fails on every rule).
Stdlib only.

Run: python code/main.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
EXAMPLE_HINT = re.compile(r"\be\.g\.", re.IGNORECASE)
FORMAT_KEYWORDS = ("format", "syntax", "pattern", "query", "filter", "code",
                    "expression", "path", "url", "email")
INJECTION_PATTERNS = [
    r"<system>",
    r"ignore (previous|all) (instructions|prompts)",
    r"bit\.ly|tinyurl",
    r"you must now",
]
TENSE_MARKERS = ("_was_", "_will_", "_been_", "_yesterday", "_tomorrow")


@dataclass
class Finding:
    severity: str   # block / warn / nit
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity:5s}] {self.path}: {self.message}"


def lint_name(name: str) -> list[Finding]:
    f: list[Finding] = []
    if not SNAKE_CASE.match(name):
        f.append(Finding("block", name, "name must be snake_case"))
    if any(m in name for m in TENSE_MARKERS):
        f.append(Finding("warn", name, "name includes tense marker"))
    if re.search(r"_(in|for|at|by)_\w+$", name):
        f.append(Finding("warn", name, "argument appears embedded in name"))
    if "_" not in name and len(name) > 12:
        f.append(Finding("nit", name, "long single-word name"))
    return f


def lint_description(desc: str, tool_name: str) -> list[Finding]:
    f: list[Finding] = []
    if len(desc) < 40:
        f.append(Finding("block", tool_name, f"description under 40 chars: {len(desc)}"))
    if len(desc) > 1024:
        f.append(Finding("block", tool_name, f"description over 1024 chars: {len(desc)}"))
    low = desc.lower()
    if "use when" not in low:
        f.append(Finding("warn", tool_name, "description missing 'Use when' pattern"))
    if "do not use" not in low:
        f.append(Finding("warn", tool_name, "description missing 'Do not use for' disambiguation"))
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, low):
            f.append(Finding("block", tool_name,
                             f"possible tool-poisoning pattern: {pattern!r}"))
    return f


def lint_schema(schema: dict, tool_name: str) -> list[Finding]:
    f: list[Finding] = []
    if schema.get("type") != "object":
        f.append(Finding("block", tool_name, "schema root must be object"))
        return f
    if "required" not in schema:
        f.append(Finding("warn", tool_name, "schema missing 'required' list"))
    props = schema.get("properties", {})
    for key, sub in props.items():
        path = f"{tool_name}.{key}"
        if "type" not in sub:
            f.append(Finding("block", path, "field has no type"))
        if sub.get("type") == "string" and "description" not in sub:
            if key not in ("id", "uuid"):
                f.append(Finding("nit", path, "string field lacks description"))
        if key == "action" and sub.get("type") == "string":
            values = sub.get("enum", [])
            if len(values) > 3 or not values:
                f.append(Finding("warn", tool_name,
                                 f"monolithic 'action' string (enum len={len(values)}); "
                                 "split into atomic tools"))
        # Composio rule: flag string params with format-related descriptions
        # that lack an inline example.
        if sub.get("type") == "string":
            desc = sub.get("description", "")
            if desc and not EXAMPLE_HINT.search(desc):
                if any(kw in desc.lower() for kw in FORMAT_KEYWORDS):
                    f.append(Finding("nit", path,
                                     "description mentions format/syntax/pattern "
                                     "but lacks an inline example (e.g.)"))
    return f


def lint_tool(tool: dict) -> list[Finding]:
    findings: list[Finding] = []
    name = tool.get("name", "")
    findings.extend(lint_name(name))
    findings.extend(lint_description(tool.get("description", ""), name))
    findings.extend(lint_schema(tool.get("input_schema", {}), name))
    return findings


def lint_registry(registry: list[dict]) -> list[Finding]:
    all_findings: list[Finding] = []
    names = [t["name"] for t in registry]
    for n in names:
        if names.count(n) > 1:
            all_findings.append(Finding("block", n, "duplicate tool name"))
    for tool in registry:
        all_findings.extend(lint_tool(tool))
    return all_findings


GOOD_REGISTRY = [
    {
        "name": "notes_list",
        "description": (
            "Use when the user wants to see all notes or a filtered list by tag. "
            "Do not use for reading a single note's full body; use notes_get instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Optional tag filter (e.g., 'urgent')"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "notes_search",
        "description": (
            "Use when the user asks to find notes by content keywords. "
            "Do not use for tag-based filters; use notes_list with the tag parameter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query (e.g., 'meeting notes')"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
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
                "body": {"type": "string", "description": "Markdown-formatted body (e.g., '# Title\n\nContent')"},
                "tag": {"type": "string", "description": "Optional single tag"},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        },
    },
]


BAD_REGISTRY = [
    {
        "name": "file_cut",
        "description": (
            "Use when you need to cut a file. Do not use for copying files; "
            "use file_copy instead"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cut": {"type": "string" , "description" : "Cut a file"},
            },
            "required": ["cut"],
            "additionalProperties": False,
        },
    },
    {
        "name": "file_copy",
        "description": (
            "Use when you need to copy a file. Do not use for cutting files; "
            "use file_cut instead"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "copy": {"type": "string" , "description" : "Copy a file"},
            },
            "required": ["copy"],
            "additionalProperties": False,
        },
    },
    {
        "name": "file_paste",
        "description": (
            "Use when you need to paste a file. Do not use for copying or cutting files; "
            "use file_copy or file_cut instead"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paste": {"type": "string" , "description" : "Paste a file"},
            },
            "required": ["paste"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Use when looks up weather in a city. Do not use for looking up weather in a country;"
            "use get_weather_in_country instead"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "units": {"type": "string" ,"enum":["celsius","farenheit"] , "description" : "Units - celsius or farenheit"},
                "city" : {"type" : "string" , "description" : "City name"}
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "notes_list",
        "description": (
            "Use when need to list all notes or a filtered list by tag. "
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
]


def report(name: str, registry: list[dict]) -> None:
    print("-" * 72)
    print(f"REGISTRY : {name} ({len(registry)} tools)")
    findings = lint_registry(registry)
    if not findings:
        print("  PASS : no findings")
        return
    severities: dict[str, int] = {}
    for f in findings:
        severities[f.severity] = severities.get(f.severity, 0) + 1
        print(f"  {f}")
    total = sum(severities.values())
    print(f"  summary: {total} findings "
          f"({severities.get('block', 0)} block, "
          f"{severities.get('warn', 0)} warn, "
          f"{severities.get('nit', 0)} nit)")


def main() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 05 - TOOL SCHEMA LINTER")
    print("=" * 72)
    print()
    report("GOOD_REGISTRY", GOOD_REGISTRY)
    print()
    report("BAD_REGISTRY", BAD_REGISTRY)


if __name__ == "__main__":
    main()
