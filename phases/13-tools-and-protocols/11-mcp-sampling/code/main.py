"""Phase 13 Lesson 11 - MCP sampling harness (server -> client LLM calls).

Simulated server-to-client sampling:
  - Server's summarize_repo tool runs two sampling rounds (pick files, then
    synthesize) by calling a 'fake_client_sample' stand-in for the client.
  - Rate-limited at max_samples_per_tool to prevent loop bombs.
  - ModelPreferences are printed so you can see the cost/speed/intelligence
    trade-off shape.

Stdlib only.

Run: python code/main.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import uuid

FAKE_REPO = {
    "README.md": "This repo implements the toy MCP notes server.",
    "server.py": "def dispatch(msg): ... handler code ...",
    "client.py": "def connect(): ... subprocess Popen ...",
    "LICENSE": "MIT",
    "tests/test_server.py": "def test_initialize(): ...",
    "assets/diagram.svg": "<svg>...</svg>",
    "docs/intro.md": "## Introduction to the toy notes server",
}

FAKE_CONTENT = {
    "chunk_1":  "Introduction",
    "chunk_2":  "Main Topic: Machine Learning",
    "chunk_3":  "Subtopic: Deep Learning",
    "chunk_4":  "Subtopic: Reinforcement Learning",
    "chunk_5":  "Conclusion"
}

CANNED_RESPONSES = {
    "pick": json.dumps(["README.md", "server.py", "docs/intro.md"]),
    "pick_chunks" : json.dumps(["chunk_1", "chunk_2", "chunk_3"]),
    "summarize": "This repo is a toy MCP server teaching the sampling loop. "
                 "The server dispatches JSON-RPC methods; clients drive it over stdio. "
                 "Documentation in docs/ introduces the pattern end to end.",
}

@dataclass
class SampleRequest:
    messages: list[dict]
    system_prompt: str
    model_preferences: dict
    max_tokens: int = 1024
    include_context: str = "none"
    tools: list[dict] | None = None
    user_approved : bool = True
    session_id : str = str(uuid.uuid4())

@dataclass
class SampleResponse:
    role: str
    content: dict
    model: str
    stop_reason: str


def fake_client_sample(req: SampleRequest) -> SampleResponse:
    """Stand-in for the client's LLM. Picks a canned response by keyword."""
    text = req.messages[-1]["content"]["text"].lower()
    
    if req.user_approved is not True:
        return SampleResponse(
            role="assistant",
            content={"type": "text", "text": "Sampling request denied by user."},
            model="claude-3-5-sonnet-fake",
            stop_reason="refusal"
        )
    if req.tools is None:
        if "chunk" in text or "pdf" in text:
            body = CANNED_RESPONSES["pick_chunks"]
        elif "pick" in text or "choose" in text:
            body = CANNED_RESPONSES["pick"]
        else:
            body = CANNED_RESPONSES["summarize"]
        session_budget_helper(req.session_id)
        return SampleResponse(
            role="assistant",
            content={"type": "text", "text": body},
            model="claude-3-5-sonnet-fake",
            stop_reason="endTurn",
        )
    else:
        tool_name = req.tools[0]["name"]
        if tool_name == "get_file_size":
            print("[client] Executing tool get_file_size for README.md...")
            size = len(FAKE_REPO["README.md"])    
            print(f"[client] Tool output: README.md is {size} bytes")
            body = CANNED_RESPONSES["pick"]
            session_budget_helper(req.session_id)
            return SampleResponse(
                role="assistant",
                content={"type": "text", "text": body},
                model="claude-3-5-sonnet-fake",
                stop_reason="endTurn",
            )
@dataclass
class SamplingBudget:
    used: int = 0
    max_samples_per_tool: int = 5


SESSION_BUDGET : dict[str , SamplingBudget] = {}

def session_budget_helper(session_id : str):
    if session_id not in SESSION_BUDGET:
        SESSION_BUDGET[session_id] = SamplingBudget()
    budget = SESSION_BUDGET[session_id]
    if budget.used >= budget.max_samples_per_tool:
        raise RuntimeError(f"Session '{session_id}' rate limit exceeded! (used {budget.used}/{budget.max_samples_per_tool})")
    budget.used += 1
    print(f"    [session '{session_id[:8]}...'] sample #{budget.used}/{budget.max_samples_per_tool}")

def sample(req: SampleRequest, budget: SamplingBudget, ask_user: bool = False) -> SampleResponse:
    if budget.used >= budget.max_samples_per_tool:
        raise RuntimeError("sampling rate limit exceeded (loop bomb guard)")
    
    if ask_user :
        answer = input(f"    [HITL Prompt] Server requests sampling ({req.system_prompt[:40]}...). Approve? [y/N]: ")
        if answer.strip().lower() != 'y':
            return SampleResponse(
                role="assistant",
                content={"type": "text", "text": "Sampling request denied by user."},
                model="claude-3-5-sonnet-fake",
                stop_reason="refusal"
            )
    
    budget.used += 1
    print(f"    [sample #{budget.used}] model_prefs={req.model_preferences} "
        f"includeContext={req.include_context!r}")
    print(f"      system: {req.system_prompt[:60]}...")
    print(f"      user  : {req.messages[-1]['content']['text'][:60]}...")
    resp = fake_client_sample(req)
    print(f"      <- model={resp.model}  stop={resp.stop_reason}  "
        f"len={len(resp.content['text'])}")
    return resp

def summarize_repo_tool(args: dict) -> dict:
    budget = SamplingBudget()

    pick_req = SampleRequest(
        messages=[{"role": "user", "content": {"type": "text", "text":
            "Given this file list, pick five files most likely to describe the repo's purpose. "
            f"Files: {list(FAKE_REPO.keys())}. Reply as a JSON array of filenames."}}],
        system_prompt="You select representative files for repo summarization.",
        model_preferences={
            "costPriority": 0.5,
            "speedPriority": 0.3,
            "intelligencePriority": 0.1,
            "hints": [{"name": "claude-3-5-haiku"}],
        },
        max_tokens=256,
        include_context="none",
        tools = [{
            "name": "get_file_size",
            "description": "Returns byte count of a file",
            "inputSchema": {"type": "object", "properties": {"filename": {"type": "string"}}}
        }]

    )
    pick_resp = sample(pick_req, budget , ask_user=True)
    if pick_resp.stop_reason == "refusal":
        raise RuntimeError("sampling request denied by user")
    
    picked = json.loads(pick_resp.content["text"])
    print(f"    picked files: {picked}")

    combined = "\n\n".join(f"=== {f} ===\n{FAKE_REPO[f]}" for f in picked if f in FAKE_REPO)

    summ_req = SampleRequest(
        messages=[{"role": "user", "content": {"type": "text", "text":
            f"Summarize the repo in three paragraphs given these files:\n\n{combined}"}}],
        system_prompt="You write concise, accurate repo summaries.",
        model_preferences={
            "costPriority": 0.2,
            "speedPriority": 0.2,
            "intelligencePriority": 0.9,
            "hints": [{"name": "claude-3-5-sonnet"}],
        },
        max_tokens=512,
        include_context="none",
    )
    summ_resp = sample(summ_req, budget)

    return {
        "content": [{"type": "text", "text": summ_resp.content["text"]}],
        "isError": False,
        "_meta": {"samplesUsed": budget.used},
    }

def summarize_pdf_tool(args: dict) -> dict:
    budget = SamplingBudget()

    chunk_req = SampleRequest(
        messages=[{"role": "user", "content": {"type": "text", "text":
            "Given the table of contents, pick three most relevant chunks. "
            f"Contents: {list(FAKE_CONTENT.keys())}. Reply as a JSON array of chunks."}}],
        system_prompt="You select representative chunks for their summarization.",
        model_preferences={
            "costPriority": 0.5,
            "speedPriority": 0.3,
            "intelligencePriority": 0.2,
            "hints": [{"name": "claude-3-5-haiku"}],
        },
        max_tokens=256,
        include_context="none",
        )
    
    chunk_resp = sample(chunk_req, budget , ask_user=True)
    if chunk_resp.stop_reason == "refusal":
        raise RuntimeError("sampling request denied by user")
    
    picked = json.loads(chunk_resp.content["text"])
    print(f"    picked chunks: {picked}")

    combined = "\n\n".join(f"=== {f} ===\n{FAKE_CONTENT[f]}" for f in picked if f in FAKE_CONTENT)
    summ_req = SampleRequest(
        messages=[{"role": "user", "content": {"type": "text", "text":
        f"Summarize the document in three paragraphs given these chunks:\n\n{combined}"}}],
        system_prompt="You write concise, accurate document summaries.",
        model_preferences={
            "costPriority": 0.2,
            "speedPriority": 0.2,
            "intelligencePriority": 0.6,
            "hints": [{"name": "claude-3-5-sonnet"}],
        },
        max_tokens=512,
        include_context="none",
    )
    summ_resp = sample(summ_req, budget)

    return {
        "content": [{"type": "text", "text": summ_resp.content["text"]}],
        "isError": False,
        "_meta": {"samplesUsed": budget.used},
    }

def main() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 11 - MCP SAMPLING HARNESS")
    print("=" * 72)
    print()
    print("summarize_repo invoked (no server-side LLM credentials)")
    print("-" * 72)
    try:
        result = summarize_repo_tool({})
        print("\n  result.content[0].text:")
        print(f"    {result['content'][0]['text']}")
        print(f"\n  samples used: {result['_meta']['samplesUsed']}")
        result2 = summarize_pdf_tool({})
        print("\n  result2.content[0].text:")
        print(f"    {result2['content'][0]['text']}")
        print(f"\n  samples used: {result2['_meta']['samplesUsed']}")
    except RuntimeError as e:
        print(f"  loop-bomb guard triggered: {e}")

    # Test session rate limiting across 3 tool calls
    # for i in range(1, 4):
    #     print(f"\n--- Invocation #{i} ---")
    #     try:
    #         result = summarize_repo_tool({})
    #         print(f"  Result summary length: {len(result['content'][0]['text'])}")
    #     except RuntimeError as e:
    #         print(f"  [BLOCKED BY RATE LIMITER]: {e}")
    #         break


if __name__ == "__main__":
    main()
