from __future__ import annotations

from typing import Annotated, TypedDict , NotRequired
from pydantic import BaseModel , Field

# from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command , Send

class State(TypedDict):
    task : str
    research_brief : str
    draft : str
    review : str
    next_step : str

class ResearcherState(TypedDict):
    task : str
    research_brief : str

class WriterState(TypedDict):
    research_brief : str
    draft : str

class EditorState(TypedDict):
    draft : str
    review : str

@tool
def calculator(expression: str) -> str:
    """Evaluate a Python arithmetic expression like '2 + 2 * 3'. Returns the
    result as a string."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "ERROR: only digits and + - * / ( ) are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"ERROR: {exc!r}"


@tool
def web_lookup(query: str) -> str:
    """Fake web search. Returns canned facts for known queries and 'unknown'
    otherwise. Stand-in for a real retrieval tool."""
    facts = {
        "anthropic headquarters": "Anthropic is headquartered in San Francisco, California.",
        "python release year": "Python was first released in 1991.",
    }
    return facts.get(query.strip().lower(), "unknown")


TOOLS = [calculator, web_lookup]

def build_researcher(llm=None):
    if llm is None:
        llm = ChatOllama(model="gemma4:e4b" , temperature=0)
    
    def research_node(state : ResearcherState) -> dict:
        task = state["task"]
        prompt = (
            "You are a world class researcher. Your job is to perform in-depth research on the task assigned to you. Task : " + task + ".Provide a comprehensive research brief"
        )
        result = llm.invoke(prompt)
        return {"research_brief" : result.content}
    
    graph = StateGraph(ResearcherState)
    graph.add_node("research" , research_node)
    graph.set_entry_point("research")
    graph.add_edge("research" , END)

    return graph.compile(checkpointer=MemorySaver())

def build_writer(llm=None):
    if llm is None:
        llm = ChatOllama(model="gemma4:e4b" , temperature=0)
    
    def writer_node(state : WriterState) -> dict:
        research_brief = state["research_brief"]
        prompt = (
            "You are a world class writer. Your job is to write a comprehensive article based on the research brief provided to you. Research Brief : " + research_brief + ".Write a comprehensive article"
        )
        result = llm.invoke(prompt)
        return {"draft" : result.content}
    
    graph = StateGraph(WriterState)
    graph.add_node("writer" , writer_node)
    graph.set_entry_point("writer")
    graph.add_edge("writer" , END)

    return graph.compile(checkpointer=MemorySaver())
    
def build_reviewer(llm=None):
    if llm is None:
        llm = ChatOllama(model="gemma4:e4b" , temperature=0)
    
    def reviewer_node(state : EditorState) -> dict:
        draft = state["draft"]
        prompt = (
            "You are a world class editor. Your job is to review the article provided to you. Draft : " + draft + ".Review the article"
        )
        result = llm.invoke(prompt)
        return {"review" : result.content}
    
    graph = StateGraph(EditorState)
    graph.add_node("reviewer" , reviewer_node)
    graph.set_entry_point("reviewer")
    graph.add_edge("reviewer" , END)

    return graph.compile(checkpointer=MemorySaver())


def build_supervisor(llm=None):
    if llm is None:
        llm = ChatOllama(model="gemma4:e4b" , temperature=0)

    def supervisor_node(state : State) -> dict:
        if not state.get("research_brief"):
            return {"next_step" : "research"}
        if not state.get("draft"):
            return {"next_step" : "writer"}
        if not state.get("review"):
            return {"next_step" : "reviewer"}
        return {"next_step" : "FINISH"}
        
    def route_supervisor(state : State) -> Send | str:
        if state.get("next_step") == "research":
            return Send("researcher" , {"task" : state["task"] })
        if state.get("next_step") == "writer":
            return Send("writer" , {"research_brief" : state["research_brief"] })
        if state.get("next_step") == "reviewer":
            return Send("reviewer" , {"draft" : state["draft"] })
        return END
    
    graph = StateGraph(State)
    graph.add_node("supervisor" , supervisor_node)
    graph.add_node("researcher" , build_researcher(llm=llm))
    graph.add_node("writer" , build_writer(llm=llm))
    graph.add_node("reviewer" , build_reviewer(llm=llm))

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            END: END
        }
    )
    graph.add_edge("researcher" , "supervisor")
    graph.add_edge("writer" , "supervisor")
    graph.add_edge("reviewer" , "supervisor")

    app = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["writer"]
    )

    return app


def run_supervisor_demo() -> None:
    app = build_supervisor()
    config = {"configurable": {"thread_id": "real-demo-99"}}
    
    # 1. Start execution with a real task
    print("\n=== STARTING SUPERVISOR GRAPH ===")
    task = "Write a short article about Lionel Messi"
    print(f"Task: {task}\n")
    
    for event in app.stream({"task": task}, config, stream_mode="updates"):
        print(f"Event Update: {event}")
        
    # It will pause at the interrupt before writer
    state = app.get_state(config)
    print("\n=== INTERRUPTED BEFORE WRITER ===")
    print("Pending node to run next:", state.next)
    print("Generated Research Brief:")
    print("-" * 50)
    print(state.values.get("research_brief"))
    print("-" * 50)
    
    # 2. Interactively edit the state in the terminal
    choice = input("\nDo you want to edit this research brief? (yes/no): ").strip().lower()
    if choice in ["yes", "y"]:
        print("Type/paste your revised research brief below:")
        custom_brief = input("> ")
        if custom_brief.strip():
            # Overwrite the state before resuming
            app.update_state(config, {"research_brief": custom_brief})
            print("\nResearch brief successfully updated!")
    
    # 3. Resume execution
    print("\n=== RESUMING GRAPH ===")
    for event in app.stream(Command(resume=True), config, stream_mode="updates"):
        print(f"Event Update: {event}")
        
    # Completed execution
    final_state = app.get_state(config)
    print("\n=== WORKFLOW COMPLETE ===")
    print("Final Draft:")
    print("-" * 50)
    print(final_state.values.get("draft"))
    print("-" * 50)
    print("Final Review:")
    print("-" * 50)
    print(final_state.values.get("review"))
    print("-" * 50)



if __name__ == "__main__":
    run_supervisor_demo()
