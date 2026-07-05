from typing import TypedDict , List
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

class ResearchState(TypedDict):
    topic : str
    plan :  str
    search_results : List[str]
    draft : str
    final_brief : str

def plan_node(state : ResearchState) -> dict:
    topic = state["topic"]
    prompt = (
        "Create a detailed research plan for the following topic: " + topic + ". Return the plan as a list of search queries"
    )
    llm = ChatOllama(model="gemma4:e4b" , temperature=0)
    result = llm.invoke(prompt)
    return {"plan" : result.content}

def search_node(state : ResearchState) -> dict:
    plan = state["plan"]
    llm = ChatOllama(model="gemma4:e4b" , temperature=0)
    results = []
    print("\n=== SEARCHING FOR : \n" + plan)
    for query in plan.split(","):
        result = llm.invoke(query)
        results.append(result.content)
    return {"search_results" : results}

def write_node(state : ResearchState) -> dict:
    topic = state["topic"]
    search_results = state["search_results"]
    prompt = (
        "Write a comprehensive research brief on the following topic: " + topic + ". Use the following search results: " + "\n".join(search_results)
    )
    llm = ChatOllama(model="gemma4:e4b" , temperature=0)
    result = llm.invoke(prompt)
    return {"draft" : result.content}

def cite_node(state : ResearchState) -> dict:
    llm = ChatOllama(model="gemma4:e4b" , temperature=0)
    draft = state["draft"]
    prompt = (
        "Add citations to the following draft: " + draft + ". Use the following search results: " + "\n".join(state["search_results"])
    )
    result = llm.invoke(prompt)
    return {"final_brief" : result.content}

def build_app():
    graph = StateGraph(ResearchState)
    graph.add_node("plan" , plan_node)
    graph.add_node("search" , search_node)
    graph.add_node("write" , write_node)
    graph.add_node("cite" , cite_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan" , "search")
    graph.add_edge("search" , "write")
    graph.add_edge("write" , "cite")
    graph.add_edge("cite" , END)

    return graph.compile()

if __name__ == "__main__":
    app = build_app()
    res = app.invoke({"topic" : "Anthropic's headquarters"})
    print("\n\nsearch_results\n\n")
    print(res["search_results"])
    print("\n\ndraft\n\n")
    print(res["draft"])
    print("\n\nfinal_brief\n\n")
    print(res["final_brief"])

