from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.ollama import Ollama

def search_tool(query : str) -> str:
    """Search the web for company details and office history."""
    # Logic to fetch info (or return mock data for testing)
    return "Anthropic HQ is at 95 Hawthorne St, San Francisco, CA. Formerly at 530 Lytton Ave, Palo Alto."

def write_and_cite_tool(brief_content : str) -> str:
    """Save the final brief and citation details."""
    return f"Saved brief: {brief_content[:50]}..."

db = SqliteDb(session_table="research_sessions", db_file="agno_sessions.db")

agent = Agent(
    name="Researcher",
    model=Ollama(id="gemma4:e4b"), 
    markdown=True,
    instructions=[
        "Locate the headquarters of the topic using the search tool.",
        "Write a 200-word brief on the headquarters.",
        "Cite the sources you found." 
    ],
    tools=[search_tool , write_and_cite_tool],
    # Attach a session store so the agent remembers sessions across restarts
    db=db,
    read_tool_call_history=True
)

agent.print_response("Research Anthropic's headquarters, write a 200-word brief, and cite sources.")