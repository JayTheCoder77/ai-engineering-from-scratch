from crewai import Agent , Task , Crew , Process

from crewai import LLM

llm = LLM(
    model="ollama/gemma4:e4b",
    base_url="http://localhost:11434"
)

researcher = Agent(
    role="Headquarters Researcher",
    goal="Locate and research the exact address, details, and history of {topic}'s headquarters.",
    backstory="You are a meticulous researcher specialized in corporate offices, real estate moves, and company history.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Senior Briefing Writer",
    goal="Create a polished, professional briefing based on research findings.",
    backstory="""You are an expert technical writer. Your job is to take raw, unformatted research and turn it into
    a clear, well-structured, and professional briefing document.
    Follow this format: 
    1. Summary  
    2. Key Details
    3. History   
    4. Insights
    
    NO MARKDOWN. NO FORMATTING. Just plain text.
    """,
    llm=llm,
    verbose=True,
)

editor = Agent(
    role="Final Briefing Editor-in-Chief",
    goal="Review the drafted briefing and make any final improvements for clarity, flow, and professionalism.",
    backstory="""You are the final gatekeeper. You check for typos, awkward phrasing, repetition, and logical
    inconsistencies.
    You do NOT change the core information—you only polish the presentation.
    """,
    llm=llm,
    verbose=True,
)

research_task = Task(
    description = "Research the headquarters of {topic}. Find the exact location, address history, and key facts.",
    expected_output="A comprehensive list of findings and facts about the headquarters.",
    agent=researcher,
)

write_task = Task(
    description = "Write a professional briefing based on the research findings.",
    expected_output="A well-structured briefing document with summary, details, history, and insights.",
    agent=writer,
    context=[research_task]
)

edit_task = Task(
    description = "Edit and polish the drafted briefing.",
    expected_output="A final, professional briefing document ready for delivery.",
    agent=editor,
    context=[write_task]
)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={"topic" : "Anthropic's headquarters"})
print(result)