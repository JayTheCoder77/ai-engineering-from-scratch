from autogen import ConversableAgent , GroupChat , GroupChatManager

researcher = ConversableAgent(
    name="Researcher",
    system_message="You research Anthropic's headquarters and provide bullet points.",
    llm_config={"config_list": [{"model": "gemma4:e4b", "price": [0, 0] , "api_type": "ollama", "client_host": "http://127.0.0.1:11434/"}]}
)   

writer = ConversableAgent(
    name="Writer",
    system_message="You write a professional briefing based on the research findings.",
    llm_config={"config_list": [{"model": "gemma4:e4b", "price": [0, 0] , "api_type": "ollama", "client_host": "http://127.0.0.1:11434/"}]}
)
   
editor = ConversableAgent(
    name="Editor",
    system_message="You verify the writer's brief against search facts and add citations",
    llm_config={"config_list": [{"model": "gemma4:e4b", "price": [0, 0] , "api_type": "ollama", "client_host": "http://127.0.0.1:11434/"}]}
)   

groupchat = GroupChat(
    agents=[researcher , writer , editor],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list" : [{"model": "gemma4:e4b", "price": [0, 0] , "api_type": "ollama", "client_host": "http://127.0.0.1:11434/"}]}
)

from autogen import UserProxyAgent

user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config=False
)

user_proxy.initiate_chat(
    manager,
    message="Research Anthropic's headquarters, write a 200-word brief, and cite sources."
)