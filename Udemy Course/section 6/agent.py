# Code 2 - Updated
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import create_openai_functions_agent, AgentExecutor
from dotenv import load_dotenv

from tools.sql import run_query_tool  # Assuming this is correctly importing from the updated path

load_dotenv()

chat = ChatOpenAI()

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools = [run_query_tool]

agent = create_openai_functions_agent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools
)

# Using invoke to call the agent_executor
response = agent_executor.invoke({"input": "How many users are in the database?"})
print(response)