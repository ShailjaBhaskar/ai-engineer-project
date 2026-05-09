from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool

from app.core.config import OPENAI_API_KEY


# STEP 1 — Create LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
)


# STEP 2 — Create Tool
python_tool = PythonREPLTool()


# STEP 3 — Define Tools List
tools = [
    Tool(
        name="Python Calculator",
        func=python_tool.run,
        description="Useful for solving math calculations"
    )
]


# STEP 4 — Create Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)


# STEP 5 — Ask Question
response = agent.invoke(
    "What is (458 * 923) + 100?"
)


# STEP 6 — Print
print(response)