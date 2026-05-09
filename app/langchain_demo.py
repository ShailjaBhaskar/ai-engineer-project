from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import OPENAI_API_KEY


# STEP 1 — Create LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3
)


# STEP 2 — Create Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms"
)


# STEP 3 — Create Output Parser
parser = StrOutputParser()


# STEP 4 — Create Chain
chain = prompt | llm | parser


# STEP 5 — Run Chain
response = chain.invoke({"topic": "AI"})


# STEP 6 — Print Response
print(response)