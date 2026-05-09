from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.core.config import OPENAI_API_KEY


# STEP 1 — Define State
class ChatState(TypedDict):
    question: str
    answer: str


# STEP 2 — Create LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3
)


# STEP 3 — Define Node Function
def generate_answer(state: ChatState):

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
    }


# STEP 4 — Create Graph
graph = StateGraph(ChatState)


# STEP 5 — Add Node
graph.add_node("answer_node", generate_answer)


# STEP 6 — Set Entry Point
graph.set_entry_point("answer_node")


# STEP 7 — Add Edge to END
graph.add_edge("answer_node", END)


# STEP 8 — Compile Graph
app = graph.compile()


# STEP 9 — Run Graph
result = app.invoke({
    "question": "What is AI?"
})


# STEP 10 — Print Result
print(result)