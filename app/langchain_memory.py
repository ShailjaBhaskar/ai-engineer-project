from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import OPENAI_API_KEY


# STEP 1 — Create model
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3
)


# STEP 2 — Memory list
chat_history = []


while True:

    # STEP 3 — User input
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break


    # STEP 4 — Add user message to history
    chat_history.append(HumanMessage(content=user_input))


    # STEP 5 — Send full history to model
    response = llm.invoke(chat_history)


    # STEP 6 — Print response
    print("AI:", response.content)


    # STEP 7 — Store AI response
    chat_history.append(AIMessage(content=response.content))