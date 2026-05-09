from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import OPENAI_API_KEY


# STEP 1 — Documents
documents = [
    "AI is the simulation of human intelligence.",
    "Machine Learning is a subset of AI.",
    "FastAPI is used for building APIs."
]


# STEP 2 — Embedding Model
embedding_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)


# STEP 3 — Create Vector Store
vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model
)


# STEP 4 — Create Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# STEP 5 — LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3
)


# STEP 6 — Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question ONLY from the context below.

Context:
{context}

Question:
{question}
"""
)


# STEP 7 — User Question
question = "What is AI?"


# STEP 8 — Retrieve Documents
retrieved_docs = retriever.invoke(question)


# STEP 9 — Combine Context
context = "\n".join([doc.page_content for doc in retrieved_docs])


# STEP 10 — Create Chain
chain = prompt | llm | StrOutputParser()


# STEP 11 — Run Chain
response = chain.invoke({
    "context": context,
    "question": question
})


# STEP 12 — Print
print(response)