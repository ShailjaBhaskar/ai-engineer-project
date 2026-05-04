from app.services.embedding_service import get_embedding
import numpy as np
from openai import OpenAI
from app.core.config import OPENAI_API_KEY
from app.services.vector_db import query_db

client = OpenAI(api_key=OPENAI_API_KEY)

# def chunk_text(text, chunk_size=50):
#     words = text.split()
#     chunks = []

#     for i in range(0, len(words), chunk_size):
#         chunk = " ".join(words[i:i+chunk_size])
#         chunks.append(chunk)

#     return chunks

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

documents = [
    "AI is the simulation of human intelligence. It includes machine learning and deep learning.",
    "FastAPI is a modern Python web framework used for building APIs."
]

all_chunks = []

for doc in documents:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

doc_embeddings = [(chunk, get_embedding(chunk)) for chunk in all_chunks]

# def create_document_embeddings():
#     embeddings = []

#     for doc in documents:
#         emb = get_embedding(doc)
#         embeddings.append((doc, emb))

#     return embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve(query, doc_embeddings, top_k=3):
    query_emb = get_embedding(query)

    scores = []

    for doc, emb in doc_embeddings:
        score = cosine_similarity(query_emb, emb)
        scores.append((doc, score))

    # sort by highest score
    scores.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scores[:top_k]]

def generate_answer(query, context_docs):
    context = "\n".join(context_docs)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful assistant.

                Answer ONLY from the provided context.
                If answer is not in context, say "I don't know".
                """
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# def rag_pipeline(query):
#     # doc_embeddings = create_document_embeddings()
#     relevant_docs = retrieve(query)
#     answer = generate_answer(query, relevant_docs)

#     return answer


def rag_pipeline(query):
    relevant_docs = query_db(query)
    answer = generate_answer(query, relevant_docs)
    return answer