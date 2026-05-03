from app.services.embedding_service import get_embedding
import numpy as np
from openai import OpenAI
from app.core.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

documents = [
    "AI is the simulation of human intelligence.",
    "Machine Learning is a subset of AI.",
    "FastAPI is a Python framework for building APIs.",
    "Embeddings convert text into numerical vectors."
]

def create_document_embeddings():
    embeddings = []

    for doc in documents:
        emb = get_embedding(doc)
        embeddings.append((doc, emb))

    return embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve(query, doc_embeddings, top_k=2):
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
                "content": "Answer based ONLY on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def rag_pipeline(query):
    doc_embeddings = create_document_embeddings()
    relevant_docs = retrieve(query, doc_embeddings)
    answer = generate_answer(query, relevant_docs)

    return answer