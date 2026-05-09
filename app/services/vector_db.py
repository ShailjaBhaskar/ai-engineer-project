import chromadb
from app.services.embedding_service import get_embedding

client = chromadb.Client()
# delete collection if exists
try:
    client.delete_collection(name="docs")
except:
    pass

collection = client.create_collection(name="docs")

def store_documents(docs):
    for i, doc in enumerate(docs):
        embedding = get_embedding(doc)

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc]
        )

def query_db(query, top_k=2):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]        