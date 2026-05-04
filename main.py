# from app.services.ai_service import get_ai_response

# def main():
#     while True:
#         user_input = input("You: ")
        
#         if user_input.lower() == "exit":
#             break
        
#         response = get_ai_response(user_input)
#         print("AI:", response)

# if __name__ == "__main__":
#     main()

# from app.services.embedding_service import get_embedding
# import numpy as np

# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))    

# def main():
#     emb1 = get_embedding("AI")
#     emb2 = get_embedding("Artificial Intelligence")

#     print("AI:", emb1[:5])
#     print("Artificial Intelligence:", emb2[:5])
#     vec1 = get_embedding("AI")
#     vec2 = get_embedding("Artificial Intelligence")
#     vec3 = get_embedding("Pizza")

#     print("AI vs AI:", cosine_similarity(vec1, vec1))
#     print("AI vs Artificial Intelligence:", cosine_similarity(vec1, vec2))
#     print("AI vs Pizza:", cosine_similarity(vec1, vec3))


# if __name__ == "__main__":
#     main()

from app.services.rag_service import rag_pipeline
from app.services.vector_db import store_documents
from app.services.rag_service import all_chunks

def main():
    store_documents(all_chunks)
    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        answer = rag_pipeline(query)
        print("AI:", answer)

if __name__ == "__main__":
    main()