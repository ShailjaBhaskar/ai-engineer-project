from fastapi import FastAPI
from pydantic import BaseModel
from app.services.rag_service import rag_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_ai(request: QueryRequest):
    answer = rag_pipeline(request.query)
    return {"answer": answer}