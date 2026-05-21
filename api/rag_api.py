from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.rag_chain import ask_question, chat_history

app = FastAPI(
    title="Farmspherica RAG Smart Farming Assistant",
    description="Ask questions about hydroponic farming. Get answers with citations.",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list
    chunks_retrieved: int

@app.get("/")
def root():
    return {"message": "Farmspherica RAG API is running!"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """Ask a hydroponic farming question. Returns answer with source citation."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = ask_question(request.question)
    return result

@app.post("/clear-memory")
def clear_memory():
    """Clear the conversation history."""
    chat_history.clear()
    return {"message": "Conversation memory cleared."}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Farmspherica RAG API"}