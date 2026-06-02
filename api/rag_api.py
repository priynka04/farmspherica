from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.rag_chain import ask_question, chat_history

app = FastAPI(
    title="Farmspherica RAG Smart Farming Assistant",
    description="Ask hydroponic farming questions. Get answers with citations.",
    version="2.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question:         str
    answer:           str
    sources:          list
    chunks_retrieved: int
    confidence:       str   # HIGH or LOW
    rewritten_query:  str   # the improved version of the question that was searched

@app.get("/")
def root():
    return {"message": "Farmspherica RAG API is running!"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Ask a hydroponic farming question.
    Returns the answer, sources cited, confidence level,
    and the rewritten version of your question.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = ask_question(request.question)
    return result

@app.post("/clear-memory")
def clear_memory():
    """Clears the conversation history so the next question starts fresh."""
    chat_history.clear()
    return {"message": "Conversation memory cleared."}

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "service": "Farmspherica RAG API",
        "version": "2.0.0 — with query rewriting, compression, guardrails"
    }