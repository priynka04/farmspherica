import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

VECTOR_STORE_PATH = "models/faiss_index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load vector store once at startup
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Conversation memory
chat_history = ChatMessageHistory()

# Groq LLM (free)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    max_tokens=1024
)


def ask_question(question: str) -> dict:
    # Step 1: Retrieve top 5 relevant document chunks
    retrieved_docs = vector_store.similarity_search(question, k=5)

    # Step 2: Build context from retrieved chunks
    context_parts = []
    sources = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source_file", "Unknown")
        topic = doc.metadata.get("topic", "Unknown")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # Step 3: Get conversation history (last 4 messages)
    history_text = ""
    for msg in chat_history.messages[-4:]:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    # Step 4: Build prompt
    prompt = f"""You are a smart hydroponic farming assistant for Farmspherica Innovations.
Answer the user's question using ONLY the provided context documents.
Always cite your source at the end of your answer.
If the answer is not in the context, say: "I don't have enough information on that topic in my knowledge base."

Previous conversation:
{history_text if history_text else "None"}

Context from knowledge base:
{context}

User question: {question}

Answer (cite sources at the end):"""

    # Step 5: Generate answer
    response = llm.invoke(prompt)
    answer = response.content

    # Step 6: Save to memory
    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks_retrieved": len(retrieved_docs)
    }


def test_rag():
    test_questions = [
        "What is the ideal pH range for hydroponic lettuce?",
        "What nutrients are most important for hydroponic plants?",
        "How does EC (electrical conductivity) affect plant growth?",
        "What temperature should the water be in a hydroponic system?",
        "How do I manage nutrient deficiency in hydroponics?",
    ]

    print("=== RAG SYSTEM TEST ===\n")
    for i, question in enumerate(test_questions, 1):
        print(f"Q{i}: {question}")
        result = ask_question(question)
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Sources: {result['sources'][:2]}")
        print("-" * 60)


if __name__ == "__main__":
    test_rag()