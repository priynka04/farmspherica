import os
import cohere
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

VECTOR_STORE_PATH = "models/faiss_index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

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

# Cohere reranker (free)
co = cohere.Client(COHERE_API_KEY)


def ask_question(question: str) -> dict:

    # ── Step 1: Retrieve top 10 chunks from FAISS (wider net for reranker) ──
    retrieved_docs = vector_store.similarity_search(question, k=10)

    # ── Step 1b: RERANKING — reorder by relevance, keep top 5 ──────────────
    try:
        rerank_results = co.rerank(
            query=question,
            documents=[doc.page_content for doc in retrieved_docs],
            top_n=5,
            model="rerank-english-v3.0"
        )
        # Reorder retrieved_docs based on reranker's ranking
        retrieved_docs = [retrieved_docs[r.index] for r in rerank_results.results]
        print(f"  Reranking done — top 5 selected from 10 candidates")
    except Exception as e:
        # If reranking fails for any reason, just use top 5 from FAISS as fallback
        print(f"  Reranking skipped (using FAISS top 5 as fallback): {e}")
        retrieved_docs = retrieved_docs[:5]

    # ── Step 2: Build context from reranked chunks ──────────────────────────
    context_parts = []
    sources = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source_file", "Unknown")
        topic = doc.metadata.get("topic", "Unknown")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # ── Step 3: Get conversation history (last 4 messages) ──────────────────
    history_text = ""
    for msg in chat_history.messages[-4:]:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    # ── Step 4: Build prompt ─────────────────────────────────────────────────
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

    # ── Step 5: Generate answer ──────────────────────────────────────────────
    response = llm.invoke(prompt)
    answer = response.content

    # ── Step 6: Save to memory ───────────────────────────────────────────────
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
        "What is NFT (Nutrient Film Technique) in hydroponics?",
        "How often should I change the nutrient solution?",
        "What sensors are used to monitor hydroponic systems?",
        "What are common signs of overwatering in hydroponics?",
        "How does light affect hydroponic plant growth?",
    ]

    print("=== RAG SYSTEM TEST (with reranking) ===\n")
    correct = 0
    for i, question in enumerate(test_questions, 1):
        print(f"Q{i}: {question}")
        result = ask_question(question)
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Sources: {result['sources'][:2]}")

        # Simple self-check — did we get a real answer (not "I don't have info")?
        if "don't have enough information" not in result['answer'].lower():
            correct += 1
            print("  Result: ANSWERED")
        else:
            print("  Result: NO ANSWER FOUND")
        print("-" * 60)

    accuracy = (correct / len(test_questions)) * 100
    print(f"\nAccuracy: {correct}/{len(test_questions)} questions answered = {accuracy:.0f}%")
    print("Add this number to your rag_architecture.md under Test Results!")


if __name__ == "__main__":
    test_rag()