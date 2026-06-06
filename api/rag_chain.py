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
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
COHERE_API_KEY    = os.getenv("COHERE_API_KEY")

# ── Load vector store once when the file starts ──────────────────────────────
# ── Lazy loading — loads only when first question is asked ───────────────────
embeddings   = None
vector_store = None

def get_vector_store():
    global embeddings, vector_store
    if vector_store is None:
        print("[INFO] Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("[INFO] Loading FAISS index...")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[INFO] Vector store loaded successfully")
    return vector_store

# ── Conversation memory ───────────────────────────────────────────────────────
chat_history = ChatMessageHistory()

# ── Groq LLM ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    max_tokens=1024
)

# ── Cohere reranker ───────────────────────────────────────────────────────────
co = cohere.Client(COHERE_API_KEY)


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION 1 — Query rewriting
# Makes vague questions clearer before searching
# Example: "what about pH?" → "What is the ideal pH range for hydroponics?"
# ════════════════════════════════════════════════════════════════════════════
def rewrite_query(question: str) -> str:
    rewrite_prompt = f"""You are a hydroponic farming expert.
Rewrite this question to be more specific and searchable.
Keep it as one clear sentence. Return ONLY the rewritten question, nothing else.

Original question: {question}

Rewritten question:"""

    try:
        response  = llm.invoke(rewrite_prompt)
        rewritten = response.content.strip()
        if len(rewritten) < 10:
            return question
        print(f"  Query rewritten: '{question}' -> '{rewritten}'")
        return rewritten
    except Exception:
        return question


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION 2 — Context compression
# Removes sentences from a chunk that are NOT relevant to the question
# So the LLM gets cleaner, shorter context and gives better answers
# ════════════════════════════════════════════════════════════════════════════
def compress_chunk(chunk_text: str, question: str) -> str:
    compress_prompt = f"""Extract only the sentences from the text below that are
directly relevant to answering this question. Return only those sentences, nothing else.
If the entire text is relevant, return it as-is.
If nothing is relevant, return exactly: NOT_RELEVANT

Question: {question}

Text:
{chunk_text}

Relevant sentences:"""

    try:
        response   = llm.invoke(compress_prompt)
        compressed = response.content.strip()
        if compressed == "NOT_RELEVANT" or len(compressed) < 20:
            return ""
        return compressed
    except Exception:
        return chunk_text


# ════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION — ask_question()
# This is called every time someone asks a question
# ════════════════════════════════════════════════════════════════════════════
def ask_question(question: str) -> dict:

    # ── Step 1a: Rewrite question to be clearer ──────────────────────────────
    search_query = rewrite_query(question)

    # ── Step 1b: Search FAISS for top 10 matching chunks ─────────────────────
    retrieved_docs = get_vector_store().similarity_search(search_query, k=10)

    # ── Step 1c: Rerank top 10 → keep best 5 using Cohere ────────────────────
    try:
        rerank_results = co.rerank(
            query=search_query,
            documents=[doc.page_content for doc in retrieved_docs],
            top_n=5,
            model="rerank-english-v3.0"
        )
        retrieved_docs = [retrieved_docs[r.index] for r in rerank_results.results]
        print(f"  Reranking done — top 5 selected from 10 candidates")
    except Exception as e:
        print(f"  Reranking skipped (fallback to FAISS top 5): {e}")
        retrieved_docs = retrieved_docs[:5]

    # ── Step 2: Compress each chunk — remove irrelevant sentences ────────────
    context_parts = []
    sources       = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source_file", "Unknown")
        topic  = doc.metadata.get("topic",       "Unknown")

        # Keep only sentences relevant to the question
        compressed_text = compress_chunk(doc.page_content, question)

        if compressed_text:
            context_parts.append(f"[Source {i+1}: {source}]\n{compressed_text}")

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # ── Step 3: Load last 4 messages from conversation memory ─────────────────
    history_text = ""
    for msg in chat_history.messages[-4:]:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    # ── Step 4: Build the full prompt for the LLM ────────────────────────────
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

    # ── Step 5: Generate answer using Groq LLM ───────────────────────────────
    response = llm.invoke(prompt)
    answer   = response.content

    # ── Step 6: Save question and answer to memory ───────────────────────────
    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)

    # ── Step 7: Guardrails — check confidence of the answer ──────────────────
    low_confidence_phrases = [
        "i don't have enough information",
        "i don't know",
        "i cannot find",
        "not mentioned in",
        "no information",
        "cannot answer",
    ]
    is_low_confidence = any(
        phrase in answer.lower() for phrase in low_confidence_phrases
    )
    confidence = "LOW" if is_low_confidence else "HIGH"

    return {
        "question":         question,
        "answer":           answer,
        "sources":          sources,
        "chunks_retrieved": len(retrieved_docs),
        "confidence":       confidence,
        "rewritten_query":  search_query,
    }


# ════════════════════════════════════════════════════════════════════════════
# TEST FUNCTION — run 10 questions and measure accuracy
# ════════════════════════════════════════════════════════════════════════════
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

    print("=== RAG SYSTEM TEST (rewriting + reranking + compression + guardrails) ===\n")
    correct = 0
    for i, question in enumerate(test_questions, 1):
        print(f"Q{i}: {question}")
        result = ask_question(question)
        print(f"Answer: {result['answer'][:250]}...")
        print(f"Sources: {result['sources'][:2]}")
        print(f"Confidence: {result['confidence']}")
        print(f"Rewritten query: {result['rewritten_query']}")

        if "don't have enough information" not in result["answer"].lower():
            correct += 1
            print("  Result: ANSWERED")
        else:
            print("  Result: NO ANSWER FOUND")
        print("-" * 60)

    accuracy = (correct / len(test_questions)) * 100
    print(f"\nAccuracy: {correct}/{len(test_questions)} = {accuracy:.0f}%")


if __name__ == "__main__":
    test_rag()