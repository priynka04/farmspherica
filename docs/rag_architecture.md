# RAG System Architecture — Farmspherica Smart Farming Assistant

## What it does
Answers hydroponic farming questions using a knowledge base of research papers,
with source citations, reranking, and conversation memory.

## How it works (step by step)
1. User asks a question via the POST /ask API endpoint
2. Question is converted to an embedding vector using HuggingFace
   sentence-transformers/all-MiniLM-L6-v2
3. FAISS searches the vector store for the top 10 candidate document chunks
4. Cohere Reranker (rerank-english-v3.0) reorders the 10 candidates
   and selects the best 5 by true semantic relevance
5. The reranked chunks + question + conversation history are sent to Groq LLM
   (llama-3.1-8b-instant)
6. LLM generates an answer with source citations
7. Answer is returned to the user and saved to ChatMessageHistory memory

## Files
- api/document_loader.py       → loads and chunks all PDFs and text files
- api/vector_store.py          → builds and saves FAISS index to disk
- api/rag_chain.py             → retrieval + reranking + LLM + memory
- api/rag_api.py               → FastAPI /ask endpoint with citation support
- api/test_project_questions.py → project-specific RAG test script
- api/download_papers.py       → auto-downloads papers from Semantic Scholar
- models/faiss_index/          → saved FAISS vector store (auto-generated)
- tests/test_rag.py            → 3 unit tests for RAG system
- docs/knowledge_base/         → all research PDFs organised by topic

## API Endpoints
- POST /ask               → ask a farming question, returns answer + sources
- POST /clear-memory      → reset conversation history
- GET  /health            → check if API is running
- GET  /docs              → interactive Swagger API documentation

## Knowledge Base Statistics
- Total PDF pages loaded  : 3169 pages
- Total chunks indexed    : 9014 chunks
- Chunk size              : 1000 tokens, 150 overlap
- Embedding model         : sentence-transformers/all-MiniLM-L6-v2 (HuggingFace, local)
- Vector store            : FAISS (saved locally at models/faiss_index/)
- LLM                     : Groq — llama-3.1-8b-instant (free)
- Reranker                : Cohere rerank-english-v3.0 (free tier)
- Retrieval strategy      : FAISS top-10 → Cohere rerank → top-5 sent to LLM

## Topics Covered in Knowledge Base
| Folder         | Content                                               |
|----------------|-------------------------------------------------------|
| plant_health   | Plant disease, stress detection, precision farming    |
| nutrients      | Nutrient solutions, deficiency management, NPK        |
| ph_management  | pH control, EC management, automated monitoring       |
| crop_growth    | Growth stages, yield optimization, lettuce/tomato     |
| iot_sensors    | IoT systems, sensor types, automation control logic   |

## Test Results

### Unit Tests (tests/test_rag.py)
| Test                        | Result  |
|-----------------------------|---------|
| RAG returns answer          | PASSED  |
| Sources returned            | PASSED  |
| Multi-turn memory working   | PASSED  |

### General Retrieval Accuracy (10 questions — with reranking)
| Question                                        | Answered | Source Cited |
|-------------------------------------------------|----------|--------------|
| Ideal pH range for hydroponic lettuce?          | YES      | YES          |
| Most important nutrients for hydroponic plants? | YES      | YES          |
| How does EC affect plant growth?                | YES      | YES          |
| Water temperature for hydroponic system?        | YES      | YES          |
| Managing nutrient deficiency in hydroponics?    | YES      | YES          |
| What is NFT (Nutrient Film Technique)?          | YES      | YES          |
| How often to change nutrient solution?          | YES      | YES          |
| Sensors used to monitor hydroponic systems?     | YES      | YES          |
| Signs of overwatering in hydroponics?           | YES      | YES          |
| How does light affect hydroponic plant growth?  | YES      | YES          |

**Overall accuracy: 10/10 = 100%**

### Project-Specific Questions (Farmspherica Nano PAW)
| Question                                              | Result                    |
|-------------------------------------------------------|---------------------------|
| pH range for Farmspherica hydroponic system?          | [fill after running test] |
| Ideal TDS level for crops being grown?                | [fill after running test] |
| Sensors needed for Nano PAW monitoring system?        | [fill after running test] |
| Nutrient level for healthy lettuce growth?            | [fill after running test] |
| How often to change nutrient solution in the system?  | [fill after running test] |

## Known Issues / Limitations
- Gemini free tier quota was exhausted — switched to Groq (free, faster)
- HuggingFace Hub sends unauthenticated warning — add HF_TOKEN to .env to suppress
- models/faiss_index/ and docs/knowledge_base/ excluded from Git (large auto-generated files)
- Reranker falls back to FAISS top-5 automatically if Cohere API is unavailable

## How to Run
1. Build vector store (first time only):
   python api/vector_store.py

2. Test RAG chain with reranking (10 general questions):
   python api/rag_chain.py

3. Test project-specific questions:
   python api/test_project_questions.py

4. Start API server:
   uvicorn api.rag_api:app --reload --port 8000

5. Run unit tests:
   python tests/test_rag.py

6. Open interactive docs:
   http://localhost:8000/docs