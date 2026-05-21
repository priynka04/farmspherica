# RAG System Architecture — Farmspherica Smart Farming Assistant

## What it does
Answers hydroponic farming questions using a knowledge base of research papers,
with source citations and conversation memory.

## How it works (step by step)
1. User asks a question via the POST /ask API endpoint
2. Question is converted to an embedding vector using HuggingFace
   sentence-transformers/all-MiniLM-L6-v2
3. FAISS searches the vector store for the 5 most relevant document chunks
4. The chunks + question + conversation history are sent to Groq LLM
   (llama-3.3-70b-versatile)
5. LLM generates an answer with source citations
6. Answer is returned to the user and saved to ConversationBufferMemory

## Files
- api/document_loader.py  → loads and chunks all PDFs and text files
- api/vector_store.py     → builds and saves FAISS index to disk
- api/rag_chain.py        → retrieval + LLM answer generation + memory
- api/rag_api.py          → FastAPI /ask endpoint with citation support
- models/faiss_index/     → saved FAISS vector store (auto-generated)
- tests/test_rag.py       → 3 unit tests for RAG system

## API Endpoints
- POST /ask               → ask a farming question, returns answer + sources
- POST /clear-memory      → reset conversation history
- GET  /health            → check if API is running
- GET  /docs              → interactive Swagger API documentation

## Knowledge Base Statistics
- Total PDF pages loaded : 3169 pages
- Total chunks indexed   : 9014 chunks
- Chunk size             : 1000 tokens, 150 overlap
- Embedding model        : sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
- Vector store           : FAISS (saved locally at models/faiss_index/)
- LLM                    : Groq — llama-3.3-70b-versatile

## Topics Covered in Knowledge Base
| Folder         | Content                                              |
|----------------|------------------------------------------------------|
| plant_health   | Plant disease, stress detection, precision farming   |
| nutrients      | Nutrient solutions, deficiency management, NPK       |
| ph_management  | pH control, EC management, automated monitoring      |
| crop_growth    | Growth stages, yield optimization, lettuce/tomato    |
| iot_sensors    | IoT systems, sensor types, automation control logic  |

## Test Results
| Test                        | Result  |
|-----------------------------|---------|
| RAG returns answer          | PASSED  |
| Sources returned            | PASSED  |
| Multi-turn memory working   | PASSED  |
| pH question retrieval       | CORRECT |
| Nutrient question retrieval | CORRECT |
| EC question retrieval       | CORRECT |
| Temperature retrieval       | CORRECT |

## Sample Questions Tested
1. What is the ideal pH range for hydroponic lettuce? → Answered with citation
2. What nutrients are most important for hydroponic plants? → Answered with citation
3. How does EC affect plant growth? → Answered with citation
4. What temperature should water be in hydroponics? → Answered with citation
5. How do I manage nutrient deficiency? → Answered with citation

## Known Issues / Limitations
- Gemini free tier quota was exhausted — switched to Groq (free, faster)
- HuggingFace hub sends unauthenticated warning — set HF_TOKEN in .env to fix
- models/faiss_index/ and docs/knowledge_base/ excluded from Git (large files)

## How to Run
1. Build vector store (first time only):
   python api/vector_store.py

2. Test RAG chain:
   python api/rag_chain.py

3. Start API server:
   uvicorn api.rag_api:app --reload --port 8000

4. Run tests:
   python tests/test_rag.py

5. Open interactive docs:
   http://localhost:8000/docs