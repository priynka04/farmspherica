import os
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from document_loader import load_all_documents, split_documents

load_dotenv()

VECTOR_STORE_PATH = "models/faiss_index"

def build_vector_store():
    """Load documents, create embeddings, save FAISS index."""
    
    print("Step 1: Loading documents...")
    documents = load_all_documents()
    
    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(documents)
    
    print("\nStep 3: Creating embeddings (this takes a few minutes)...")
    # Using HuggingFace (free, no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("\nStep 4: Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print("\nStep 5: Saving vector store to disk...")
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    print(f"\nVector store saved to: {VECTOR_STORE_PATH}")
    print(f"Total chunks indexed: {len(chunks)}")
    return vector_store


def load_vector_store():
    """Load existing FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded successfully.")
    return vector_store


def test_retrieval(vector_store):
    """Test with sample queries."""
    test_queries = [
        "What is the ideal pH for hydroponics?",
        "What are the best nutrients for lettuce?",
        "How does EC affect plant growth?",
        "What temperature is best for hydroponic systems?",
    ]
    
    print("\n--- RETRIEVAL TEST ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(query, k=3)
        print(f"Top result from: {results[0].metadata.get('source_file', 'unknown')}")
        print(f"Topic: {results[0].metadata.get('topic', 'unknown')}")
        print(f"Content preview: {results[0].page_content[:200]}...")


if __name__ == "__main__":
    # Build and save the vector store
    vs = build_vector_store()
    
    # Test it
    test_retrieval(vs)