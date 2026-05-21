from fileinput import filename
import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress PDF warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

KNOWLEDGE_BASE_DIR = "docs/knowledge_base"

def load_all_documents():
    """Load all PDFs and text files from the knowledge base folders."""
    all_documents = []
    
    for root, dirs, files in os.walk(KNOWLEDGE_BASE_DIR):
        for filename in files:
            filepath = os.path.join(root, filename)
            topic = Path(root).name  # folder name = topic
            
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                elif filename.endswith(".txt"):
                    loader = TextLoader(filepath, encoding="utf-8")
                    docs = loader.load()
                else:
                    continue
                
                # Tag each document with metadata
                for i, doc in enumerate(docs):
                    doc.metadata["topic"] = topic
                    doc.metadata["source_file"] = filename
                    doc.metadata["page"] = i + 1
                    doc.metadata["document_type"] = "pdf" if filename.endswith(".pdf") else "text"
                
                all_documents.extend(docs)
                print(f"  Loaded: {filename} ({len(docs)} pages) [{topic}]")
                
            except Exception as e:
                print(f"  Could not load {filename}: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)} pages")
    return all_documents


def split_documents(documents):
    """Split documents into chunks of 1000 tokens with 120 overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_all_documents()
    
    print("\nSplitting into chunks...")
    chunks = split_documents(docs)
    
    print(f"\nSample chunk:")
    print(chunks[0].page_content[:300])
    print(f"\nMetadata: {chunks[0].metadata}")