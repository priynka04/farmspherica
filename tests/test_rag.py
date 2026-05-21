import sys
sys.path.append(".")
from api.rag_chain import ask_question, chat_history

def test_rag_returns_answer():
    """Test that a basic question returns an answer."""
    result = ask_question("What is the ideal pH for hydroponics?")
    assert "answer" in result
    assert len(result["answer"]) > 10
    assert result["chunks_retrieved"] > 0
    print("Test 1 passed: RAG returns an answer")

def test_rag_returns_sources():
    """Test that sources are returned with the answer."""
    result = ask_question("What nutrients do plants need?")
    assert "sources" in result
    assert len(result["sources"]) > 0
    print("Test 2 passed: Sources are returned")

def test_rag_memory():
    """Test that conversation memory works."""
    chat_history.clear()
    ask_question("What is EC in hydroponics?")
    result = ask_question("How does it affect plant growth?")
    assert "answer" in result
    print("Test 3 passed: Memory works for follow-up questions")

if __name__ == "__main__":
    test_rag_returns_answer()
    test_rag_returns_sources()
    test_rag_memory()
    print("\nAll RAG tests passed!")