import sys
sys.path.append(".")
from api.rag_chain import ask_question, chat_history

project_questions = [
    "What pH range should Farmspherica target for their hydroponic system?",
    "What TDS level is ideal for the crops being grown?",
    "What sensors are needed for the Nano PAW monitoring system?",
    "What nutrient level should we observe for healthy lettuce growth?",
    "How often should the nutrient solution be changed in the system?",
]

print("=== FARMSPHERICA PROJECT-SPECIFIC RAG TEST ===\n")
chat_history.clear()

for i, q in enumerate(project_questions, 1):
    result = ask_question(q)
    print(f"Q{i}: {q}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Sources: {result['sources'][:2]}")
    print("-" * 60)

print("\nProject-specific test complete!")