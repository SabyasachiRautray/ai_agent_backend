"""
Quick text-only test script -- no mic, no voice, just checks retrieval and
answers directly in the terminal. Use this to tune RELEVANCE_THRESHOLD in
rag.py before wiring things into the full voice loop.

Run: python query_hospital.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag import HospitalRagChain

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.3,
    max_tokens=512,
)

rag_chain = HospitalRagChain(llm)

print("Type a question (or 'quit' to exit). Try both hospital-specific and general questions.\n")

while True:
    query = input("You: ").strip()
    if query.lower() in {"quit", "exit"}:
        break
    if not query:
        continue

    # Peek at raw retrieval scores -- helpful for tuning RELEVANCE_THRESHOLD
    results = rag_chain.vectorstore.similarity_search_with_score(query, k=4)
    print("  [scores]", [round(score, 3) for _doc, score in results])

    answer = rag_chain.invoke({"query": query, "language": "English"})
    print(f"JARVIS: {answer}\n")