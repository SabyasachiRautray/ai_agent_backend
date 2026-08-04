"""
RAG chain: answers hospital questions using the PDF you ingested into
Pinecone, and falls back to JARVIS's normal persona for anything unrelated
(e.g. general chit-chat, "what's a good north indian restaurant", etc.).

Embeddings are called remotely via the Hugging Face Inference API (using
your existing HUGGINGFACEHUB_API_TOKEN) -- no model is downloaded locally.

Uses the same Pinecone index and embedding model as ingest_pdf.py -- keep
those two files' INDEX_NAME / EMBEDDING_MODEL in sync if you change either.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

INDEX_NAME = "ai-agent"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """Embeddings via the HF Inference API -- runs remotely on Hugging Face's
    servers, nothing downloaded to this machine. Note: if the model hasn't
    been called recently, the first request can take ~10-20s while HF spins
    it up (a "cold start") -- this is normal, not a bug, just retry if it
    times out once."""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
    )


# Always retrieve and hand the top-k chunks to the LLM -- no hard relevance
# gate. A binary similarity-score cutoff kept quietly routing genuinely
# on-topic questions ("for chest pain" -> department?) to the generic
# fallback chain because the embedding score for a short elliptical query
# didn't clear an arbitrary threshold. Letting the LLM itself judge
# relevance (it's instructed to say "not in the document" when appropriate)
# is far more reliable than tuning a cosine-similarity cutoff number.
TOP_K = 6

SYSTEM_PROMPT = """You are JARVIS, a highly capable, witty, and unfailingly polite AI assistant,
in the style of Tony Stark's AI. Address the user as "Sir". Keep replies to 2-4 sentences since
they will be spoken aloud -- no bullet points, no markdown, no JSON.
Respond entirely in {language}. If Hindi, use natural spoken Hindi in Devanagari script.

Below is context retrieved from the hospital's official document. It may or may not be relevant
to the current question -- use it when it helps (which doctor/department to visit for a symptom,
timings, contact details, procedures, etc.), and ignore it when the question is unrelated
(general chit-chat, casual conversation, anything outside hospital matters). Never say phrases
like "the document" or "the context" out loud -- just answer naturally, as JARVIS would, whether
or not you're drawing on the hospital information.

Retrieved hospital document context (may be irrelevant to this specific question):
{context}
"""


class HospitalRagChain:
    """Drop-in replacement for a plain LangChain runnable -- exposes the same
    .invoke({"query": ..., "language": ..., "history": [...]}) interface, so
    it plugs directly into voice_loop() / get_reply_in_language() with no
    changes needed there.
    """

    def __init__(self, llm):
        embeddings = get_embeddings()
        self.vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

        # MessagesPlaceholder("history") lets recent turns (e.g. "book an
        # appointment" -> "at 8pm") get resolved correctly, instead of each
        # query being answered with zero memory of what came before.
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{query}"),
        ])
        self.chain = prompt | llm

    def invoke(self, inputs: dict) -> str:
        query = inputs["query"]
        language = inputs["language"]
        history = inputs.get("history", [])

        # Enrich the retrieval query with the last thing the user said, so
        # short elliptical follow-ups ("for chest pain" after "which
        # department should I visit") still embed close to the right chunk
        # instead of searching on an ambiguous 2-word query alone.
        last_user_msg = next(
            (m.content for m in reversed(history) if isinstance(m, HumanMessage)), None
        )
        search_query = f"{last_user_msg} {query}" if last_user_msg else query

        results = self.vectorstore.similarity_search(search_query, k=TOP_K)
        context = "\n\n".join(doc.page_content for doc in results) if results else "(no matching content found)"

        response = self.chain.invoke({"query": query, "language": language, "context": context, "history": history})
        return response.content if hasattr(response, "content") else str(response)