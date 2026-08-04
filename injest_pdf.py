"""
One-time (or repeat-whenever-the-PDF-changes) ingestion script.
Loads a PDF, splits it into chunks, embeds them via the Hugging Face
Inference API (no local model download), and upserts into Pinecone.

pip install pinecone langchain-pinecone langchain-community langchain-text-splitters \
            langchain-huggingface pypdf python-dotenv

Needs PINECONE_API_KEY and HUGGINGFACEHUB_API_TOKEN in your .env
(get a free Pinecone key at https://www.pinecone.io).
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
INDEX_NAME = "ai-agent"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, called via HF Inference API
EMBEDDING_DIM = 384


def ingest_pdf(pdf_path: str, clear_existing: bool = True) -> None:
    print(f"Loading {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} page(s).")

    # This PDF packs each doctor/department/symptom entry as a dense block
    # separated by blank lines. Splitting at 800 chars was slicing a doctor's
    # name away from their schedule, or a symptom away from its routing note.
    # Bigger chunks + separators that prefer blank-line boundaries keep each
    # entry (doctor, department, symptom row) intact as one retrievable unit.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    elif clear_existing:
        # Without this, re-running ingestion after a chunking change just
        # ADDS new vectors alongside the old, worse-chunked ones -- so both
        # versions get retrieved and the old ones add noise. Clear first.
        print(f"Clearing existing vectors from '{INDEX_NAME}' before re-ingesting ...")
        pc.Index(INDEX_NAME).delete(delete_all=True)

    print("Upserting chunks into Pinecone (calls the HF embedding API for each chunk --")
    print("first call may take ~10-20s while the model spins up on HF's end)...")
    PineconeVectorStore.from_documents(chunks, embedding=embeddings, index_name=INDEX_NAME)
    print(f"Done -- {len(chunks)} chunks from '{pdf_path}' are now searchable in Pinecone.")


if __name__ == "__main__":
    # Change this to the actual path of your hospital PDF.
    ingest_pdf("KLIMS_RAG_KnowledgeBase.pdf")