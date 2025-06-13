
"""
embedding_utils.py

Utility functions for loading legal documents, chunking,
embedding, and indexing them using FAISS (Flat).

Used in RAG pipeline to support document retrieval.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Constants
PDF_FILE_PATH = "data/President_of_India.pdf"
INDEX_PATH = "data/faiss_index"
#EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME = "models/embedding-001"


def load_and_split_documents():
    """Loads and splits PDFs from the data folder."""
    all_docs = []

    loader = PyPDFLoader(PDF_FILE_PATH)
    docs = loader.load()
    all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(all_docs)
    return chunks


def create_and_save_vectorstore():
    """Embeds chunks and saves FAISS index to disk."""
    print("⚙️ Creating FAISS index from legal PDFs...")

    docs = load_and_split_documents()
    #embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_NAME)


    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"✅ FAISS index saved to: {INDEX_PATH}")
    return vectorstore


def load_vectorstore():
    """Loads FAISS vectorstore from disk or creates if not found."""
    #embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_NAME)


    if os.path.exists(INDEX_PATH):
        print("📦 Loading existing FAISS index...")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return create_and_save_vectorstore()