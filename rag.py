from policy_loader import load_policy_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

CHROMA_DIR = "./chroma_db"

def build_vector_store():
    documents = load_policy_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Vector store built, chunk count:{len(chunks)}, stored at {CHROMA_DIR}")
    return db

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_DIR):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        print("Loaded existing Chroma vector store")
        return db
    else:
        return build_vector_store()

if __name__ == "__main__":
    get_vector_store()
