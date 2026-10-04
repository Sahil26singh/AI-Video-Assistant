import uuid
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=f"video_{uuid.uuid4().hex}",   # unique per analysis (in-memory)
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
