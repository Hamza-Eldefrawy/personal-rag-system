from langchain_core.documents import Document
from langchain_classic.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    FAISS_INDEX_PATH,
)


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> TokenTextSplitter:
    """Create a token-based text splitter."""
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_documents(
    documents: list[Document],
    splitter: TokenTextSplitter = None,
) -> list[Document]:
    """Split documents into chunks."""
    if splitter is None:
        splitter = get_text_splitter()
    return splitter.split_documents(documents)


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = EMBEDDING_DEVICE,
) -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )


def create_vector_db(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings = None,
) -> FAISS:
    """Generate embeddings for document chunks and create a FAISS vector store."""
    if embeddings is None:
        embeddings = get_embedding_model()
    return FAISS.from_documents(chunks, embeddings)


def save_vector_db(vector_db: FAISS, path=FAISS_INDEX_PATH):
    """Persist FAISS index to disk."""
    vector_db.save_local(str(path))


def load_vector_db(path=FAISS_INDEX_PATH, embeddings: HuggingFaceEmbeddings = None) -> FAISS:
    """Load FAISS index from disk."""
    if embeddings is None:
        embeddings = get_embedding_model()
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
