from langchain_core.documents import Document
from config import RETRIEVER_K


def get_retriever(vector_db, k: int = RETRIEVER_K, search_type: str = "similarity", search_kwargs: dict = None):
    """Obtain a retriever interface from the FAISS vector database."""
    if search_kwargs is None:
        search_kwargs = {"k": k}
    return vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def retrieve_similar_documents(
    vector_db,
    query: str,
    k: int = RETRIEVER_K,
) -> list[tuple[Document, float]]:
    """Search for most similar documents with their similarity scores."""
    return vector_db.similarity_search_with_score(query, k=k)
