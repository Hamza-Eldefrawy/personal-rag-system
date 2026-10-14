from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from config import RETRIEVER_K


class CVEnrichedRetriever(BaseRetriever):
    """Custom retriever that always includes CV/PDF chunks alongside similarity results.

    Ensures the person's resume is always part of the retrieved context,
    regardless of the query, so the LLM always has access to core personal information.
    """

    vector_db: Any
    k: int = RETRIEVER_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> list[Document]:
        # 1. Standard similarity search
        similar_docs = self.vector_db.similarity_search(query, k=self.k)

        # 2. Always include CV/PDF chunks from the docstore
        cv_docs = [
            doc
            for doc in self.vector_db.docstore._dict.values()
            if "pdf" in doc.metadata.get("source", "").lower()
        ]

        # 3. Merge: CV docs first, then similarity results, deduplicated
        seen_texts = set()
        result = []
        for doc in cv_docs + similar_docs:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                result.append(doc)

        return result


def get_retriever(vector_db, k: int = RETRIEVER_K):
    """Obtain a CVEnrichedRetriever that always includes CV documents."""
    return CVEnrichedRetriever(vector_db=vector_db, k=k)


def retrieve_similar_documents(
    vector_db,
    query: str,
    k: int = RETRIEVER_K,
) -> list[tuple[Document, float]]:
    """Search for most similar documents with their similarity scores."""
    return vector_db.similarity_search_with_score(query, k=k)
