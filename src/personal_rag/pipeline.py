from pathlib import Path
from dotenv import load_dotenv

from config import (
    PDF_PATH,
    GITHUB_USERNAME,
    GITHUB_DATA_DIR,
    FAISS_INDEX_PATH,
    LLM_MODEL,
    LLM_TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
)
from data_load import load_all_documents
from chunking_embedding import (
    split_documents,
    get_text_splitter,
    get_embedding_model,
    create_vector_db,
    save_vector_db,
    load_vector_db,
)
from data_retrieval import get_retriever
from chain import (
    get_llm,
    get_prompt_template,
    get_conversation_memory,
    build_conversational_chain,
)

load_dotenv()


class PersonalRAG:
    """End-to-end Personal RAG Pipeline."""

    def __init__(
        self,
        pdf_path=PDF_PATH,
        github_username: str = GITHUB_USERNAME,
        github_data_dir=GITHUB_DATA_DIR,
        index_path=FAISS_INDEX_PATH,
        llm_model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.pdf_path = Path(pdf_path)
        self.github_username = github_username
        self.github_data_dir = Path(github_data_dir)
        self.index_path = Path(index_path)
        self.llm_model = llm_model
        self.temperature = temperature

        self.vector_db = None
        self.chain = None

    def build(self, force_reindex: bool = False):
        """Build or load the vector database and initialize the conversational chain."""
        if self.index_path.exists() and not force_reindex:
            print(f"Loading existing FAISS index from {self.index_path.resolve()}...")
            self.vector_db = load_vector_db(self.index_path)
        else:
            print("Loading documents...")
            documents = load_all_documents(
                pdf_path=str(self.pdf_path),
                github_username=self.github_username,
                github_data_dir=str(self.github_data_dir),
            )
            print(f"Loaded {len(documents)} raw documents.")

            print("Splitting documents into chunks...")
            splitter = get_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = split_documents(documents, splitter)
            print(f"Created {len(chunks)} chunks.")

            print("Embedding chunks and building FAISS vector store...")
            embeddings = get_embedding_model()
            self.vector_db = create_vector_db(chunks, embeddings)

            save_vector_db(self.vector_db, self.index_path)
            print(f"FAISS index saved to {self.index_path.resolve()}.")

        retriever = get_retriever(self.vector_db, k=RETRIEVER_K)
        llm = get_llm(model=self.llm_model, temperature=self.temperature)
        prompt = get_prompt_template()
        memory = get_conversation_memory()

        self.chain = build_conversational_chain(
            llm=llm,
            retriever=retriever,
            prompt=prompt,
            memory=memory,
        )
        print("Personal RAG system is ready.")
        return self

    def ask(self, question: str) -> dict:
        """Query the conversational chain."""
        if self.chain is None:
            raise RuntimeError("Pipeline has not been built yet. Call .build() first.")

        response = self.chain.invoke({"question": question})
        return {
            "answer": response.get("answer", ""),
            "source_documents": response.get("source_documents", []),
        }


def main():
    rag = PersonalRAG().build()

    print("\n--- Testing Personal RAG ---")
    q1 = "What is the personal information of Hamza Eldafrawy?"
    print(f"\nUser: {q1}")
    res1 = rag.ask(q1)
    print(f"Assistant: {res1['answer']}")

    q2 = "I want to know what does Hamza specialize in?"
    print(f"\nUser: {q2}")
    res2 = rag.ask(q2)
    print(f"Assistant: {res2['answer']}")


if __name__ == "__main__":
    main()
