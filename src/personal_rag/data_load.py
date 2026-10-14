import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from config import PDF_PATH, GITHUB_DATA_DIR, GITHUB_USERNAME


def load_pdf(pdf_path: str = PDF_PATH) -> list[Document]:
    """Load pages from a PDF file."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path.resolve()}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    return pages


def load_saved_github_documents(
    username: str = GITHUB_USERNAME,
    data_dir: str = GITHUB_DATA_DIR,
) -> list[Document]:
    """Load saved GitHub JSON data into LangChain Documents."""
    root = Path(data_dir) / username
    documents = []

    # 1. Load profile
    profile_path = root / "profile.json"

    if profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))

        profile_text = "\n".join(
            f"{key}: {value}"
            for key, value in profile.items()
            if value is not None
        )

        documents.append(
            Document(
                page_content=profile_text,
                metadata={
                    "source": profile.get("html_url", ""),
                    "type": "github_profile",
                    "username": profile.get("login", username),
                },
            )
        )

    # 2. Load repositories
    repos_dir = root / "repositories"

    if repos_dir.exists():
        for repo_path in repos_dir.glob("*.json"):
            data = json.loads(repo_path.read_text(encoding="utf-8"))
            repo = data["metadata"]

            repo_name = repo.get("full_name", repo_path.stem)
            repo_url = repo.get("html_url", "")

            # Repository metadata
            metadata_text = "\n".join([
                f"Repository: {repo_name}",
                f"Description: {repo.get('description') or 'N/A'}",
                f"Language: {repo.get('language') or 'N/A'}",
                f"Topics: {', '.join(repo.get('topics') or [])}",
                f"Stars: {repo.get('stargazers_count', 0)}",
                f"URL: {repo_url}",
            ])

            documents.append(
                Document(
                    page_content=metadata_text,
                    metadata={
                        "source": repo_url,
                        "type": "repository_metadata",
                        "repository": repo_name,
                    },
                )
            )

            # README
            readme = data.get("readme")

            if readme and readme.get("content"):
                documents.append(
                    Document(
                        page_content=readme["content"],
                        metadata={
                            "source": readme.get("url", repo_url),
                            "type": "readme",
                            "repository": repo_name,
                            "file_path": readme.get("path", "README"),
                        },
                    )
                )

            # Source files
            for file in data.get("source_files", []):
                if not file.get("content"):
                    continue

                documents.append(
                    Document(
                        page_content=file["content"],
                        metadata={
                            "source": file.get("url", repo_url),
                            "type": "source_code",
                            "repository": repo_name,
                            "file_path": file.get("path", ""),
                        },
                    )
                )

    return documents


def load_all_documents(
    pdf_path: str = PDF_PATH,
    github_username: str = GITHUB_USERNAME,
    github_data_dir: str = GITHUB_DATA_DIR,
) -> list[Document]:
    """Load both PDF and GitHub documents."""
    all_docs = []

    try:
        pdf_docs = load_pdf(pdf_path)
        all_docs.extend(pdf_docs)
    except FileNotFoundError:
        print(f"Warning: PDF file not found at {pdf_path}")

    github_docs = load_saved_github_documents(github_username, github_data_dir)
    all_docs.extend(github_docs)

    return all_docs
