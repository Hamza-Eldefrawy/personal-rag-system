import os
from pathlib import Path
from dotenv import load_dotenv

# Project Root is 2 levels up from src/personal_rag
ROOT_DIR = Path(__file__).resolve().parents[2]

# Load .env from project root
load_dotenv(ROOT_DIR / ".env")

# Paths relative to Root Directory
DATA_DIR = ROOT_DIR / "data"
PDF_PATH = DATA_DIR / "HamzaEldafrawy.pdf"
GITHUB_DATA_DIR = DATA_DIR / "github"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"

# GitHub Configuration
GITHUB_USERNAME = "Hamza-Eldefrawy"

# Model Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openrouter/free"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Chunking & Retrieval Configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
RETRIEVER_K = 3

