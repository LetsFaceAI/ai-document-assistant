from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Define fields with their expected types and default values
    app_name: str = "AI Document Assistant"
    port: int = 8000
    debug_mode: bool = False
    openrouter_api_key: str = "FakeKey123"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_base_model: str = "Model"
    environment: str ="dev"
    BASE_DIR: Path = PROJECT_ROOT
    # Default to <PROJECT_ROOT>/storage if not set in .env
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"

    # Chunking Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 50

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 32

    VECTOR_DB_PROVIDER: str = "chromadb"
    VECTOR_DB_PATH: str = "storage/chromadb"
    VECTOR_COLLECTION: str = "documents"

    # Retrieval Settings
    RETRIEVAL_TOP_K: int = 10
    MIN_SIMILARITY_SCORE: float = 0.50

    # Tell pydantic where to find the .env files
    model_config = SettingsConfigDict(
    env_file=BACKEND_DIR / ".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


settings = Settings()