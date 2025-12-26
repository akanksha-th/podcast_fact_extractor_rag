"""
Configuration Management System
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings configuration – Override defaults with a .env file
    """

    # ---- Project Paths ----
    project_root: Path = Path(__file__).parent
    models_dir: Path = project_root / "models"
    outputs_dir: Path = project_root / "outputs"
    logs_dir: Path = project_root / "logs"
    checkpoints_dir: Path = project_root / "checkpoints"

    # ---- Chunking Settings ----
    chunk_size: int = 800
    chunk_overlap: int = 200
    chunk_separators: List[str] = ["\n\n", "\n", ".", " ", ""]
    min_chunk_size: int = 200

    # ---- Embedding Settings ----
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_batch_size: int = 32

    # ---- LLM Settings ----
    llm_model_path: str = "./models/phi-2.Q4_0.gguf"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 128
    llm_top_k: int = 40
    llm_top_p: float = 0.9
    llm_repeat_penalty: float = 1.1
    llm_verbose: bool = False

    # ---- Notes Generation LLM Settings ----
    chunk_notes_max_tokens: int = 64
    section_notes_max_tokens: int = 96

    # ---- Retrieval Settings ----
    top_k_retrieval: int = 5
    min_relevance_score: float = 0.3
    max_context_length: int = 4000

    # ---- Qdrant Settings ----
    qdrant_collection: str = "podcast_101"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_vector_size: int = 768 # for multilingual-e5-base

    # ---- Notes Settings ----
    chunk_notes_max_words: int = 12
    section_batch_size: int = 4
    save_intermediate_notes: bool = True

    # ---- Transcription Settings ----
    max_transcript_length: int = 1_000_000
    min_transcript_length: int = 100

    # ---- Input Validation ----
    max_query_length: int = 500
    valid_tasks: List[str] = ["1", "2"]

    # ---- Logging Settings ----
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"

    # ---- Performance Settings ----
    async_batch_size: int = 5
    enable_caching: bool = True
    checkpoint_after_embedding: bool = True

    # ---- Error Handling ----
    max_retries: int = 3
    retry_delay_seconds: int = 2


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for dir_path in [self.outputs_dir, self.logs_dir, self.checkpoints_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def validate_model_exists(self) -> bool:
        return Path(self.llm_model_path).exists()
    
    def get_log_file_path(self) -> Path:
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        return self.logs_dir / f"agent_{today}.log"
    
    def get_notes_path(self, timestamp: str) -> tuple:
        chunk_path = self.outputs_dir / f"chunk_notes_{timestamp}.md"
        section_path = self.outputs_dir / f"section_notes_{timestamp}.md"
        return chunk_path, section_path
    
settings = Settings()

if __name__ == "__main__":
    print("Configuration Settings")
    print("=" * 60)
    print(f"Chunk Size: {settings.chunk_size}")
    print(f"Chunk Overlap: {settings.chunk_overlap}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"LLM Model: {settings.llm_model_path}")
    print(f"LLM Model Exists: {settings.validate_model_exists()}")
    print(f"Qdrant Collection: {settings.qdrant_collection}")
    print(f"Retrieval Top K: {settings.top_k_retrieval}")
    print(f"Min Relevance Score: {settings.min_relevance_score}")
    print(f"Logs Directory: {settings.logs_dir}")
    print(f"Outputs Directory: {settings.outputs_dir}")
    print("=" * 60)
    print("\nConfiguration loaded successfully!")