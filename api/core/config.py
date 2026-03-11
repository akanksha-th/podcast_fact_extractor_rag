from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # APP
    app_env: str = "development"    # development | production
    log_levl: str = "INFO"

    # GROQ
    groq_api_key: str
    groq_model: str = "llama3-8b-8192"
    groq_timeout_secs: int = 30
    groq_retries: int = 2

    # TELEGRAM
    telegram_bot_token: str
    webhook_base_url: str = ""
    webhook_path: str = "/webhook/telegram"

    # REDIS
    redis_url: str = "redis://redis:6379/0"
    redis_pool_min_size: int = 5
    redis_pool_max_size: int = 20
    session_ttl_sec: int = 7200     # 2 hrs

    # POSTGRES
    postgres_url: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/podcast_rag"

    # QDRANT
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    # EMBEDDINGS
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # CHUNKING
    chunk_size: int = 384
    chunk_overlap: int = 64

    # RETRIEVAL
    retrieval_top_k: int = 5

    # RATE LIMITS
    max_videos_per_user_per_day: int = 10
    max_notes_per_video_per_model: int = 2

    # INPUT VALIDATION
    max_question_chars: int = 500

    # NOTES GENERATION
    notes_chunk_batch_size: int = 10
    notes_min_words: int = 500
    notes_max_words: int = 5000

    # MLFLOW
    mlflow_tracking_url: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "podcast-rag"

    # ARQ WORKER
    arq_redis_url: str = "redis://redis:6379/1"     # separate DB for job queue


@lru_cache
def get_settings() -> Settings:
    return Settings()