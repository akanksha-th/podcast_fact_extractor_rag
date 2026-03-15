from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pydantic import computed_field, field_validator
from pathlib import Path


class GROQSettings(BaseSettings):
    api_key: str
    groq_model: str = "llama3-8b-8192"
    model_version: str = "1"
    timeout_secs: int = 30
    retries: int = 2

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("GROQ API Key not found.")
        return v


class RedisSettings(BaseSettings):
    cache_url: str = "redis://redis:6379/0"
    min_pool_size: int = 5
    max_pool_size: int = 20
    session_ttl: int = 7200     # 2 hrs = 7200 seconds

    # ARQ WORKER
    arq_url: str = "redis://redis:6379/1"     # separate DB for job queue


class QdrantSettings(BaseSettings):
    host: str = "qdrant"
    port: int = 6333
    vector_size: int = 384

    # Chunks are stored in Qdrant
    chunk_size: int = 384
    chunk_overlap: int = 64


class PostgresSettings(BaseSettings):
    user: str = "postgres"
    password: str #postgres
    host: str = "postgres"
    port: int = 5432
    db_name: str = "podcast_rag"
    min_pool_size: int = 2
    max_pool_size: int = 10


class MlflowSettings(BaseSettings):
    tracking_url: str = "http://mlflow:5000"
    experiment_name: str = "podcast-rag"


class Settings(BaseSettings):
    groq: GROQSettings = GROQSettings()
    redis: RedisSettings = RedisSettings()
    qdrant: QdrantSettings = QdrantSettings()
    postgres: PostgresSettings = PostgresSettings()
    mlflow: MlflowSettings = MlflowSettings()
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    # APP
    app_env: str = "development"    # development | production
    log_file_path: Path = Path("/var/log/podcast_rag")
    log_level: str = "INFO"

    # EMBEDDINGS
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size = 10

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

    
    @computed_field
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant.host}:{self.qdrant.port}"
    
    @computed_field
    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.postgres.user}:{self.postgres.password}@{self.postgres.host}:{self.postgres.port}/{self.postgres.db_name}"

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def api_settings() -> Settings:
    return Settings()
