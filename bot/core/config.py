from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, computed_field
from functools import lru_cache

class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    bot_token: str
    webhook_base_url: str = ""
    webhook_path: str = "/webhook/telegram"

    # API endpoints
    ingest_endpoint: str = "/api/v1/ingest"
    query_endpoint: str = "/api/v1/query"
    notes_endpoint: str = "/api/v1/notes"

    @field_validator("bot_token")
    @classmethod
    def bot_token_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Telegram Bot Token cannot be empty.")
        return v
    
    @computed_field
    @property
    def webhook_url(self) -> str:
        return f"{self.webhook_base_url}{self.webhook_path}"


@lru_cache
def bot_settings() -> TelegramSettings:
    return TelegramSettings()