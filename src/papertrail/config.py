"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAPERTRAIL_",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://papertrail:papertrail@localhost:5432/papertrail"
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    arxiv_user_agent: str = Field(
        default="PaperTrail/0.1 (educational project; github.com/mghadia1)",
        min_length=10,
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    groq_api_key: SecretStr | None = None
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = "openai/gpt-oss-120b"
    abstain_threshold: float = 0.03239446668849102
    abstain_signal: str = "rrf_top"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
