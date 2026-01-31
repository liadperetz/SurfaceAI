"""Application settings with .env support."""

from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SURFACEAI_",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "SURFACEAI_OLLAMA_BASE_URL"),
    )

    # OpenAI
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "SURFACEAI_OPENAI_API_KEY"),
    )

    # Groq
    groq_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "SURFACEAI_GROQ_API_KEY"),
    )

    # DeepSeek
    deepseek_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "SURFACEAI_DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "SURFACEAI_DEEPSEEK_BASE_URL"),
    )

    # Output
    default_out_dir: str = "runs"

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment."""
        return cls()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
