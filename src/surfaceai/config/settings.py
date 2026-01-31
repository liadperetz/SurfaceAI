"""Application settings with .env support."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SURFACEAI_",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"

    # OpenAI
    openai_api_key: Optional[str] = None

    # Groq
    groq_api_key: Optional[str] = None

    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"

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
