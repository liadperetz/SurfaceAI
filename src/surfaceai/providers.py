"""LLM provider abstraction using OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Literal, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

from surfaceai.config.settings import get_settings

ProviderName = Literal["ollama", "openai", "groq", "deepseek"]

# Default configurations per provider
PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # Ollama doesn't require a real key
        "default_model": "llama3.1:8b",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
}


class LLMClient:
    """Unified LLM client that works with any OpenAI-compatible provider."""

    def __init__(
        self,
        provider: ProviderName,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        config = PROVIDER_DEFAULTS[provider]
        self.model = model or config["default_model"]

        # Resolve base URL
        effective_base_url = base_url or config.get("base_url")

        # Resolve API key
        if api_key:
            effective_api_key = api_key
        elif "api_key" in config:
            effective_api_key = config["api_key"]
        else:
            # Load from settings/environment
            settings = get_settings()
            key_map = {
                "openai": settings.openai_api_key,
                "groq": settings.groq_api_key,
                "deepseek": settings.deepseek_api_key,
            }
            effective_api_key = key_map.get(provider)
            if not effective_api_key:
                raise ValueError(
                    f"API key for {provider} not found. "
                    f"Set {config['api_key_env']} environment variable or pass api_key parameter."
                )

        self._client = OpenAI(base_url=effective_base_url, api_key=effective_api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send chat completion request and return the response content."""
        extra_kwargs = {}
        if self.provider == "ollama":
            extra_kwargs["extra_body"] = {"think": False}

        for attempt in range(1, self.max_retries + 1):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                **extra_kwargs,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            if attempt < self.max_retries:
                logger.warning(
                    "Empty response from %s (attempt %d/%d), retrying...",
                    self.model, attempt, self.max_retries,
                )
        logger.warning("Empty response from %s after %d attempts", self.model, self.max_retries)
        return ""

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"


def create_client(
    provider: ProviderName,
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create an LLM client."""
    return LLMClient(provider=provider, model=model, **kwargs)
