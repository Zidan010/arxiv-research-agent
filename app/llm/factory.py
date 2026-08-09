"""
LLM provider factory.

Reads LLM_PROVIDER from Settings and constructs the corresponding provider.
This is the only place in the codebase that knows both concrete provider
classes exist -- everywhere else (the RAG synthesis layer) depends only on
the LLMProvider interface.

Fails fast: an unknown provider name or a missing API key raises
immediately at construction time, not on the first query a user makes.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMProvider, LLMProviderError
from app.llm.groq_provider import GroqProvider
# from app.llm.openai_provider import OpenAIProvider

_PROVIDERS = {
    "groq": lambda s: GroqProvider(api_key=s.GROQ_API_KEY, model=s.GROQ_MODEL),
    # "openai": lambda s: OpenAIProvider(api_key=s.OPENAI_API_KEY, model=s.OPENAI_MODEL),
}


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    provider_name = settings.LLM_PROVIDER.lower()

    builder = _PROVIDERS.get(provider_name)
    if builder is None:
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}'. "
            f"Valid options: {', '.join(_PROVIDERS)}."
        )

    return builder(settings)