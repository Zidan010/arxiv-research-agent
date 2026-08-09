"""
LLM provider interface.

Kept abstract so the synthesis layer depends on this interface,
not on Groq or OpenAI specifically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """
    Raised for both configuration problems (missing API key) and runtime
    failures (API error after retries exhausted). Callers (the RAG
    synthesis layer) catch this single type rather than needing to know
    about provider-specific exception classes.
    """


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        Generates a completion given a system prompt (instructions) and a
        user prompt (the actual query + retrieved context). Returns the
        raw text response.

        Raises LLMProviderError if the request fails after retries.
        """