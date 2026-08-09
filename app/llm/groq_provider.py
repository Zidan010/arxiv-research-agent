"""
Groq provider implementation.

Groq is the default synthesis provider for this project (fast inference,
generous free tier). It does not offer an embeddings endpoint, which is why
embedding is handled entirely separately by a local sentence-transformers
model (see app/embeddings) -- this provider is only used for the final
answer-synthesis step.
"""

from __future__ import annotations

import logging

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.llm.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMProviderError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and set it in your .env file."
            )
        self._client = Groq(api_key=api_key)
        self._model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _call(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        try:
            return self._call(system_prompt, user_prompt, max_tokens, temperature)
        except Exception as exc:  # groq's SDK raises several distinct exception types
            logger.error("Groq generation failed after retries: %s", exc)
            raise LLMProviderError(f"Groq generation failed: {exc}") from exc