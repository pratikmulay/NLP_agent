"""
LLM Provider abstraction — supports Ollama, Groq, and OpenAI backends.

Selected via the ``LLM_PROVIDER`` environment variable.
Provides ``summarize()`` and ``classify()`` methods.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Singleton ────────────────────────────────────────────────────────────────

_provider_instance = None


def get_llm_provider() -> "LLMProvider":
    """Return the cached LLM provider singleton."""
    global _provider_instance
    if _provider_instance is None:
        settings = get_settings()
        provider_name = settings.LLM_PROVIDER.lower()
        logger.info("Initializing LLM provider: %s", provider_name)

        if provider_name == "ollama":
            _provider_instance = OllamaProvider(settings)
        elif provider_name == "groq":
            _provider_instance = GroqProvider(settings)
        elif provider_name == "openai":
            _provider_instance = OpenAIProvider(settings)
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider_name}")
    return _provider_instance


# ── Base class ───────────────────────────────────────────────────────────────

class LLMProvider:
    """Abstract base for LLM providers."""

    async def summarize(self, text: str, mode: str = "chunk") -> str:
        raise NotImplementedError

    async def classify(self, text: str, labels: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    def _build_summarize_prompt(self, text: str, mode: str) -> str:
        if mode == "final":
            return (
                "You are given multiple summaries of document chunks. "
                "Produce a single coherent summary that captures all key points.\n\n"
                f"Chunk summaries:\n{text}\n\n"
                "Final summary:"
            )
        return (
            "Summarize the following text concisely, capturing the main points:\n\n"
            f"{text}\n\nSummary:"
        )

    def _build_classify_prompt(self, text: str, labels: list[str]) -> str:
        labels_str = ", ".join(f'"{l}"' for l in labels)
        return (
            f"Classify the following text into exactly one of these labels: [{labels_str}].\n\n"
            f"Text: {text}\n\n"
            "Respond with ONLY a JSON object: {\"label\": \"chosen_label\", \"confidence\": 0.0-1.0}"
        )

    def _parse_classify_response(self, response: str, text: str, labels: list[str]) -> dict[str, Any]:
        """Parse the LLM classification response into structured output."""
        try:
            # Try direct JSON parse
            result = json.loads(response.strip())
            return {
                "text": text[:100],
                "label": result.get("label", labels[0]),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback: check if any label appears in the response
            response_lower = response.lower()
            for label in labels:
                if label.lower() in response_lower:
                    return {"text": text[:100], "label": label, "confidence": 0.5}
            return {"text": text[:100], "label": labels[0], "confidence": 0.1}


# ── Ollama ───────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    def __init__(self, settings):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    async def _generate(self, prompt: str) -> str:
        import ollama as ollama_lib
        client = ollama_lib.AsyncClient(host=self.base_url)
        response = await client.generate(model=self.model, prompt=prompt)
        return response["response"]

    async def summarize(self, text: str, mode: str = "chunk") -> str:
        prompt = self._build_summarize_prompt(text, mode)
        return await self._generate(prompt)

    async def classify(self, text: str, labels: list[str]) -> dict[str, Any]:
        prompt = self._build_classify_prompt(text, labels)
        response = await self._generate(prompt)
        return self._parse_classify_response(response, text, labels)


# ── Groq ─────────────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    def __init__(self, settings):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    async def _generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def summarize(self, text: str, mode: str = "chunk") -> str:
        prompt = self._build_summarize_prompt(text, mode)
        return await self._generate(prompt)

    async def classify(self, text: str, labels: list[str]) -> dict[str, Any]:
        prompt = self._build_classify_prompt(text, labels)
        response = await self._generate(prompt)
        return self._parse_classify_response(response, text, labels)


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    def __init__(self, settings):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    async def _generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    async def summarize(self, text: str, mode: str = "chunk") -> str:
        prompt = self._build_summarize_prompt(text, mode)
        return await self._generate(prompt)

    async def classify(self, text: str, labels: list[str]) -> dict[str, Any]:
        prompt = self._build_classify_prompt(text, labels)
        response = await self._generate(prompt)
        return self._parse_classify_response(response, text, labels)
