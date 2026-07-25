"""
chat/llm.py
Provider-agnostic LLM client. Currently backed by Groq.
Swap providers by setting LLM_PROVIDER env var (default: groq).
All calls go through chat_complete() -- rest of codebase uses only this.
"""
from __future__ import annotations
import os, time
from typing import Any

from src.observability.tracer import record_llm_call
from src.observability.logger import log

_DEFAULT_MODEL = "qwen/qwen3.6-27b"


def chat_complete(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """
    Send messages to LLM and return the response text.
    Automatically records LLM telemetry in the current trace.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model    = model or os.getenv("LLM_MODEL", _DEFAULT_MODEL)

    if provider == "groq":
        return _groq_complete(messages, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: groq")


def _groq_complete(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )

    client  = Groq(api_key=api_key)
    prompt_preview = str(messages)[:300]
    t0 = time.time()
    error_msg = ""
    response_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        response = client.chat.completions.create(
            model       = model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        response_text     = response.choices[0].message.content or ""
        prompt_tokens     = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
    except Exception as e:
        error_msg = str(e)
        log.error("LLM call failed", extra={"error": error_msg, "model": model})
        raise

    latency = time.time() - t0
    record_llm_call(
        model             = model,
        prompt            = prompt_preview,
        response          = response_text,
        prompt_tokens     = prompt_tokens,
        completion_tokens = completion_tokens,
        latency_s         = latency,
        error             = error_msg,
    )
    return response_text
