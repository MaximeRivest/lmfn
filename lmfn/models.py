"""What a model can do, as lmcc capability facts (lmcc capabilities.md).

lmcc never sniffs capabilities: someone declares them. lmfn declares them
from the lm15 route (which provider serves the model) and a small table
of model-name prefixes, because lm15's model information does not yet
say whether a model calls tools natively or honors stop sequences.
Every guess here is overridable per function:
``@lmfn.ai(capabilities={"native_reasoning": False})``.
"""

from __future__ import annotations

# providers whose API has native tool calling and stop sequences (lm15 providers)
NATIVE_PROVIDERS = {"openai", "anthropic", "gemini", "xai"}

# model-name prefixes with an API-level thinking channel lm15 can request
_REASONING_PREFIXES = {
    "openai": ("o1", "o3", "o4", "gpt-5"),
    "anthropic": ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4-5", "claude-3-7-sonnet"),
    "gemini": ("gemini-2.5", "gemini-3"),
    "xai": ("grok-3-mini", "grok-4"),
}


def capabilities(provider: str, model: str) -> dict:
    """The capability facts lmfn declares for ``model`` served by ``provider``."""
    caps = {"instruct": True}
    if provider in NATIVE_PROVIDERS:
        caps["native_function_calling"] = True
        # lm15's `openai` provider speaks the Responses API, which has no
        # `stop` (lm15 refuses it loudly; found live 2026-09-23)
        caps["stop_sequences"] = provider != "openai"
        caps["native_reasoning"] = model.startswith(_REASONING_PREFIXES.get(provider, ()))
    else:
        # OpenAI-compatible hosts (OpenRouter, Groq, DeepSeek, ...) vary by
        # model: text tool calls and tags work everywhere.
        caps["stop_sequences"] = True
    return caps
