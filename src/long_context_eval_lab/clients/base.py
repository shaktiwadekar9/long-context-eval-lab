from __future__ import annotations

from abc import ABC, abstractmethod
from long_context_eval_lab.schemas import ModelResponse


class ModelClient(ABC):
    """Provider-neutral interface for model calls.

    The evaluator only depends on this interface. To add OpenAI, Groq,
    OpenRouter, vLLM, SGLang, or Gemini later, create another class that
    implements `generate()` and returns `ModelResponse`.
    """

    provider_name: str
    model_name: str

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        raise NotImplementedError
