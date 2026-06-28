from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    Environment variables are intentionally minimal so the project stays easy
    to run locally. External providers can later add their own API keys and
    settings without changing the evaluator core.
    """

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    request_timeout_seconds: int = 300


    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            request_timeout_seconds=int(
                os.getenv("REQUEST_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))
            ),
        )
