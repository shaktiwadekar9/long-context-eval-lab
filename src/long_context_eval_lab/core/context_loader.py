from __future__ import annotations

from pathlib import Path


def load_context(path: str | Path) -> str:
    """Load the long context/document used by all probes.

    Args:
        path: Path to the context file.

    Returns:
        The content of the context file as a string.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Context file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Context file is empty: {file_path}")
    return text


def load_context_prefix(path: str | Path, max_chars: int | None) -> str:
    """Load a prefix of a context for effective-context-length experiments.

    This is a pragmatic v1 approach. Later, you can replace this with a real
    tokenizer-based slicer for exact 8K/32K/128K token budgets.

    Args:
        path: Path to the context file.
        max_chars: Maximum number of characters to read from the context file. If None, read the entire file.

    Returns:
        The content of the context file as a string, truncated to max_chars if specified.
    """

    text = load_context(path)
    if max_chars is None or max_chars <= 0:
        return text
    return text[:max_chars]
