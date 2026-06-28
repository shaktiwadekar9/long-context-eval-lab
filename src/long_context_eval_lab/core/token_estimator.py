from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Fast approximate token counter.

    This keeps the first version dependency-light. For precise counts, replace
    with a model-specific tokenizer later.
    """

    # Common rough estimate for English/code-like text: 1 token ~= 4 chars.
    return max(1, len(text) // 4)
