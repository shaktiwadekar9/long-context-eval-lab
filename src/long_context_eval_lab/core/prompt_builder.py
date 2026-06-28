from __future__ import annotations

from long_context_eval_lab.schemas import Probe


SYSTEM_PROMPT = """You are an evaluator target model.
You must answer using only the provided context.
Do not use outside knowledge.
If the answer is not present in the context, answer exactly: NOT_FOUND.
Return only valid JSON with this schema:
{
  "answer": "string",
  "evidence": ["short exact evidence snippets from the context"],
  "confidence": 0.0
}
The evidence list should contain the smallest relevant snippets that support the answer.
"""


def build_user_prompt(context: str, probe: Probe) -> str:
    """Build the prompt for one evaluation probe."""

    return f"""# Context
{context}

# Evaluation question
{probe.question}

# Answering rules
- Use only the context above.
- Return JSON only.
- If the answer is missing, use exactly NOT_FOUND.
- Include evidence snippets copied or closely matched from the context.
"""
