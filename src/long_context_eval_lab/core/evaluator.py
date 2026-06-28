from __future__ import annotations

import json
import uuid
from pathlib import Path

from long_context_eval_lab.clients.base import ModelClient
from long_context_eval_lab.core.context_loader import load_context_prefix
from long_context_eval_lab.core.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from long_context_eval_lab.core.reporting import build_summary, write_markdown_report
from long_context_eval_lab.core.scoring import score_response
from long_context_eval_lab.core.token_estimator import estimate_tokens
from long_context_eval_lab.schemas import EvalResult, Probe, RunSummary


def run_evaluation(
    *,
    client: ModelClient,
    context_path: str | Path,
    probes: list[Probe],
    output_dir: str | Path,
    max_context_chars: int | None = None,
) -> RunSummary:
    """Run all probes against the same provided context.

    The evaluator is provider-neutral. It does not know whether the model is
    Ollama, OpenAI, vLLM, SGLang, or something else. That logic stays inside
    the model client.

    Args:
        client: Model client to use for generating responses.
        context_path: Path to the context file to use for all probes.
        probes: List of evaluation probes to run.
        output_dir: Directory to write results and reports to.
        max_context_chars: Optional maximum number of characters to read from the context file.

    Returns:
        RunSummary object with aggregated results.
    """

    run_id = uuid.uuid4().hex[:12]
    out = Path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    # Load the context and estimate its token count
    context = load_context_prefix(context_path, max_context_chars)
    context_chars = len(context)
    approx_tokens = estimate_tokens(context)

    results: list[EvalResult] = []
    jsonl_path = out / "results.jsonl"

    # Run each probe and collect results
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for index, probe in enumerate(probes, start=1):
            print(f"[{index}/{len(probes)}] Running {probe.id} ({probe.category.value})")
            user_prompt = build_user_prompt(context, probe)
            model_response = client.generate(SYSTEM_PROMPT, user_prompt)
            score = score_response(probe, model_response)

            result = EvalResult(
                run_id=run_id,
                model=client.model_name,
                provider=client.provider_name,
                context_file=str(context_path),
                context_chars=context_chars,
                approx_input_tokens=approx_tokens,
                probe=probe,
                response=model_response,
                score=score,
            )
            results.append(result)
            fp.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
            fp.flush()

    # Build summary and write report
    summary = build_summary(
        run_id=run_id,
        model=client.model_name,
        provider=client.provider_name,
        results=results,
    )
    (out / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_report(out / "report.md", summary, results)

    print(f"\nRun complete: {out}")
    print(f"Accuracy: {summary.accuracy:.2%}")
    print(f"Report: {out / 'report.md'}")
    return summary
