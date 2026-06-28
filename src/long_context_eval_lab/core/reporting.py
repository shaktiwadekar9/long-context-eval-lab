from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from long_context_eval_lab.schemas import EvalResult, RunSummary


def build_summary(
    *,
    run_id: str,
    model: str,
    provider: str,
    results: list[EvalResult],
) -> RunSummary:
    """Build a summary report from the list of EvalResult objects.

    Args:
        run_id: Unique identifier for the evaluation run.
        model: Name of the model being evaluated.
        provider: Provider of the model.
        results: List of evaluation results.

    Returns:
        A RunSummary object containing the aggregated evaluation metrics.
    """
    total = len(results)
    passed = sum(1 for item in results if item.score.passed)
    hallucinations = sum(1 for item in results if item.score.hallucination_flag)
    avg_latency = sum(item.response.latency_seconds for item in results) / total if total else 0.0

    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for item in results:
        grouped[item.probe.category.value].append(item)

    by_category: dict[str, dict[str, float | int]] = {}
    for category, items in grouped.items():
        cat_total = len(items)
        cat_passed = sum(1 for item in items if item.score.passed)
        cat_hallucinations = sum(1 for item in items if item.score.hallucination_flag)
        by_category[category] = {
            "total": cat_total,
            "passed": cat_passed,
            "accuracy": cat_passed / cat_total if cat_total else 0.0,
            "hallucination_rate": cat_hallucinations / cat_total if cat_total else 0.0,
            "avg_latency_seconds": sum(item.response.latency_seconds for item in items) / cat_total
            if cat_total
            else 0.0,
        }

    return RunSummary(
        run_id=run_id,
        model=model,
        provider=provider,
        total=total,
        passed=passed,
        accuracy=passed / total if total else 0.0,
        hallucination_rate=hallucinations / total if total else 0.0,
        avg_latency_seconds=avg_latency,
        by_category=by_category,
    )


def write_markdown_report(path: str | Path, summary: RunSummary, results: list[EvalResult]) -> None:
    """Write a Markdown report summarizing the evaluation results.

    Args:
        path: Path to the Markdown report file.
        summary: The summary of the evaluation results.
        results: List of evaluation results.

    Returns:
        None. The report is written to the specified path.
    """
    lines: list[str] = []
    lines.append(f"# Long Context Eval Report: `{summary.run_id}`")
    lines.append("")
    lines.append(f"- **Provider:** {summary.provider}")
    lines.append(f"- **Model:** {summary.model}")
    lines.append(f"- **Total probes:** {summary.total}")
    lines.append(f"- **Passed:** {summary.passed}")
    lines.append(f"- **Accuracy:** {summary.accuracy:.2%}")
    lines.append(f"- **Hallucination rate:** {summary.hallucination_rate:.2%}")
    lines.append(f"- **Average latency:** {summary.avg_latency_seconds:.2f}s")
    lines.append("")

    lines.append("## Category Scores")
    lines.append("")
    lines.append("| Category | Total | Passed | Accuracy | Hallucination Rate | Avg Latency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for category, stats in summary.by_category.items():
        lines.append(
            f"| {category} | {stats['total']} | {stats['passed']} | "
            f"{float(stats['accuracy']):.2%} | {float(stats['hallucination_rate']):.2%} | "
            f"{float(stats['avg_latency_seconds']):.2f}s |"
        )
    lines.append("")

    lines.append("## Probe Results")
    lines.append("")
    lines.append("| Probe | Category | Passed | Expected | Answer | Reason |")
    lines.append("|---|---|---:|---|---|---|")
    for item in results:
        answer = item.response.answer.replace("\n", " ")[:160]
        expected = item.probe.expected_answer.replace("\n", " ")[:120]
        lines.append(
            f"| {item.probe.id} | {item.probe.category.value} | "
            f"{str(item.score.passed)} | {expected} | {answer} | {item.score.reason} |"
        )
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
