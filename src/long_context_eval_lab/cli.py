from __future__ import annotations

import argparse
from pathlib import Path

from long_context_eval_lab.clients.ollama import OllamaClient
from long_context_eval_lab.config import Settings
from long_context_eval_lab.core.evaluator import run_evaluation
from long_context_eval_lab.core.probe_loader import load_probes
from long_context_eval_lab.data.sample_data import write_sample_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lc-eval",
        description="Evaluate how well long-context LLMs use provided context.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("make-sample", help="Create sample context and probes.")
    sample.add_argument("--out", default="data/sample", help="Output directory for sample data.")
    sample.add_argument(
        "--filler-sections",
        type=int,
        default=80,
        help="Amount of noise/filler to add to the sample context.",
    )

    run = subparsers.add_parser("run", help="Run evaluation probes against a context.")
    run.add_argument("--context", required=True, help="Path to long context/document file.")
    run.add_argument("--probes", required=True, help="Path to JSONL probe file.")
    run.add_argument("--out", default="runs", help="Directory for run outputs.")
    run.add_argument("--provider", default="ollama", choices=["ollama"], help="Model provider.")
    run.add_argument("--model", default=None, help="Model name. Defaults to OLLAMA_MODEL.")
    run.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama base URL. Defaults to OLLAMA_BASE_URL.",
    )
    run.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Optional prefix length for effective-context-length experiments.",
    )
    run.add_argument(
        "--env-file",
        default=None,
        help="Optional .env file path.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "make-sample":
        write_sample_dataset(args.out, filler_sections=args.filler_sections)
        return

    if args.command == "run":
        settings = Settings.from_env(args.env_file)
        probes = load_probes(args.probes)

        if args.provider == "ollama":
            client = OllamaClient(
                model_name=args.model or settings.ollama_model,
                base_url=args.ollama_base_url or settings.ollama_base_url,
                timeout_seconds=settings.request_timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")

        run_evaluation(
            client=client,
            context_path=Path(args.context),
            probes=probes,
            output_dir=Path(args.out),
            max_context_chars=args.max_context_chars,
        )
        return

    parser.print_help()

if __name__ == "__main__":
    main()
