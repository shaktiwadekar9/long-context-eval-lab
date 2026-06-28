#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-hf.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M}"
CONTEXT="${CONTEXT:-data/sample/company_knowledge_base.md}"
PROBES="${PROBES:-data/sample/probes.jsonl}"
OUT="${OUT:-runs}"

uv run lc-eval run \
  --context "$CONTEXT" \
  --probes "$PROBES" \
  --model "$MODEL" \
  --out "$OUT"