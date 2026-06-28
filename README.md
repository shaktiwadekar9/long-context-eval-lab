# Long Context Eval Lab

A lightweight, production-style evaluation pipeline for testing whether long-context LLMs can actually use the context they are given.

This uses Ollama locally, but the evaluator is provider-neutral. 
You can later add OpenAI, Groq, OpenRouter, vLLM, SGLang, Gemini, or any other model backend by implementing one client interface.

## Why this project is useful

Long context is not memory. A model may accept a huge context window but still fail to retrieve facts, ignore distractors, connect distant evidence, or abstain when the answer is missing.
This is explained in a bit more detail in the following explainer article: [1M Context Tokens Is Not Memory: The Beginner’s Guide to Long Context](https://medium.com/towards-artificial-intelligence/1m-context-tokens-is-not-memory-the-beginners-guide-to-long-context-f6893ae2a4e9)

This project turns those failure modes into repeatable tests.

## What this evaluates

The same provided context can be tested using multiple probe types:

| Probe type | What it checks |
|---|---|
| `needle_retrieval` | Can the model find exact facts? |
| `lost_in_middle` | Can it retrieve facts buried inside a long context? |
| `multi_hop_qa` | Can it connect distant facts across the context? |
| `distractor_robustness` | Can it ignore stale, wrong, draft, or similar-looking facts? |
| `effective_context_length` | How much context remains usable as the context grows? |
| `abstention` | Can it say `NOT_FOUND` instead of hallucinating? |

## Architecture

```text
long-context-eval-lab/
├── data/sample/                         # Sample context and probes
├── runs/                                # Eval outputs, ignored by git
├── src/long_context_eval_lab/
│   ├── clients/                         # Model provider clients
│   │   ├── base.py                      # Provider-neutral interface
│   │   └── ollama.py                    # Ollama REST client
│   ├── core/
│   │   ├── context_loader.py            # Loads long context
│   │   ├── evaluator.py                 # Runs probes
│   │   ├── probe_loader.py              # Loads JSONL probes
│   │   ├── prompt_builder.py            # Evaluation prompt
│   │   ├── reporting.py                 # JSON/Markdown reports
│   │   ├── scoring.py                   # Deterministic scoring
│   │   └── token_estimator.py           # Approx token counter
│   ├── data/sample_data.py              # Sample dataset generator
│   ├── config.py                        # Environment settings
│   ├── schemas.py                       # Pydantic data models
│   └── cli.py                           # CLI entrypoint
└── tests/                               # Unit tests
```

## Setup

```bash
bash scripts/setup.sh
```

or

```bash
uv sync
```

## Start Ollama

Install Ollama, then run:

```bash
ollama serve
```

In another terminal, pull a model:

Use normal model:

```bash
ollama pull qwen2.5:7b-instruct
```

or

Using 1M context model: (Your system may not be able to run 1M, but <=64K should work)

```bash
ollama pull hf.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M
```

## 1M context models with ollama: A list

```bash
"hf.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M"
```

You can use any Ollama model that fits your machine.

## Create sample data

```bash
uv run lc-eval make-sample --out data/sample
```

This creates:

```text
data/sample/company_knowledge_base.md
data/sample/probes.jsonl
```

## Run the evaluation

```bash
bash scripts/run_sample.sh
```

or

```bash
uv run lc-eval run \
      --context data/sample/company_knowledge_base.md \
      --probes data/sample/probes.jsonl \
      --model qwen2.5:7b-instruct \
      --out runs
```

Each run creates:

```text
runs/<run_id>/results.jsonl
runs/<run_id>/summary.json
runs/<run_id>/report.md
```

## Run an effective-context-length experiment

Use the same probes and same context, but restrict how much of the context the model receives:

```bash
bash scripts/run_limited_context.sh
```

or

```bash
uv run lc-eval run \
      --context data/sample/company_knowledge_base.md \
      --probes data/sample/probes.jsonl \
      --model qwen2.5:7b-instruct \
      --max-context-chars 20000 \
      --out runs
```

Then repeat with larger values:

```text
10000
20000
50000
100000
250000
500000
1000000
```

The largest size where the model keeps acceptable accuracy is your practical effective context length for that dataset.

## Probe format

Each line in `probes.jsonl` is one test.

Example:

```json
{
  "id": "multi_hop_001",
  "category": "multi_hop_qa",
  "question": "Which cloud region is used by the billing service of the project led by employee E-492?",
  "expected_answer": "ap-south-1",
  "required_evidence": [
    "Employee ID E-492 belongs to Anika Rao",
    "Anika Rao leads Project Mercury",
    "Project Mercury currently uses InvoiceCore",
    "InvoiceCore runs in the ap-south-1 cloud region"
  ],
  "forbidden_answers": [],
  "difficulty": "hard",
  "answer_type": "contains"
}
```

Supported `answer_type` values:

| Type | Meaning |
|---|---|
| `contains` | Model answer must contain the expected answer |
| `exact` | Model answer must exactly match expected answer |
| `not_found` | Model must answer `NOT_FOUND` |

## How to add another model provider

Create a new client:

```text
src/long_context_eval_lab/clients/openai_client.py
```

Implement this interface:

```python
from long_context_eval_lab.clients.base import ModelClient
from long_context_eval_lab.schemas import ModelResponse

class OpenAIClient(ModelClient):
    provider_name = "openai"

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        # Call provider API here.
        # Return normalized ModelResponse.
        ...
```

The evaluator does not need to change.

## Run tests

```bash
bash scripts/test.sh
```

or

```bash
uv run pytest
```

