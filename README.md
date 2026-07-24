# GB/T Document Review Benchmark

This repository provides an anonymized research artifact for evaluating large
language models on automated review of Chinese GB/T standard documents. It
contains a counterexample-based benchmark, an LLM evaluation pipeline, optional
agentic review modes, and utilities for constructing additional counterexamples
from source documents.

This artifact is released for double-blind peer review. Identifying information,
including author names, affiliations, contact details, and links to non-anonymous
project pages, is intentionally omitted.

## Overview

The benchmark evaluates whether a model can detect, locate, and diagnose errors
in standard documents. The current implementation covers five review dimensions:

| Dimension | Review target | Error types |
| --- | --- | ---: |
| C2.1 | Document structure | 6 |
| C2.2 | Scope consistency | 3 |
| C3.1 | Normative wording and modality | 7 |
| C3.2 | Terms and definitions | 5 |
| C3.3 | Normative references | 4 |

The evaluation supports two principal settings:

- **Single-prompt review:** one LLM call reviews all enabled dimensions.
- **Agentic review:** direct review, dimension specialists, error-type agents,
  and rule-based local scanners generate and consolidate candidate findings.

Each agentic component can be disabled independently for ablation experiments.

## Repository Structure

```text
.
├── config/                  # Environment-based runtime configuration
├── core/                    # LLM clients, parsing, and counterexamples
├── data/
│   └── GBT_Data_fanli_10to17/
│       ├── GBT_test_balanced_00.json
│       └── ...              # Eight benchmark shards
├── pipeline/                # General document-review pipeline
├── reviewers/               # Reviewers organized by review level
├── tests/                   # Unit and integration tests
├── main.py                  # Benchmark evaluation entry point
└── run_counter_example.py   # Counterexample construction entry point
```

The included benchmark contains **8 JSON shards and 488 document records**. Each
record contains the source text and one or more constructed examples with their
ground-truth annotations.

## Environment Setup

Python 3.10 or later is recommended. The artifact was checked with Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openai pymupdf python-docx
```

Optional dependencies:

```bash
# Unit tests
python -m pip install pytest

# Optional PDF parsing backend
python -m pip install docling
```

## Configuration

Runtime settings are read from environment variables or from a local `.env`
file in the repository root. Do not commit API keys.

For an OpenAI-compatible endpoint:

```dotenv
LLM_BACKEND=proxy
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_MODEL=your_model_name
```

For Azure OpenAI:

```dotenv
LLM_BACKEND=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.example
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

The command-line `--backend` option determines which configuration is used.
Values in `.env` are loaded only when the corresponding environment variable is
not already set.

## Running the Benchmark

Run a small smoke test on one benchmark shard:

```bash
python main.py \
  --input data/GBT_Data_fanli_10to17/GBT_test_balanced_00.json \
  --output outputs/smoke_test.json \
  --backend proxy \
  --limit 2
```

Run the single-prompt baseline on a complete shard:

```bash
python main.py \
  --input data/GBT_Data_fanli_10to17/GBT_test_balanced_00.json \
  --output outputs/single_prompt_00.json \
  --backend proxy
```

Run the agentic pipeline:

```bash
python main.py \
  --input data/GBT_Data_fanli_10to17/GBT_test_balanced_00.json \
  --output outputs/agentic_00.json \
  --backend proxy \
  --agentIsTrue \
  --agentMaxSteps 12
```

The following switches support component ablations:

```text
--no-agent-direct-all
--no-agent-dimension-specialists
--no-agent-error-type-agents
--no-agent-rule-local-scanners
```

Run `python main.py --help` for the complete command-line interface.

## Output and Metrics

For an output path such as `outputs/run.json`, the evaluator writes:

- `outputs/run.json`: per-document predictions, ground truth, and matching
  results;
- `outputs/run_summary.json`: aggregate metrics and configuration metadata;
- `logs/gbt_parse.log`: runtime logs.

Reported results include document-level detection, location, and diagnosis
scores; item-level diagnosis recall; recall grouped by dimension and error type;
and the soft NP-MCS metric implemented in `main.py`. The summary also records the
review mode, active agent modules, and metric parameters needed to interpret a
run.

## Constructing Counterexamples

`run_counter_example.py` constructs annotated counterexamples from GB/T PDF
documents. Source PDFs are not included in this artifact and must be supplied by
the user in accordance with their applicable access and redistribution terms.

Example:

```bash
python run_counter_example.py \
  --file path/to/source.pdf \
  --output data/generated_examples \
  --backend proxy \
  --dims C2.1 C2.2 C3.1 C3.2 C3.3 \
  --num 5
```

Use `--docling` to select the optional Docling parser or `--no-docling` to use
the PyMuPDF/LLM/regular-expression path.

## Tests

```bash
python -m pytest tests/unit -q
```

Tests that contact an LLM service may require valid credentials and should be
treated separately from offline unit tests.

## Reproducibility Notes

- Record the model/deployment identifier and backend used for every experiment.
- Keep temperature and model-side sampling settings fixed across comparisons.
- Store raw prediction files together with their generated summary files.
- Use identical benchmark shards and agent-module switches for paired runs.
- LLM provider updates may introduce nondeterminism even with fixed local code.

## Anonymous-Review Notice

Please do not add author names, affiliations, personal email addresses, local
absolute paths, repository-owner identifiers, acknowledgements, or links to
non-anonymous materials while this artifact is under double-blind review. A
public citation, license, and maintainer information can be added after the
review process permits de-anonymization.
