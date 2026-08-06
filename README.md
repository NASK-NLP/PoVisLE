# PoVisLE (**Po**lish **Vis**ion-**L**anguage **E**valuation)

Evaluation tooling for the PoViSLE benchmark.

## What is in this repository

- `povisle/evaluate.py` - command line entry point for evaluation.
- `povisle/tasks/` - benchmark task definitions and parsers:
  - `mcq` for multiple-choice questions,
  - `yn` for yes/no questions,
  - `open` for open-ended questions.
- `povisle/backends/` - model execution backends:
  - `api` for OpenAI-compatible chat-completions APIs,
  - `vllm` for local vLLM inference,
  - `hf` for local Hugging Face inference,
  - `random` for random baselines.
- `povisle/evaluators/` - default and circular-evaluation loops.
- `povisle/metrics.py` - overall, per-task, per-category, classification, and confusion-matrix metrics.
- `configs/` - YAML model configurations grouped by backend.
- `scripts/` - setup and local evaluation helper scripts.

## Setup

Create the virtual environment and install the package with the optional vLLM extra:

```bash
./scripts/setup.sh
```

Equivalent manual setup:

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[vllm]"
```

If you only need API or random backends, the base package is enough:

```bash
uv pip install -e .
```

For API backends, copy `.env.example` to `.env` or otherwise export the key referenced by `model_args.api_key_env` in the selected config, for example `OPENAI_API_KEY`.

## Model configs

Each evaluation uses a YAML file from `configs/`. A config declares model metadata, backend, backend-specific `model_args`, generation parameters, and optional prompt postprocessing/preprocessing steps.

Minimal random baseline example:

```yaml
name: Random (42)
org: random
model_id: random
model_family: random
model_type: random
model_size: null
backend: random
model_args:
  seed: 42
generation: {}
```

API configs use an OpenAI-compatible client. For example, an API config may set `model_args.api_key_env`, `model_args.base_url`, retry settings, and `generation` parameters passed to `chat.completions.create`.

vLLM configs pass `model_args` to `vllm.LLM` and use `generation` values such as `max_new_tokens`, `temperature`, `top_p`, `top_k`, and `repetition_penalty`.

## Run an evaluation

Basic command:

```bash
python -m povisle.evaluate \
  --model-config configs/random/random_42.yml \
  --dataset-id NASK-PIB/PoVisLE \
  --dataset-revision v1.0.1 \
  --split validation \
  --tasks all
```

Useful options:

- `--tasks all` evaluates `mcq`, `yn`, and `open`.
- `--tasks mcq yn` evaluates only selected tasks.
- `--limit 100` limits examples per task for a smoke test.
- `--results-dir results` selects the output directory.
- `--no-image` evaluates with images removed from model inputs.
- `--no-question` evaluates with questions removed from prompts.
- `--use-circural` enables circular evaluation for `mcq`.
- `--circural-mode circular` rotates answer options.
- `--circural-mode all` evaluates all option permutations.
- `--hf-push-to-hub --hf-repo-id <org>/<repo>` uploads run and leaderboard artifacts to a Hugging Face dataset repository.

The `circural` spelling is intentional in the current CLI flags and scripts.

## Helper scripts

Local loop:

```bash
./scripts/evaluate.sh
```

The local script evaluates the model configs listed in its `MODELS` array. Edit the variables at the top of the file to change dataset revision, split, task selection, sample limit, circular mode, Hub upload behavior, and selected configs.

## Outputs

By default, evaluation outputs are written under `results/`.

Run artifacts:

```text
results/runs/<run_id>/
  run_metadata.json
  summary_metrics.json
  all_predictions.json
```

Leaderboard artifacts:

```text
results/leaderboard/<org>/<model_name>/<dataset_revision>/<split>/<evaluation_mode>/
  results.json
  predictions.json
```

`run_metadata.json` includes model metadata, dataset information, selected tasks, evaluation mode, preprocessing/postprocessing config, full model config, and collected system metadata.

## Metrics

The summary metrics include:

- total examples,
- mean score,
- strict accuracy,
- parse rate,
- runtime failure rate,
- per-task metrics,
- per-category metrics when a `category` column is present,
- per-category and per-subcategory metrics when `category` and `subcategory` columns are present,
- per-image-source metrics when an `image_source` column is present,
- per-task category and image-source breakdowns,
- macro/micro precision, recall, F1, and balanced accuracy for closed-label single-prediction tasks,
- confusion matrices for default `mcq` and `yn` evaluation.

For circular `mcq` evaluation, each original example is expanded into option order variants. The final example score is `1.0` only when all variants are correct.

## Acknowledgement

This work was supported by the Polish Ministry of Digital Affairs (subsidy no. 4/WII/DBI/2026).
The computational resources were provided by the Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) under computational grant no. PLG/2026/019138. 

## Citation

```bibtex
@article{kolos2026povisle,
  title   = {Jako Tako or Fluent? Presenting PoVisLE: A Polish Vision-Language Evaluation},
  author  = {Ko{\l}os, Anna and Statkiewicz, Grzegorz and Seweryn, Karolina and Kowol, Katarzyna and Piosek, Karolina and Kusa, Wojciech},
  journal = {arXiv preprint},
  year    = {2026}
}
```

