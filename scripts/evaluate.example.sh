#!/bin/bash
set -euo pipefail

REVISION="v1.2.0"
SPLIT="validation"

# Put your model config here. You can find some example configs in the `configs/` directory.
MODEL="configs/vllm/gemma_4_31b_it.yml"

python -m povisle.evaluate \
    --dataset-revision "${REVISION}" \
    --split "${SPLIT}" \
    --model-config "${MODEL}"
