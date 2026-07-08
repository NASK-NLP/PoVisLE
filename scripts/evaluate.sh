#!/bin/bash
set -euo pipefail

REVISION="v1.0.1"
SPLIT="validation"
TASKS="all"
LIMIT=""
USE_CIRCURAL=1
CIRCURAL_MODE="circular"
HF_REPO_ID=""
PUSH_TO_HUB=0

MODELS=(
    ## Qwen3.5 Family
    # "configs/api/qwen3_5_397b_a17b_instruct.yml"
    # "configs/api/qwen3_5_397b_a17b_thinking.yml"

    ## LLaVA PLLuM Family
    # "configs/vllm/llava_pllum_12b.yml"

    ## OpenAI Models
    # "configs/api/gpt_5_2_none.yml"
    # "configs/api/gpt_5_4_mini_medium.yml"
    # "configs/api/gpt_5_4_none.yml"
    # "configs/api/gpt_5_4_mini_none.yml"

    ## Gemma Family
    "configs/vllm/gemma_4_31b_it.yml"

    ## Random
    # "configs/random/random_42.yml"
    # "configs/random/random_123.yml"
)

for config in "${MODELS[@]}"; do
    cmd=(
        python -m povisle.evaluate
        --dataset-revision "${REVISION}"
        --split "${SPLIT}"
        --model-config "${config}"
        --tasks "${TASKS}"
    )
    if [[ -n "${LIMIT}" ]]; then
        cmd+=(
            --limit "${LIMIT}"
        )
    fi
    if [[ "${USE_CIRCURAL}" -eq 1 ]]; then
        cmd+=(
            --use-circural
            --circural-mode "${CIRCURAL_MODE}"
        )
    fi
    if [[ "${PUSH_TO_HUB}" -eq 1 ]]; then
        cmd+=(
            --hf-push-to-hub
            --hf-repo-id "${HF_REPO_ID}"
        )
    fi
    echo "${cmd[@]}"
    "${cmd[@]}"
done
