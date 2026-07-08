#!/bin/bash

set -euo pipefail

VENV_NAME=".venv"
if [ ! -d $VENV_NAME ]; then
    python -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate


uv pip install -e ".[vllm]"
