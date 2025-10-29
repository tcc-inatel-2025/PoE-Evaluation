#!/usr/bin/env bash
set -euo pipefail

# List of models to run
models=(
    "qwen2.5-coder:7b"
    "codegemma:7b"
    "deepseek-coder:6.7b"
    "codellama:7b"
)

# Run generate_samples.py once for each model
for model in "${models[@]}"; do
    echo "Running model: $model"
    python generate_samples.py --model "$model"
done

echo "All runs completed successfully."
