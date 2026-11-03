#!/usr/bin/env bash
# Full benchmark (every problem, fullkv and triattention) for every model x dataset.
# Extra arguments are forwarded to run_benchmark.py, e.g.:
#   bash run_benchmark_all.sh --num-samples 1 --max-examples 3
#   MODELS="Qwen3-8B" DATASETS="aime24 math500" METHODS=triattention bash run_benchmark_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${MODELS:-Qwen3-8B DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Qwen-7B}"
DATASETS="${DATASETS:-aime24 aime25 math500}"
METHODS="${METHODS:-fullkv triattention}"

for model in $MODELS; do
  for dataset in $DATASETS; do
    for method in $METHODS; do
      echo "===== $model / $dataset / $method ====="
      python "$HERE/run_benchmark.py" --model "$model" --dataset "$dataset" --method "$method" "$@"
    done
  done
done
