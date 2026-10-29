#!/usr/bin/env bash
# Run fullkv and triattention for every model x dataset with vLLM.
# Extra arguments are forwarded to run_vllm.py, e.g.:
#   bash vllm_runner/run_all.sh --num-samples 1 --max-examples 3
#   MODELS="Qwen3-8B" DATASETS="aime24 math500" METHODS=triattention bash vllm_runner/run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${MODELS:-Qwen3-8B DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Qwen-7B}"
DATASETS="${DATASETS:-aime24 aime25 math500}"
METHODS="${METHODS:-fullkv triattention}"

for model in $MODELS; do
  for dataset in $DATASETS; do
    for method in $METHODS; do
      echo "===== $model / $dataset / $method ====="
      python "$HERE/run_vllm.py" --model "$model" --dataset "$dataset" --method "$method" "$@"
    done
  done
done
