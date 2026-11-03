#!/usr/bin/env bash
# Generate one problem per benchmark with fullkv for every model and dump its KV cache.
# Run from vllm_runner/:
#   bash run_all.sh                       # problem 0 of aime24, aime25, math500 x 3 models
#   PROBLEM=3 bash run_all.sh             # another problem index
#   MODELS="Qwen3-8B" DATASETS="aime24" bash run_all.sh
#   PREFIX_CACHING=1 bash run_all.sh      # enable vLLM prefix caching (off by default)
# Results: results/<model>/<dataset>/problem<N>/ (kv.pt, tokens.json, meta.json, eval/ ...)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${MODELS:-Qwen3-8B DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Qwen-7B}"
DATASETS="${DATASETS:-aime24 aime25 math500}"
PROBLEM="${PROBLEM:-0}"
PREFIX_ARGS=()
[[ "${PREFIX_CACHING:-0}" == "1" ]] && PREFIX_ARGS=(--prefix-caching)

for model in $MODELS; do
  for dataset in $DATASETS; do
    echo "===== $model / $dataset / problem $PROBLEM (fullkv, prefix caching ${PREFIX_CACHING:-0}) ====="
    python "$HERE/kv_dump.py" --model "$model" --dataset "$dataset" --problem "$PROBLEM" \
      --method fullkv ${PREFIX_ARGS[@]+"${PREFIX_ARGS[@]}"} "$@"
  done
done
