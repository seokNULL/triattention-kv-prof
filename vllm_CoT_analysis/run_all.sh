#!/usr/bin/env bash
# Generate benchmark problems with fullkv for every model and dump each one's KV cache.
# Run from vllm_CoT_analysis/:
#   bash run_all.sh                       # problem 0 of aime24, aime25, math500 x 3 models
#   PROBLEM=3 bash run_all.sh             # another problem index
#   PROBLEM=0-4 bash run_all.sh           # a range (inclusive)
#   PROBLEM=0,3,7 bash run_all.sh         # a list (ranges allowed: 0-2,10)
#   PROBLEM=all bash run_all.sh           # every problem of each dataset
#   MODELS="Qwen3-8B" DATASETS="aime24" bash run_all.sh
#   PREFIX_CACHING=1 bash run_all.sh      # enable vLLM prefix caching (off by default)
#   SKIP_EXISTING=1 PROBLEM=all bash run_all.sh   # skip problems that already have kv.pt
# Results: results/<model>/<dataset>/problem<N>/ (kv.pt, tokens.json, meta.json, eval/ ...)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
MODELS="${MODELS:-Qwen3-8B DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Qwen-7B}"
DATASETS="${DATASETS:-aime24 aime25 math500}"
PROBLEM="${PROBLEM:-0}"
PREFIX_ARGS=()
[[ "${PREFIX_CACHING:-0}" == "1" ]] && PREFIX_ARGS=(--prefix-caching)

# Number of problems (non-empty lines) in a dataset file.
num_problems() {
  local file="$REPO/data/$1.jsonl"
  [[ "$1" == "quick" ]] && file="$HERE/data/quick.jsonl"
  if [[ ! -f "$file" ]]; then
    echo "dataset file not found: $file (run scripts/prepare_assets.py)" >&2
    exit 1
  fi
  grep -c . "$file"
}

# Expand PROBLEM ("all", "3", "0-4", "0,3,7-9") into indices for one dataset.
problem_indices() {
  local dataset="$1" n part
  if [[ "$PROBLEM" == "all" ]]; then
    n="$(num_problems "$dataset")" || exit 1
    seq 0 $((n - 1))
    return
  fi
  IFS=, read -ra parts <<< "$PROBLEM"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      echo "$part"
    else
      echo "invalid PROBLEM part: '$part' (use all, N, N-M or a comma list)" >&2
      exit 1
    fi
  done
}

for model in $MODELS; do
  for dataset in $DATASETS; do
    indices="$(problem_indices "$dataset")"
    total="$(wc -w <<< "$indices")"
    count=0
    for p in $indices; do
      count=$((count + 1))
      if [[ "${SKIP_EXISTING:-0}" == "1" && -f "$HERE/results/$model/$dataset/problem$p/kv.pt" ]]; then
        echo "===== $model / $dataset / problem $p ($count/$total): already dumped, skipping ====="
        continue
      fi
      echo "===== $model / $dataset / problem $p ($count/$total) (fullkv, prefix caching ${PREFIX_CACHING:-0}) ====="
      python "$HERE/kv_dump.py" --model "$model" --dataset "$dataset" --problem "$p" \
        --method fullkv ${PREFIX_ARGS[@]+"${PREFIX_ARGS[@]}"} "$@"
    done
  done
done
