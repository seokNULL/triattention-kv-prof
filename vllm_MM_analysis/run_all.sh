#!/usr/bin/env bash
# Dump the KV cache of benchmark items for every model and frame count.
# Run from vllm_MM_analysis/:
#   bash run_all.sh                                   # item 0 of mvbench, 3 models, 16 frames
#   ITEMS=0-9 FRAMES="16 32 64" bash run_all.sh       # items 0..9, three frame counts
#   ITEMS=0,5,7 DATASETS=videomme TASKS=short bash run_all.sh
#   MODELS="Qwen2.5-VL-7B" SKIP_EXISTING=1 bash run_all.sh
# ITEMS indexes the item list of the dataset (after TASKS); results:
# results/<model>/<dataset>/f<frames>/<item id>/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${MODELS:-LLaVA-OneVision-7B Qwen2.5-VL-7B InternVL3-8B}"
DATASETS="${DATASETS:-mvbench}"
FRAMES="${FRAMES:-16}"
ITEMS="${ITEMS:-0}"
TASK_ARGS=()
SKIP_ARGS=()
[[ "${SKIP_EXISTING:-0}" == "1" ]] && SKIP_ARGS=(--skip-existing)
[[ -n "${TASKS:-}" ]] && read -ra TASK_ARGS <<< "--tasks $TASKS"

indices() {
  local part
  IFS=, read -ra parts <<< "$ITEMS"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      echo "$part"
    else
      echo "invalid ITEMS part: '$part' (use N, N-M or a comma list)" >&2
      exit 1
    fi
  done
}
idx="$(indices)"

for model in $MODELS; do
  for dataset in $DATASETS; do
    for f in $FRAMES; do
      for i in $idx; do
        echo "===== $model / $dataset / ${f} frames / item $i ====="
        python "$HERE/kv_dump.py" --model "$model" --dataset "$dataset" --num-frames "$f" \
          --item "$i" ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} ${SKIP_ARGS[@]+"${SKIP_ARGS[@]}"} "$@"
      done
    done
  done
done
