#!/usr/bin/env bash
# Block-merge opportunity of video KV dumps (kv_dump.py / run_all.sh output).
# Reuses ../vllm_CoT_analysis/block_merge.py as is, then splits holes into
# video vs text tokens (mm_breakdown.py). Run from vllm_MM_analysis/:
#   bash opportunity.sh                                          # every dump
#   bash opportunity.sh --model LLaVA-OneVision-7B --dataset mvbench --frames 16,32
#   bash opportunity.sh --item action_count_3 --min-match 0.5 --scope recent:4
# --model/--dataset/--frames/--item take comma lists (default: all found);
# other arguments go to block_merge.py.
# Per dump: results/<model>/<dataset>/f<frames>/<item>/opportunity/
#   (decisions.csv, summary.json, merges.json from block_merge.py; modality.json)
# Overview of the dumps analyzed in this run: results/opportunity_summary.csv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCK_MERGE="$HERE/../vllm_CoT_analysis/block_merge.py"
models="*"; datasets="*"; frames="*"; items="*"
merge_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   models="$2"; shift 2 ;;
    --dataset) datasets="$2"; shift 2 ;;
    --frames)  frames="$2"; shift 2 ;;
    --item)    items="$2"; shift 2 ;;
    *)         merge_args+=("$1"); shift ;;
  esac
done

IFS=, read -ra model_list <<< "$models"
IFS=, read -ra dataset_list <<< "$datasets"
IFS=, read -ra frame_list <<< "$frames"
IFS=, read -ra item_list <<< "$items"
shopt -s nullglob
dumps=()
for m in "${model_list[@]}"; do
  for d in "${dataset_list[@]}"; do
    for f in "${frame_list[@]}"; do
      for i in "${item_list[@]}"; do
        for kv in "$HERE"/results/$m/$d/f$f/$i/kv.pt; do
          dumps+=("$(dirname "$kv")")
        done
      done
    done
  done
done
if [[ ${#dumps[@]} -eq 0 ]]; then
  echo "no KV dumps match model=$models dataset=$datasets frames=$frames item=$items" >&2
  echo "available:" >&2
  for kv in "$HERE"/results/*/*/f*/*/kv.pt; do echo "  ${kv#"$HERE"/results/}" >&2; done
  exit 1
fi

for dir in "${dumps[@]}"; do
  echo "===== ${dir#"$HERE"/} ====="
  python "$BLOCK_MERGE" "$dir" ${merge_args[@]+"${merge_args[@]}"}
done
python "$HERE/mm_breakdown.py" "$HERE/results" "${dumps[@]}"
