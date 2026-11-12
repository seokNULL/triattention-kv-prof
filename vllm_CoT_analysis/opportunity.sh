#!/usr/bin/env bash
# Run the block-merge opportunity analysis on KV dumps under results/.
# Run from vllm_CoT_analysis/ after run_all.sh.
#   bash opportunity.sh                                        # every dump
#   bash opportunity.sh --model Qwen3-8B --dataset aime24 --problem 0
#   bash opportunity.sh --model Qwen3-8B,DeepSeek-R1-Distill-Qwen-7B --problem 0,1,2
#   bash opportunity.sh --dataset math500 --min-match 0.8 --scope recent:4
# --model/--dataset/--problem take comma lists and default to everything found;
# all other arguments go to block_merge.py.
# Per dump: results/<model>/<dataset>/problem<N>/opportunity/
# Overview of the dumps analyzed in this run: results/opportunity_summary.csv
#   (hole ratio overall and split into prompt vs generated tokens; LongBench domain)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models="*"; datasets="*"; problems="*"
merge_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   models="$2"; shift 2 ;;
    --dataset) datasets="$2"; shift 2 ;;
    --problem) problems="$2"; shift 2 ;;
    *)         merge_args+=("$1"); shift ;;
  esac
done

# Split the comma lists without glob-expanding "*" against the current directory.
IFS=, read -ra model_list <<< "$models"
IFS=, read -ra dataset_list <<< "$datasets"
IFS=, read -ra problem_list <<< "$problems"
shopt -s nullglob
dumps=()
for m in "${model_list[@]}"; do
  for d in "${dataset_list[@]}"; do
    for p in "${problem_list[@]}"; do
      [[ "$p" == "*" ]] && pattern="problem*" || pattern="problem$p"
      for kv in "$HERE"/results/$m/$d/$pattern/kv.pt; do
        dumps+=("$(dirname "$kv")")
      done
    done
  done
done
if [[ ${#dumps[@]} -eq 0 ]]; then
  echo "no KV dumps match model=$models dataset=$datasets problem=$problems under $HERE/results" >&2
  echo "available:" >&2
  for kv in "$HERE"/results/*/*/problem*/kv.pt; do echo "  ${kv#"$HERE"/results/}" >&2; done
  exit 1
fi

for dir in "${dumps[@]}"; do
  echo "===== ${dir#"$HERE"/} ====="
  python "$HERE/block_merge.py" "$dir" ${merge_args[@]+"${merge_args[@]}"}
done

python - "$HERE/results" "${dumps[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

import pandas as pd


def split_by_phase(dump: Path):
    """Hole ratio of prompt (prefill) tokens and of generated tokens."""
    n_prompt = json.loads((dump / "tokens.json").read_text())["num_prompt_tokens"]
    dec = pd.read_csv(dump / "opportunity" / "decisions.csv", usecols=["token", "kept"])
    hole = dec["kept"] == 0
    prompt = dec["token"] < n_prompt
    ratio = lambda m: round(float(hole[m].mean()), 4) if m.any() else 0.0  # noqa: E731
    return n_prompt, ratio(prompt), ratio(~prompt)

root = Path(sys.argv[1])
rows = []
for summary in (Path(d) / "opportunity" / "summary.json" for d in sys.argv[2:]):
    s = json.loads(summary.read_text())
    layers = s["layers"].values()
    rel = summary.parent.parent.relative_to(root).parts
    t = s["total"]
    dump = summary.parent.parent
    meta = json.loads((dump / "meta.json").read_text()) if (dump / "meta.json").exists() else {}
    n_prompt, prompt_ratio, output_ratio = split_by_phase(dump)
    rows.append({
        "model": rel[0], "dataset": rel[1], "problem": rel[2], "domain": meta.get("domain", ""),
        "tokens": next(iter(layers))["tokens"], "prompt_tokens": n_prompt, "layers": t["layers"],
        "min_match": s["min_match"], "scope": s["scope"],
        "slots": t["slots"], "holes": t["holes"], "hole_ratio": t["dropped_ratio"],
        "prompt_hole_ratio": prompt_ratio, "output_hole_ratio": output_ratio,
        "dropped_ratio_min_layer": min(l["dropped_ratio"] for l in layers),
        "dropped_ratio_max_layer": max(l["dropped_ratio"] for l in layers),
        "blocks": t["blocks"], "blocks_with_holes": t["blocks_with_holes"],
        "total_kv_mib": round(t["total_kv_bytes"] / 2**20, 2),
        "hole_mib": round(t["hole_bytes"] / 2**20, 2),
        "free_mib_without_migration": round(t["free_bytes_without_migration"] / 2**20, 2),
    })
out = root / "opportunity_summary.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n{'model':<30} {'dataset':<18} {'problem':<9} {'tokens':>6} {'holes / slots (all layers)':>28} "
      f"{'prompt':>7} {'output':>7} "
      f"{'blocks w/ holes':>17} {'KV total':>11} {'holes (migration)':>18} {'no migration':>13}")
for r in rows:
    print(f"{r['model']:<30} {r['dataset']:<18} {r['problem']:<9} {r['tokens']:>6} "
          f"{r['holes']:>10}/{r['slots']:<9} ({r['hole_ratio']:>6.1%}) "
          f"{r['prompt_hole_ratio']:>7.1%} {r['output_hole_ratio']:>7.1%} "
          f"{r['blocks_with_holes']:>8}/{r['blocks']:<8} {r['total_kv_mib']:>8.1f}MiB "
          f"{r['hole_mib']:>15.1f}MiB {r['free_mib_without_migration']:>10.1f}MiB")
print(f"wrote {out}")
PY
