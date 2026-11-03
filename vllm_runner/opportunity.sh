#!/usr/bin/env bash
# Run the block-merge opportunity analysis on every KV dump under results/.
# Run from vllm_runner/ after run_all.sh. Extra arguments go to block_merge.py:
#   bash opportunity.sh                              # --min-match 0.9 --scope block
#   bash opportunity.sh --min-match 0.8 --scope recent:4
# Per dump: results/<model>/<dataset>/problem<N>/opportunity/
# Overview: results/opportunity_summary.csv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shopt -s nullglob
dumps=("$HERE"/results/*/*/problem*/kv.pt)
if [[ ${#dumps[@]} -eq 0 ]]; then
  echo "no KV dumps under $HERE/results; run run_all.sh first" >&2
  exit 1
fi

for kv in "${dumps[@]}"; do
  dir="$(dirname "$kv")"
  echo "===== ${dir#"$HERE"/} ====="
  python "$HERE/block_merge.py" "$dir" "$@"
done

python - "$HERE/results" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary in sorted(root.glob("*/*/problem*/opportunity/summary.json")):
    s = json.loads(summary.read_text())
    layers = s["layers"].values()
    rel = summary.parent.parent.relative_to(root).parts
    rows.append({
        "model": rel[0], "dataset": rel[1], "problem": rel[2],
        "tokens": next(iter(layers))["tokens"],
        "min_match": s["min_match"], "scope": s["scope"],
        "dropped_ratio_mean": round(sum(l["dropped_ratio"] for l in layers) / len(layers), 4),
        "dropped_ratio_min": min(l["dropped_ratio"] for l in layers),
        "dropped_ratio_max": max(l["dropped_ratio"] for l in layers),
        "blocks": next(iter(layers))["blocks"],
        "blocks_with_holes_mean": round(sum(l["blocks_with_holes"] for l in layers) / len(layers), 1),
        "blocks_with_holes_ratio_mean": round(sum(l["blocks_with_holes_ratio"] for l in layers) / len(layers), 4),
    })
out = root / "opportunity_summary.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n{'model':<30} {'dataset':<8} {'problem':<9} {'tokens':>6} {'dropped(mean/min/max)':>24} {'blocks w/ holes (mean)':>24}")
for r in rows:
    print(f"{r['model']:<30} {r['dataset']:<8} {r['problem']:<9} {r['tokens']:>6} "
          f"{r['dropped_ratio_mean']:>8.1%} {r['dropped_ratio_min']:>6.1%} {r['dropped_ratio_max']:>6.1%} "
          f"{r['blocks_with_holes_mean']:>10}/{r['blocks']:<6} ({r['blocks_with_holes_ratio_mean']:.1%})")
print(f"wrote {out}")
PY
