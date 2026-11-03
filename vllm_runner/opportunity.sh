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
    t = s["total"]
    rows.append({
        "model": rel[0], "dataset": rel[1], "problem": rel[2],
        "tokens": next(iter(layers))["tokens"], "layers": t["layers"],
        "min_match": s["min_match"], "scope": s["scope"],
        "slots": t["slots"], "holes": t["holes"], "hole_ratio": t["dropped_ratio"],
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
print(f"\n{'model':<30} {'dataset':<8} {'problem':<9} {'tokens':>6} {'holes / slots (all layers)':>28} "
      f"{'blocks w/ holes':>17} {'KV total':>11} {'holes (migration)':>18} {'no migration':>13}")
for r in rows:
    print(f"{r['model']:<30} {r['dataset']:<8} {r['problem']:<9} {r['tokens']:>6} "
          f"{r['holes']:>10}/{r['slots']:<9} ({r['hole_ratio']:>6.1%}) "
          f"{r['blocks_with_holes']:>8}/{r['blocks']:<8} {r['total_kv_mib']:>8.1f}MiB "
          f"{r['hole_mib']:>15.1f}MiB {r['free_mib_without_migration']:>10.1f}MiB")
print(f"wrote {out}")
PY
