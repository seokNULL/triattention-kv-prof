#!/usr/bin/env python3
"""Split block_merge.py results of video KV dumps into video vs text tokens.

For each dump folder (after ../vllm_CoT_analysis/block_merge.py has written
opportunity/), writes opportunity/modality.json with hole ratios of video
and text tokens (overall, per layer, per frame group), then prints a table
over all given dumps and writes it to <results>/opportunity_summary.csv.

Usually called by opportunity.sh.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def breakdown(dump: Path) -> dict:
    tokens = json.loads((dump / "tokens.json").read_text())
    meta = json.loads((dump / "meta.json").read_text())
    summary = json.loads((dump / "opportunity" / "summary.json").read_text())
    dec = pd.read_csv(dump / "opportunity" / "decisions.csv", usecols=["layer", "token", "block", "kept"])
    group = np.asarray(tokens["vision_group"])
    dec["group"] = group[dec["token"].to_numpy()]
    dec["vision"] = dec["group"] != -1
    dec["hole"] = dec["kept"] == 0

    def ratio(df):
        return round(float(df["hole"].mean()), 4) if len(df) else 0.0

    vis, txt = dec[dec["vision"]], dec[~dec["vision"]]
    per_layer = {int(l): {"video": ratio(d[d["vision"]]), "text": ratio(d[~d["vision"]])}
                 for l, d in dec.groupby("layer")}
    per_group = {int(g): ratio(d) for g, d in vis[vis["group"] >= 0].groupby("group")}
    block_kind = dec[dec["layer"] == dec["layer"].iloc[0]].groupby("block")["vision"].agg(["all", "any"])
    total = summary["total"]
    result = {
        "model": meta["model"], "dataset": meta["dataset"], "frames": meta["num_frames"],
        "item": meta["id"], "task": meta["task"], "correct": meta["correct"],
        "tokens": len(group), "video_tokens": int((group != -1).sum()),
        "layers": total["layers"], "min_match": summary["min_match"], "scope": summary["scope"],
        "slots": total["slots"], "holes": total["holes"], "hole_ratio": total["dropped_ratio"],
        "video_slots": len(vis), "video_holes": int(vis["hole"].sum()), "video_hole_ratio": ratio(vis),
        "text_slots": len(txt), "text_holes": int(txt["hole"].sum()), "text_hole_ratio": ratio(txt),
        "blocks": total["blocks"], "blocks_with_holes": total["blocks_with_holes"],
        "video_only_blocks_per_layer": int(block_kind["all"].sum()),
        "mixed_blocks_per_layer": int((block_kind["any"] & ~block_kind["all"]).sum()),
        "total_kv_mib": round(total["total_kv_bytes"] / 2**20, 2),
        "hole_mib": round(total["hole_bytes"] / 2**20, 2),
        "free_mib_without_migration": round(total["free_bytes_without_migration"] / 2**20, 2),
    }
    (dump / "opportunity" / "modality.json").write_text(json.dumps(
        {**result, "per_layer_hole_ratio": per_layer, "per_frame_group_hole_ratio": per_group},
        indent=2, ensure_ascii=False))
    return result


def main() -> None:
    results_root, dumps = Path(sys.argv[1]), [Path(d) for d in sys.argv[2:]]
    rows = [breakdown(d) for d in dumps]
    out = results_root / "opportunity_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'model':<20} {'dataset':<9} {'frm':>3} {'item':<24} {'ok':>2} {'tokens':>6} {'video':>6} "
          f"{'holes/slots':>20} {'video holes':>11} {'text holes':>10} {'KV MiB':>8} {'hole MiB':>8} "
          f"{'no-migr MiB':>11}")
    for r in rows:
        print(f"{r['model']:<20} {r['dataset']:<9} {r['frames']:>3} {r['item'][:24]:<24} "
              f"{'Y' if r['correct'] else 'N':>2} {r['tokens']:>6} {r['video_tokens']:>6} "
              f"{r['holes']:>9}/{r['slots']:<9} ({r['hole_ratio']:>5.1%}) {r['video_hole_ratio']:>11.1%} "
              f"{r['text_hole_ratio']:>10.1%} {r['total_kv_mib']:>8.1f} {r['hole_mib']:>8.1f} "
              f"{r['free_mib_without_migration']:>11.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
