#!/usr/bin/env python3
"""Offline block-merge simulation on a view_kv_dump.py CSV.

Tokens are appended in order into blocks of block_size slots, separately for
every layer. When token t arrives, it is compared with the kept tokens of
its block (--scope block) or with the kept tokens among the previous N
positions of its block (--scope recent:N). Two tokens are compared over one
whole slot, i.e. every kv_head x head_dim element of the layer, and an
element counts as equal when its bf16 sign and exponent bits are equal.
If some candidate has at least --min-match equal elements in K and in V,
the new token is dropped (the existing token is kept as is) and its slot
becomes a hole.

The CSV must contain K and V rows (view_kv_dump.py --kv KV, the default)
with every head and dim, e.g.:
    python vllm_runner/view_kv_dump.py $D --layer 18 --tokens all --csv kv_l18.csv

Example:
    python vllm_runner/block_merge.py kv_l18.csv --min-match 0.9
    python vllm_runner/block_merge.py kv_l18.csv --min-match 900 --scope recent:4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv")
    p.add_argument(
        "--min-match",
        type=float,
        default=0.9,
        help="Equal (sign, exponent) elements needed in both K and V: a count such as 900, "
        "or a fraction of kv_heads x head_dim such as 0.9.",
    )
    p.add_argument("--scope", default="block", help="block, or recent:N (previous N positions in the block).")
    p.add_argument("--protect-first", type=int, default=0, help="Never drop the first N tokens (e.g. the prompt).")
    p.add_argument("--out-dir", default=None, help="Default: <csv stem>_merge/ next to the CSV.")
    return p.parse_args()


def sign_exponent(values: np.ndarray) -> np.ndarray:
    """bf16 sign and exponent bits (9 bits) of float values, round-to-nearest-even."""
    bits = values.astype(np.float32).view(np.uint32)
    bf16 = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (bf16 >> 7).astype(np.uint16)


def load(csv_path: str):
    """Returns {layer: (K_se, V_se) each [T, H*D]}, tokens, block/slot per token, head count."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    dims = [c for c in df.columns if c.startswith("d") and c[1:].isdigit()]
    if set(df["kv"]) != {"K", "V"}:
        raise SystemExit("CSV needs both K and V rows (export with --kv KV)")
    heads = sorted(df["head"].unique())
    tokens = sorted(df["token"].unique())
    expected = len(heads) * len(tokens) * 2
    layout = df.drop_duplicates("token").set_index("token").loc[tokens, ["block", "slot"]]
    out = {}
    for layer, ldf in df.groupby("layer"):
        if len(ldf) != expected:
            raise SystemExit(f"layer {layer}: {len(ldf)} rows, expected {expected} (all heads x tokens x K/V)")
        per_kind = []
        for kind in ("K", "V"):
            kdf = ldf[ldf["kv"] == kind].sort_values(["token", "head"])
            vals = kdf[dims].to_numpy(dtype=np.float32).reshape(len(tokens), len(heads) * len(dims))
            per_kind.append(sign_exponent(vals))
        out[int(layer)] = tuple(per_kind)
    return out, np.array(tokens), layout["block"].to_numpy(), layout["slot"].to_numpy(), len(heads), len(dims)


def candidates(i: int, kept: np.ndarray, block: np.ndarray, scope: str) -> np.ndarray:
    """Indices of kept earlier tokens in the same block within scope."""
    same_block = np.flatnonzero((block[:i] == block[i]) & kept[:i])
    if scope == "block":
        return same_block
    if scope.startswith("recent:"):
        n = int(scope.split(":")[1])
        return same_block[same_block >= i - n]
    raise SystemExit(f"unknown --scope {scope}")


def simulate(k_se, v_se, block, scope, min_match, protect_first):
    t_len = k_se.shape[0]
    kept = np.ones(t_len, dtype=bool)
    merged_into = np.full(t_len, -1)
    best_k = np.full(t_len, -1)
    best_v = np.full(t_len, -1)
    for i in range(t_len):
        cand = candidates(i, kept, block, scope)
        if len(cand) == 0:
            continue
        mk = (k_se[cand] == k_se[i]).sum(axis=1)
        mv = (v_se[cand] == v_se[i]).sum(axis=1)
        both = np.minimum(mk, mv)
        j = int(both.argmax())
        best_k[i], best_v[i] = mk[j], mv[j]
        if i >= protect_first and both[j] >= min_match:
            kept[i] = False
            merged_into[i] = cand[j]
    return kept, merged_into, best_k, best_v


def main() -> None:
    args = parse_args()
    data, tokens, block, slot, n_heads, head_dim = load(args.csv)
    n_elem = n_heads * head_dim
    min_match = int(round(args.min_match * n_elem)) if args.min_match <= 1 else int(args.min_match)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / f"{csv_path.stem}_merge"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.csv}: layers {sorted(data)}, {len(tokens)} tokens, {n_heads} kv_heads x {head_dim} dims "
          f"= {n_elem} elements per slot; merge if >= {min_match} equal (sign, exponent) in K and V; "
          f"scope={args.scope}")

    rows, summary = [], {}
    for layer, (k_se, v_se) in sorted(data.items()):
        kept, merged_into, best_k, best_v = simulate(k_se, v_se, block, args.scope, min_match,
                                                     args.protect_first)
        for i, t in enumerate(tokens):
            rows.append({
                "layer": layer, "token": int(t), "block": int(block[i]), "slot": int(slot[i]),
                "kept": int(kept[i]),
                "merged_into": int(tokens[merged_into[i]]) if merged_into[i] >= 0 else -1,
                "best_match_k": int(best_k[i]), "best_match_v": int(best_v[i]),
            })
        holes_per_block = pd.Series(~kept).groupby(block).sum()
        summary[layer] = {
            "tokens": len(tokens),
            "dropped": int((~kept).sum()),
            "dropped_ratio": round(float((~kept).mean()), 4),
            "blocks": int(len(holes_per_block)),
            "holes_per_block_mean": round(float(holes_per_block.mean()), 2),
            "holes_per_block_hist": {int(h): int(c) for h, c in holes_per_block.value_counts().sort_index().items()},
        }
        print(f"layer {layer:>3}: dropped {summary[layer]['dropped']}/{len(tokens)} "
              f"({summary[layer]['dropped_ratio']:.1%}), holes per block mean "
              f"{summary[layer]['holes_per_block_mean']} of {int(np.bincount(block).max())}")

    pd.DataFrame(rows).to_csv(out_dir / "decisions.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "csv": args.csv, "scope": args.scope, "min_match": min_match, "elements_per_slot": n_elem,
        "protect_first": args.protect_first, "layers": summary,
    }, indent=2))
    print(f"wrote {out_dir}/decisions.csv and summary.json")


if __name__ == "__main__":
    main()
