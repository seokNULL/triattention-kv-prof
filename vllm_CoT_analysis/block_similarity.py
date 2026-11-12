#!/usr/bin/env python3
"""Cosine similarity vs bit-wise similarity of the K vectors inside one KV block.

Takes one KV block (block_size consecutive tokens, block t // block_size) of
one layer of a kv_dump.py result, and compares every pair of its K slots
(all kv_heads x head_dim elements of the layer, e.g. 8 x 128 = 1024) by
    cosine     cosine similarity of the two K vectors
    bit match  number of elements whose bf16 sign and exponent bits are equal
               (the block_merge.py criterion)
and splits the pairs by the two thresholds: merged by both, by cosine only,
by bit match only, or by neither.

Outputs in <dump>/block_similarity/layer<L>_block<B>/:
    pairs.csv     one row per pair: tokens, pieces, cosine, bit match (count and fraction), class
    heatmaps.png  cosine and bit-match matrices side by side
    scatter.png   bit match vs cosine with both thresholds
    summary.json  correlations and class counts

Works on vllm_CoT_analysis and vllm_MM_analysis dumps. Example (from vllm_CoT_analysis/):
    python block_similarity.py results/Qwen3-8B/aime24/problem0 --layer 18 --block 100
    python block_similarity.py results/Qwen3-8B/aime24/problem0 --layer 18 --block 100 \\
        --cos-threshold 0.9 --min-match 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from block_merge import sign_exponent

CLASSES = ["both", "cosine_only", "bit_only", "neither"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dump", help="kv_dump.py result folder (kv.pt, tokens.json).")
    p.add_argument("--layer", type=int, required=True, help="Layer id (as in kv.pt 'layers').")
    p.add_argument("--block", type=int, required=True,
                   help="Logical block index: tokens [block*block_size, (block+1)*block_size).")
    p.add_argument("--cos-threshold", type=float, default=0.85, help="Merge if cosine >= this.")
    p.add_argument("--min-match", type=float, default=0.5,
                   help="Merge if equal (sign, exponent) elements >= this: a fraction (<= 1) or a count.")
    p.add_argument("--out-dir", help="Default: <dump>/block_similarity/layer<L>_block<B>/")
    return p.parse_args()


def load_block(dump: Path, layer: int, block: int):
    """K of one block [slots, H*D] (float32 of the stored bf16), token ids and pieces."""
    import torch

    kv = torch.load(dump / "kv.pt", map_location="cpu")
    layers = [int(x) for x in kv["layers"]]
    if layer not in layers:
        raise SystemExit(f"layer {layer} not in dump (layers {layers[0]}..{layers[-1]})")
    bs, t_len = int(kv["block_size"]), kv["key"].shape[1]
    n_blocks = (t_len + bs - 1) // bs
    if not 0 <= block < n_blocks:
        raise SystemExit(f"--block must be in [0, {n_blocks - 1}] ({t_len} tokens, block size {bs})")
    start, end = block * bs, min((block + 1) * bs, t_len)
    k = kv["key"][layers.index(layer), start:end].float()
    n_heads, head_dim = k.shape[1], k.shape[2]
    pieces = json.loads((dump / "tokens.json").read_text()).get("pieces") or []
    return (k.reshape(end - start, -1).numpy(), list(range(start, end)),
            [pieces[t] if t < len(pieces) else "" for t in range(start, end)], n_heads, head_dim)


def pairwise(k: np.ndarray):
    unit = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-30)
    cos = unit @ unit.T
    se = sign_exponent(k)
    match = (se[:, None, :] == se[None, :, :]).sum(axis=2)
    return cos, match


def classify(cos: float, match: int, cos_t: float, match_t: int) -> str:
    by_cos, by_bit = cos >= cos_t, match >= match_t
    return CLASSES[0] if by_cos and by_bit else CLASSES[1] if by_cos else CLASSES[2] if by_bit else CLASSES[3]


def plot(cos, match, df, tokens, n_elem, cos_t, match_t, title, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(t) for t in tokens]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, mat, name, vmin, vmax, fmt in (
        (axes[0], cos, "cosine similarity", -1, 1, "{:.2f}"),
        (axes[1], match / n_elem, f"bit match (equal sign+exponent / {n_elem})", 0, 1, "{:.2f}"),
    ):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(name)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j:
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=5,
                            color="white" if mat[i, j] < (vmin + vmax) / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmaps.png", dpi=150)
    plt.close(fig)

    colors = {"both": "tab:green", "cosine_only": "tab:orange", "bit_only": "tab:purple", "neither": "tab:gray"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        ax.scatter(sub["cosine"], sub["bit_match"], s=18, c=colors[cls], label=f"{cls} ({len(sub)})")
    ax.axvline(cos_t, color="k", ls="--", lw=0.8)
    ax.axhline(match_t, color="k", ls=":", lw=0.8)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel(f"bit match (equal sign+exponent of {n_elem})")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, n_elem)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dump = Path(args.dump)
    k, tokens, pieces, n_heads, head_dim = load_block(dump, args.layer, args.block)
    n_elem = n_heads * head_dim
    match_t = int(round(args.min_match * n_elem)) if args.min_match <= 1 else int(args.min_match)
    cos, match = pairwise(k)

    rows = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            rows.append({"token_i": tokens[i], "token_j": tokens[j], "piece_i": pieces[i],
                         "piece_j": pieces[j], "cosine": round(float(cos[i, j]), 5),
                         "bit_match": int(match[i, j]), "bit_match_ratio": round(match[i, j] / n_elem, 5),
                         "class": classify(cos[i, j], match[i, j], args.cos_threshold, match_t)})
    df = pd.DataFrame(rows)
    pearson = float(np.corrcoef(df["cosine"], df["bit_match"])[0, 1]) if len(df) > 2 else float("nan")
    spearman = float(df["cosine"].rank().corr(df["bit_match"].rank())) if len(df) > 2 else float("nan")
    counts = {c: int((df["class"] == c).sum()) for c in CLASSES}
    summary = {
        "dump": str(dump), "layer": args.layer, "block": args.block, "tokens": [tokens[0], tokens[-1]],
        "elements_per_slot": n_elem, "kv_heads": n_heads, "head_dim": head_dim, "pairs": len(df),
        "cos_threshold": args.cos_threshold, "bit_match_threshold": match_t,
        "pearson": round(pearson, 4), "spearman": round(spearman, 4), "classes": counts,
        "cosine": df["cosine"].describe().round(4).to_dict(),
        "bit_match": df["bit_match"].describe().round(1).to_dict(),
    }

    out_dir = Path(args.out_dir) if args.out_dir else dump / "block_similarity" / f"layer{args.layer}_block{args.block}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pairs.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    title = (f"{dump.name} layer {args.layer} block {args.block} (tokens {tokens[0]}-{tokens[-1]}), "
             f"K {n_heads}x{head_dim}")
    plot(cos, match, df, tokens, n_elem, args.cos_threshold, match_t, title, out_dir)

    print(f"{title}: {len(df)} pairs")
    print(f"  cosine     min {df['cosine'].min():.3f}  median {df['cosine'].median():.3f}  max {df['cosine'].max():.3f}")
    print(f"  bit match  min {df['bit_match'].min()}  median {df['bit_match'].median():.0f}  "
          f"max {df['bit_match'].max()}  (of {n_elem})")
    print(f"  correlation: Pearson {pearson:.3f}, Spearman {spearman:.3f}")
    print(f"  thresholds: cosine >= {args.cos_threshold}, bit match >= {match_t}")
    for c in CLASSES:
        print(f"    {c:<12} {counts[c]:>4}")
    top = df.sort_values("cosine", ascending=False).head(5)
    print("  highest-cosine pairs:")
    for r in top.to_dict("records"):
        print(f"    {r['token_i']:>6} {r['piece_i']!r:<14} {r['token_j']:>6} {r['piece_j']!r:<14} "
              f"cos {r['cosine']:.3f}  bit {r['bit_match']} ({r['bit_match_ratio']:.1%})  {r['class']}")
    print(f"wrote {out_dir}/pairs.csv, heatmaps.png, scatter.png, summary.json")


if __name__ == "__main__":
    main()
