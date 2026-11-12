#!/usr/bin/env python3
"""How many K slots of one KV block get merged by cosine similarity vs by bit match.

Takes one KV block (block_size consecutive tokens, block t // block_size) of
one layer of a kv_dump.py result. Tokens arrive in order, as in
block_merge.py: a new token is compared with the kept earlier tokens of the
block and merged (dropped) if its best match reaches the threshold. This is
run twice on the same K vectors (all kv_heads x head_dim elements, e.g.
8 x 128 = 1024), once with each criterion:
    cosine     cosine similarity >= --cos-threshold
    bit match  equal bf16 sign+exponent elements >= --min-match (block_merge.py)
and the merged tokens of the two are compared. Prints only; writes nothing.

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

from block_merge import sign_exponent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dump", help="kv_dump.py result folder (kv.pt, tokens.json).")
    p.add_argument("--layer", type=int, required=True, help="Layer id (as in kv.pt 'layers').")
    p.add_argument("--block", type=int, required=True,
                   help="Logical block index: tokens [block*block_size, (block+1)*block_size).")
    p.add_argument("--cos-threshold", type=float, default=0.85, help="Merge if cosine >= this.")
    p.add_argument("--min-match", type=float, default=0.5,
                   help="Merge if equal (sign, exponent) elements >= this: a fraction (<= 1) or a count.")
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


def sequential_merge(score: np.ndarray, threshold: float):
    """Merge each token into its best-scoring kept earlier token if score >= threshold.

    Returns (merged_into per token, -1 if kept; best score against kept earlier tokens).
    """
    n = len(score)
    kept = np.ones(n, dtype=bool)
    merged_into = np.full(n, -1)
    best = np.full(n, np.nan)
    for i in range(1, n):
        cand = np.flatnonzero(kept[:i])
        j = cand[int(score[i, cand].argmax())]
        best[i] = score[i, j]
        if best[i] >= threshold:
            kept[i] = False
            merged_into[i] = j
    return merged_into, best


def main() -> None:
    args = parse_args()
    dump = Path(args.dump)
    k, tokens, pieces, n_heads, head_dim = load_block(dump, args.layer, args.block)
    n_elem = n_heads * head_dim
    match_t = int(round(args.min_match * n_elem)) if args.min_match <= 1 else int(args.min_match)

    unit = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-30)
    cos = unit @ unit.T
    se = sign_exponent(k)
    match = (se[:, None, :] == se[None, :, :]).sum(axis=2)

    into_cos, best_cos = sequential_merge(cos, args.cos_threshold)
    into_bit, best_bit = sequential_merge(match, match_t)
    by_cos, by_bit = into_cos >= 0, into_bit >= 0

    print(f"{dump}  layer {args.layer}  block {args.block} (tokens {tokens[0]}-{tokens[-1]}), "
          f"K {n_heads}x{head_dim} = {n_elem} elements")
    print(f"criteria: cosine >= {args.cos_threshold}  |  bit match >= {match_t}/{n_elem} "
          f"({match_t / n_elem:.0%})\n")
    print(f"{'token':>7} {'piece':<16} {'best cos':>8} {'merged (cos)':>13} "
          f"{'best bit':>9} {'merged (bit)':>13}")
    for i, t in enumerate(tokens):
        if i == 0:
            print(f"{t:>7} {pieces[i]!r:<16} {'(first token of the block, always kept)':>45}")
            continue
        cos_col = f"-> {tokens[into_cos[i]]}" if by_cos[i] else "kept"
        bit_col = f"-> {tokens[into_bit[i]]}" if by_bit[i] else "kept"
        print(f"{t:>7} {pieces[i]!r:<16} {best_cos[i]:>8.3f} {cos_col:>13} "
              f"{int(best_bit[i]):>5} ({best_bit[i] / n_elem:>4.0%}) {bit_col:>9}")

    print(f"\nmerged K slots (of {len(tokens)})")
    print(f"  by cosine     {int(by_cos.sum()):>3}")
    print(f"  by bit match  {int(by_bit.sum()):>3}")
    print(f"  by both       {int((by_cos & by_bit).sum()):>3}")
    print(f"  cosine only   {int((by_cos & ~by_bit).sum()):>3}")
    print(f"  bit only      {int((~by_cos & by_bit).sum()):>3}")


if __name__ == "__main__":
    main()
