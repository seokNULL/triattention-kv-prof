#!/usr/bin/env python3
"""How many K slots of one KV block get merged by cosine similarity vs by bit match.

Takes one KV block (block_size consecutive tokens, block t // block_size) of
one layer of a kv_dump.py result. Every token except the first of the block
is compared with the token right before it (consecutive-token merging, as in
KVMerger) on the same K vectors (all kv_heads x head_dim elements, e.g.
8 x 128 = 1024), and counts as merged under each criterion it meets:
    cosine     cosine similarity >= --cos-threshold
    bit match  equal bf16 sign+exponent elements >= --min-match (block_merge.py),
               or with --bits sign equal sign bits only
Prints only; writes nothing.

Works on vllm_CoT_analysis and vllm_MM_analysis dumps. Example (from vllm_CoT_analysis/):
    python block_similarity.py results/Qwen3-8B/aime24/problem0 --layer 18 --block 100
    python block_similarity.py results/Qwen3-8B/aime24/problem0 --layer 18 --block 100 \\
        --cos-threshold 0.9 --min-match 0.6
    python block_similarity.py results/Qwen3-8B/aime24/problem0 --layer 18 --block 100 --bits sign

With --bits sign and no --min-match, the bit threshold is the sign agreement a
zero-mean Gaussian pair at the cosine threshold has on average,
1 - arccos(cos) / pi (Sheppard's formula), e.g. 0.75 -> 77%, 0.85 -> 82%.
"""

from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument("--bits", default="sign-exponent", choices=["sign-exponent", "sign"],
                   help="Bits compared per element: sign + exponent (9 bits, block_merge.py) or sign only.")
    p.add_argument("--min-match", type=float, default=None,
                   help="Merge if elements with equal --bits >= this: a fraction (<= 1) or a count. "
                   "Default: 0.5 for sign-exponent; 1 - arccos(cos threshold) / pi for sign.")
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


def main() -> None:
    args = parse_args()
    dump = Path(args.dump)
    k, tokens, pieces, n_heads, head_dim = load_block(dump, args.layer, args.block)
    n_elem = n_heads * head_dim
    min_match = args.min_match
    if min_match is None:
        min_match = 0.5 if args.bits == "sign-exponent" else 1 - math.acos(args.cos_threshold) / math.pi
    match_t = int(round(min_match * n_elem)) if min_match <= 1 else int(min_match)

    # Similarity of token i with token i - 1 (index 0: no previous token in the block).
    unit = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-30)
    cos_prev = np.r_[np.nan, (unit[1:] * unit[:-1]).sum(axis=1)]
    se = sign_exponent(k)
    if args.bits == "sign":
        se = se >> 8
    bit_prev = np.r_[0, (se[1:] == se[:-1]).sum(axis=1)]
    by_cos = np.r_[False, cos_prev[1:] >= args.cos_threshold]
    by_bit = np.r_[False, bit_prev[1:] >= match_t]

    print(f"{dump}  layer {args.layer}  block {args.block} (tokens {tokens[0]}-{tokens[-1]}), "
          f"K {n_heads}x{head_dim} = {n_elem} elements")
    bits = "sign+exponent" if args.bits == "sign-exponent" else "sign"
    note = " (Sheppard equivalent of the cosine threshold)" if args.bits == "sign" and args.min_match is None else ""
    print(f"criteria: cosine >= {args.cos_threshold}  |  equal {bits} >= {match_t}/{n_elem} "
          f"({match_t / n_elem:.0%}){note}\n")
    print("each token vs the previous token of the block")
    print(f"{'token':>7} {'piece':<16} {'cos(prev)':>9} {'merged':>7} {'bit(prev)':>15} {'merged':>7}")
    for i, t in enumerate(tokens):
        if i == 0:
            print(f"{t:>7} {pieces[i]!r:<16} {'(first token of the block: no previous token)':>45}")
            continue
        print(f"{t:>7} {pieces[i]!r:<16} {cos_prev[i]:>9.3f} {'yes' if by_cos[i] else '-':>7} "
              f"{int(bit_prev[i]):>7} ({bit_prev[i] / n_elem:>4.0%}) {'yes' if by_bit[i] else '-':>7}")

    print(f"\nmerged K slots (of {len(tokens)}; {len(tokens) - 1} compared)")
    print(f"  by cosine     {int(by_cos.sum()):>3}")
    print(f"  by bit match  {int(by_bit.sum()):>3}")
    print(f"  by both       {int((by_cos & by_bit).sum()):>3}")
    print(f"  cosine only   {int((by_cos & ~by_bit).sum()):>3}")
    print(f"  bit only      {int((~by_cos & by_bit).sum()):>3}")


if __name__ == "__main__":
    main()
