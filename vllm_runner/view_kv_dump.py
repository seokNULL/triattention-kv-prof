#!/usr/bin/env python3
"""Print or export a kv_dump.py dump as rows: layer, head, token, ..., values.

Every row is one (layer, kv_head, token, K or V) vector, with the physical
vLLM block/slot the token occupied and its decoded token piece:

    layer head token block slot kv piece          d0       d1  ...
       18    0     0    11    0  K <|im_start|>  0.0123  -1.2344  ...

Examples:
    D=experiments/kv_dumps/Qwen3-8B/fullkv/problem0
    python vllm_runner/view_kv_dump.py $D --layer 18 --head 0 --tokens 0:8
    python vllm_runner/view_kv_dump.py $D --layer 0,18 --kv K --dims 16
    python vllm_runner/view_kv_dump.py $D --layer 18 --csv kv_layer18.csv      # all dims
    python vllm_runner/view_kv_dump.py $D --npz kv_float32.npz                 # numpy arrays
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


def parse_selection(text: str | None, size: int, name: str) -> list[int]:
    """'all', '3', '0,4,7' or 'start:stop' -> sorted index list."""
    if text is None or text == "all":
        return list(range(size))
    picked: list[int] = []
    for part in text.split(","):
        if ":" in part:
            start, stop = part.split(":")
            picked.extend(range(int(start or 0), min(int(stop) if stop else size, size)))
        else:
            picked.append(int(part))
    bad = [i for i in picked if not 0 <= i < size]
    if bad:
        raise SystemExit(f"{name} out of range [0, {size - 1}]: {bad}")
    return sorted(set(picked))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dump_dir")
    p.add_argument("--layer", default="all", help="Model layer ids, e.g. 18 or 0,18,35 or 0:4.")
    p.add_argument("--head", default="all", help="KV head ids, e.g. 0 or 0:4.")
    p.add_argument("--tokens", default="0:16", help="KV positions, e.g. 0:16, 100:120 or all.")
    p.add_argument("--kv", default="KV", choices=["K", "V", "KV"])
    p.add_argument("--dims", type=int, default=8, help="Values printed per row (0 = all). CSV always has all.")
    p.add_argument("--precision", type=int, default=4)
    p.add_argument("--csv", help="Write the selected rows with all dims to this CSV file.")
    p.add_argument("--npz", help="Write key/value as float32 numpy arrays (selection applied).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    dump = torch.load(dump_dir / "kv.pt", map_location="cpu")
    tokens_path = dump_dir / "tokens.json"
    tokens = json.loads(tokens_path.read_text()) if tokens_path.exists() else {}
    pieces = tokens.get("pieces") or []

    dumped_layers = list(dump["layers"])
    n_layers, kv_len, n_heads, head_dim = dump["key"].shape
    block_ids, block_size = dump["block_ids"], dump["block_size"]

    wanted_layers = (
        dumped_layers if args.layer == "all"
        else [int(x) for x in parse_selection(args.layer, max(dumped_layers) + 1, "layer")]
    )
    missing = [l for l in wanted_layers if l not in dumped_layers]
    if missing:
        raise SystemExit(f"layers {missing} were not dumped; available: {dumped_layers}")
    layer_rows = [dumped_layers.index(l) for l in wanted_layers]
    heads = parse_selection(args.head, n_heads, "head")
    toks = parse_selection(args.tokens, kv_len, "token")
    kinds = [k for k in "KV" if k in args.kv]
    tensors = {"K": dump["key"], "V": dump["value"]}

    print(
        f"# {dump_dir}: layers {dumped_layers[0]}..{dumped_layers[-1]} ({n_layers} dumped), "
        f"{n_heads} kv_heads, head_dim {head_dim}, {kv_len} tokens, block_size {block_size}",
        file=sys.stderr,
    )

    if args.npz:
        import numpy as np

        sel = (torch.tensor(layer_rows)[:, None, None], torch.tensor(toks)[None, :, None],
               torch.tensor(heads)[None, None, :])
        np.savez(
            args.npz,
            key=dump["key"][sel].float().numpy(),
            value=dump["value"][sel].float().numpy(),
            layers=np.array(wanted_layers),
            heads=np.array(heads),
            tokens=np.array(toks),
            block=np.array([block_ids[t // block_size] for t in toks]),
            slot=np.array([t % block_size for t in toks]),
        )
        print(f"# wrote {args.npz}: key/value [layer, token, head, dim] = "
              f"[{len(layer_rows)}, {len(toks)}, {len(heads)}, {head_dim}] float32", file=sys.stderr)
        if not args.csv:
            return

    def rows():
        for li, layer in zip(layer_rows, wanted_layers):
            for head in heads:
                for t in toks:
                    for kind in kinds:
                        vec = tensors[kind][li, t, head].float()
                        yield (layer, head, t, block_ids[t // block_size], t % block_size, kind,
                               pieces[t] if t < len(pieces) else "", vec)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "head", "token", "block", "slot", "kv", "piece"]
                            + [f"d{i}" for i in range(head_dim)])
            n = 0
            for layer, head, t, block, slot, kind, piece, vec in rows():
                writer.writerow([layer, head, t, block, slot, kind, piece]
                                + [f"{x:.{args.precision}g}" for x in vec.tolist()])
                n += 1
        print(f"# wrote {args.csv}: {n} rows x {head_dim} dims", file=sys.stderr)
        return

    shown = head_dim if args.dims <= 0 else min(args.dims, head_dim)
    width = args.precision + 6
    header = (f"{'layer':>5} {'head':>4} {'token':>5} {'block':>5} {'slot':>4} {'kv':>2} "
              f"{'piece':<14}" + "".join(f"{f'd{i}':>{width}}" for i in range(shown))
              + ("  ..." if shown < head_dim else ""))
    print(header)
    for layer, head, t, block, slot, kind, piece, vec in rows():
        piece = repr(piece)[1:-1][:14]
        values = "".join(f"{x:>{width}.{args.precision}f}" for x in vec[:shown].tolist())
        print(f"{layer:>5} {head:>4} {t:>5} {block:>5} {slot:>4} {kind:>2} {piece:<14}{values}"
              + ("  ..." if shown < head_dim else ""))


if __name__ == "__main__":
    main()
