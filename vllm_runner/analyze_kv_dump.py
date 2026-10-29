#!/usr/bin/env python3
"""Example analysis of a kv_dump.py dump: KV similarity and simulated holes.

A starting point to adapt, not a finished metric suite. For every dumped
layer and KV head it reports:

1. adjacent:  cos(K_t, K_{t-1}) and cos(V_t, V_{t-1})
2. in-block:  mean pairwise K cosine inside each vLLM block vs. random pairs
3. append-time redundancy: for each generated token, the best cosine to any
   earlier token in the cache (K, and V at the K-argmax), i.e. what an
   append-time accumulate/append/evict decision would see
4. simulated holes: if every generated token whose best earlier match has
   cos >= tau (K and V) were merged away, how many slots become holes at
   three granularities of the vLLM layout [block, slot, kv_head, head_dim]:
     head slot  one (token, kv_head) entry          (per-head eviction)
     token slot all kv_heads of a token in a layer  (vLLM slot granularity)
     block      all slots of a block in a layer     (block can be freed)

Keys in the vLLM cache are post-RoPE. The redundancy test compares each
token with all earlier tokens, not only with tokens kept by a policy, so it
is an upper bound on what a sequential policy would merge.

Example:
    python vllm_runner/analyze_kv_dump.py vllm_runner/kv_dumps/Qwen3-8B/fullkv/problem0
    python vllm_runner/analyze_kv_dump.py <dump> --taus 0.9,0.95 --plot 18:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dump_dir")
    p.add_argument("--taus", default="0.9,0.95,0.99", help="Merge thresholds for the hole simulation.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--plot", default=None, help="layer:head to save a K-similarity heatmap for.")
    p.add_argument("--plot-tokens", type=int, default=512, help="Tokens shown in the heatmap.")
    return p.parse_args()


def normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def best_previous_match(k: torch.Tensor, v: torch.Tensor, start: int):
    """For tokens t >= start: max_{s<t} cos(K_t, K_s) and cos(V_t, V_argmax).

    k, v: [T, H, D] normalized. Returns two [T - start, H] tensors.
    """
    t_len = k.shape[0]
    k_h = k.transpose(0, 1)  # [H, T, D]
    v_h = v.transpose(0, 1)
    sim = k_h[:, start:] @ k_h.transpose(1, 2)  # [H, T-start, T]
    rows = torch.arange(start, t_len, device=k.device).view(1, -1, 1)
    cols = torch.arange(t_len, device=k.device).view(1, 1, -1)
    sim = sim.masked_fill(cols >= rows, -2.0)
    k_best, arg = sim.max(dim=-1)  # [H, T-start]
    v_match = (v_h[:, start:] * torch.gather(v_h, 1, arg.unsqueeze(-1).expand(-1, -1, v_h.shape[-1]))).sum(-1)
    return k_best.transpose(0, 1), v_match.transpose(0, 1)


def in_block_similarity(k: torch.Tensor, block_size: int) -> tuple[float, float]:
    """Mean pairwise K cosine inside full blocks, and for random token pairs."""
    n_blocks = k.shape[0] // block_size
    if n_blocks == 0:
        return float("nan"), float("nan")
    blocks = k[: n_blocks * block_size].view(n_blocks, block_size, *k.shape[1:])  # [B, S, H, D]
    blocks = blocks.permute(0, 2, 1, 3)  # [B, H, S, D]
    sim = blocks @ blocks.transpose(-1, -2)  # [B, H, S, S]
    off_diag = ~torch.eye(block_size, dtype=torch.bool, device=k.device)
    in_block = sim[..., off_diag].mean().item()
    gen = torch.Generator(device="cpu").manual_seed(0)
    i = torch.randint(0, k.shape[0], (4096,), generator=gen).to(k.device)
    j = torch.randint(0, k.shape[0], (4096,), generator=gen).to(k.device)
    keep = i != j
    rand = (k[i[keep]] * k[j[keep]]).sum(-1).mean().item()
    return in_block, rand


def simulate_holes(merged: torch.Tensor, start: int, kv_len: int, block_size: int) -> dict:
    """merged: [T - start, H] bool for generated tokens. Prompt tokens are kept."""
    n_heads = merged.shape[1]
    removed = torch.zeros(kv_len, n_heads, dtype=torch.bool, device=merged.device)
    removed[start:] = merged
    n_blocks = (kv_len + block_size - 1) // block_size
    padded = torch.zeros(n_blocks * block_size, n_heads, dtype=torch.bool, device=merged.device)
    padded[:kv_len] = removed
    slot_free = padded.all(dim=1)  # every kv_head of the token removed
    block_free = slot_free.view(n_blocks, block_size).all(dim=1)
    return {
        "head_slot_holes": removed.float().mean().item(),
        "token_slot_holes": slot_free[:kv_len].float().mean().item(),
        "free_blocks": int(block_free.sum().item()),
        "blocks": n_blocks,
    }


def quantiles(x: torch.Tensor) -> dict:
    x = x.flatten().float()
    q = torch.quantile(x[torch.randperm(x.numel(), device=x.device)[:100_000]],
                       torch.tensor([0.1, 0.5, 0.9], device=x.device))
    return {"mean": round(x.mean().item(), 4), "p10": round(q[0].item(), 4),
            "p50": round(q[1].item(), 4), "p90": round(q[2].item(), 4)}


def plot_heatmap(k: torch.Tensor, layer: int, head: int, n_tokens: int, block_size: int, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kh = k[:n_tokens, head]
    sim = (kh @ kh.T).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
    for b in range(block_size, kh.shape[0], block_size):
        ax.axhline(b - 0.5, color="white", lw=0.2, alpha=0.4)
        ax.axvline(b - 0.5, color="white", lw=0.2, alpha=0.4)
    ax.set_title(f"K cosine similarity, layer {layer} kv_head {head} (first {kh.shape[0]} tokens)")
    ax.set_xlabel("token")
    ax.set_ylabel("token")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    dump = torch.load(dump_dir / "kv.pt", map_location="cpu")
    meta = json.loads((dump_dir / "meta.json").read_text())
    taus = [float(t) for t in args.taus.split(",")]
    block_size, kv_len = dump["block_size"], dump["kv_len"]
    start = min(meta["num_prompt_tokens"], kv_len - 1)
    layers = dump["layers"]
    print(
        f"model={meta['model']} method={meta['method']} kv_len={kv_len} "
        f"(prompt {meta['num_prompt_tokens']}) block_size={block_size} "
        f"layers={len(layers)} kv_heads={dump['key'].shape[2]} head_dim={dump['key'].shape[3]}"
    )
    if meta["method"] != "fullkv":
        print("[note] compressed cache: positions are not contiguous, adjacent/in-block stats mix tokens")

    per_layer = []
    for li, layer in enumerate(layers):
        k = normalize(dump["key"][li].to(args.device))  # [T, H, D]
        v = normalize(dump["value"][li].to(args.device))
        adj_k = (k[1:] * k[:-1]).sum(-1)
        adj_v = (v[1:] * v[:-1]).sum(-1)
        blk, rnd = in_block_similarity(k, block_size)
        k_best, v_match = best_previous_match(k, v, start)
        row = {
            "layer": layer,
            "adjacent_k": round(adj_k.mean().item(), 4),
            "adjacent_v": round(adj_v.mean().item(), 4),
            "in_block_k": round(blk, 4),
            "random_pair_k": round(rnd, 4),
            "best_prev_k": quantiles(k_best),
            "v_at_best_k": quantiles(v_match),
            "holes": {},
        }
        for tau in taus:
            merged = (k_best >= tau) & (v_match >= tau)
            row["holes"][str(tau)] = simulate_holes(merged, start, kv_len, block_size)
        per_layer.append(row)
        if args.plot and int(args.plot.split(":")[0]) == layer:
            head = int(args.plot.split(":")[1])
            path = dump_dir / f"k_sim_layer{layer}_head{head}.png"
            plot_heatmap(k, layer, head, args.plot_tokens, block_size, path)
            print(f"[plot] {path}")

    print(f"\n{'layer':>5} {'adjK':>6} {'adjV':>6} {'blockK':>7} {'randK':>6} "
          f"{'bestK p50':>9} {'V@bestK':>8}  " + "  ".join(f"merge@{t}" for t in taus))
    for row in per_layer:
        merges = "  ".join(
            f"{row['holes'][str(t)]['head_slot_holes']:>8.3f}" for t in taus
        )
        print(f"{row['layer']:>5} {row['adjacent_k']:>6.3f} {row['adjacent_v']:>6.3f} "
              f"{row['in_block_k']:>7.3f} {row['random_pair_k']:>6.3f} "
              f"{row['best_prev_k']['p50']:>9.3f} {row['v_at_best_k']['p50']:>8.3f}  {merges}")

    print("\nsimulated holes (mean over layers):")
    print(f"{'tau':>5} {'head-slot':>10} {'token-slot':>11} {'free blocks':>12}")
    summary = {"meta": meta, "per_layer": per_layer, "taus": {}}
    for tau in taus:
        hs = [r["holes"][str(tau)] for r in per_layer]
        head_slot = sum(h["head_slot_holes"] for h in hs) / len(hs)
        token_slot = sum(h["token_slot_holes"] for h in hs) / len(hs)
        free = sum(h["free_blocks"] for h in hs) / len(hs)
        summary["taus"][str(tau)] = {"head_slot_holes": head_slot, "token_slot_holes": token_slot,
                                     "free_blocks_per_layer": free, "blocks": hs[0]["blocks"]}
        print(f"{tau:>5} {head_slot:>10.3f} {token_slot:>11.3f} {free:>7.1f}/{hs[0]['blocks']}")
    out = dump_dir / "analysis.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
