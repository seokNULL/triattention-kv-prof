#!/usr/bin/env python3
"""Quantify and plot the holes similarity-based merging would leave in vLLM blocks.

Replays a kv_dump.py dump token by token with an append-time policy:

    token t is merged (its slot becomes a hole) in (layer, kv_head) if some
    still-kept earlier token s in scope has cos(K_t, K_s) >= tau_k and
    cos(V_t, V_s) >= tau_v; with --accumulate mean, K_s/V_s become the
    running mean of the tokens merged into s.

Prompt tokens are always kept. The policy runs for every tau in --taus and
reports, per tau, holes at each granularity of the vLLM layout

    kv_cache[layer] = [K/V, block, slot(16), kv_head, head_dim]

    head-slot    (token, kv_head) entries removed / all entries
    token-slot   slots whose every kv_head is removed, per layer
    layer-block  (layer, block) pairs with every entry removed
    block id     blocks empty in every layer: what vLLM could free as is
    compacted    blocks still needed if each (layer, head) packs its kept
                 tokens to the front (TriAttention-style per-head compaction)
    DRAM rows    rows of --row-bytes holding only holes / some holes / none,
                 for the NHD (slot-major) and HND (head-major) block layouts,
                 assuming each 32KB-style (layer, K/V, block) chunk maps to
                 consecutive rows

Outputs go to <dump>/holes/: report.json and PNG figures.

Example:
    D=vllm_runner/kv_dumps/Qwen3-8B/fullkv/problem0
    python vllm_runner/hole_analysis.py $D
    python vllm_runner/hole_analysis.py $D --taus 0.8,0.9,0.95 --scope block --plot-tau 0.9
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dump_dir")
    p.add_argument("--taus", default="0.7,0.8,0.9,0.95,0.99", help="K thresholds to sweep.")
    p.add_argument("--tau-v", type=float, default=None, help="V threshold (default: same as tau).")
    p.add_argument(
        "--scope",
        default="all",
        help="Which kept tokens a new token may merge into: all, block (same vLLM block), "
        "or window:N (the previous N positions).",
    )
    p.add_argument("--accumulate", default="mean", choices=["mean", "none"],
                   help="mean: merged-into vectors become running means; none: keep them unchanged.")
    p.add_argument("--row-bytes", type=int, default=1024, help="DRAM row size for the row view.")
    p.add_argument("--layers", default="all", help="Dumped layers to analyze, e.g. 0,18,35 (default: all).")
    p.add_argument("--plot-tau", type=float, default=None, help="tau for the occupancy maps (default: middle of --taus).")
    p.add_argument("--plot-layer", type=int, default=None, help="Layer for the token x head map (default: middle dumped layer).")
    p.add_argument("--plot-tokens", type=int, default=512, help="Tokens shown in the token x head map.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ------------------------------------------------------------------ policy


def scope_mask(scope: str, t: int, block_size: int, device) -> torch.Tensor | None:
    """Positions < t a token at position t may merge into (None = all)."""
    if scope == "all":
        return None
    pos = torch.arange(t, device=device)
    if scope == "block":
        return pos >= (t // block_size) * block_size
    if scope.startswith("window:"):
        return pos >= t - int(scope.split(":")[1])
    raise SystemExit(f"unknown --scope {scope}")


def simulate(k: torch.Tensor, v: torch.Tensor, start: int, tau_k: float, tau_v: float,
             scope: str, accumulate: str, block_size: int) -> torch.Tensor:
    """k, v: [L, H, T, D]. Returns kept: [L, H, T] bool."""
    n_layers, n_heads, t_len, _ = k.shape
    unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # noqa: E731
    # Running sums when accumulating; their direction is the running mean's.
    kc, vc = k.clone(), v.clone()
    kept = torch.ones(n_layers, n_heads, t_len, dtype=torch.bool, device=k.device)
    ln = torch.arange(n_layers, device=k.device).view(-1, 1)
    hn = torch.arange(n_heads, device=k.device).view(1, -1)
    for t in range(max(start, 1), t_len):
        kt, vt = unit(k[:, :, t]), unit(v[:, :, t])  # [L, H, D]
        sim_k = torch.einsum("lhd,lhsd->lhs", kt, unit(kc[:, :, :t]))
        allowed = kept[:, :, :t]
        mask = scope_mask(scope, t, block_size, k.device)
        if mask is not None:
            allowed = allowed & mask
        sim_k = sim_k.masked_fill(~allowed, -2.0)
        best, arg = sim_k.max(dim=-1)  # [L, H]
        sim_v = (vt * unit(vc[ln, hn, arg])).sum(-1)
        merge = (best >= tau_k) & (sim_v >= tau_v)
        if not merge.any():
            continue
        kept[:, :, t] = ~merge
        if accumulate == "mean":
            l_idx, h_idx = merge.nonzero(as_tuple=True)
            s_idx = arg[l_idx, h_idx]
            kc[l_idx, h_idx, s_idx] += k[l_idx, h_idx, t]
            vc[l_idx, h_idx, s_idx] += v[l_idx, h_idx, t]
    return kept


# ------------------------------------------------------------------ metrics


def hole_metrics(kept: torch.Tensor, block_size: int, head_dim: int, row_bytes: int) -> dict:
    """kept: [L, H, T] bool -> hole statistics at every layout granularity."""
    n_layers, n_heads, t_len = kept.shape
    n_blocks = math.ceil(t_len / block_size)
    pad = n_blocks * block_size - t_len
    # Padding = slots past the last token: unused, counted as neither live nor hole.
    live = torch.nn.functional.pad(kept, (0, pad), value=False)
    valid = torch.nn.functional.pad(torch.ones_like(kept), (0, pad), value=False)
    live_b = live.view(n_layers, n_heads, n_blocks, block_size)  # [L, H, B, S]
    valid_b = valid.view(n_layers, n_heads, n_blocks, block_size)

    slot_live = live_b.any(dim=1)  # [L, B, S]: slot has a live head
    layer_block_live = slot_live.any(dim=-1)  # [L, B]
    block_live = layer_block_live.any(dim=0)  # [B]
    live_per_head = kept.sum(dim=-1)  # [L, H]
    compacted_blocks = math.ceil(live_per_head.max().item() / block_size)

    # DRAM rows inside each (layer, K/V, block) chunk; K and V share the pattern.
    entry_bytes = head_dim * 2
    per_row = max(1, row_bytes // entry_bytes)
    rows = {}
    for name, order in (("NHD", live_b.permute(0, 2, 3, 1)), ("HND", live_b.permute(0, 2, 1, 3))):
        flat = order.reshape(n_layers, n_blocks, -1)  # [L, B, entries] in memory order
        vflat = (valid_b.permute(0, 2, 3, 1) if name == "NHD" else valid_b.permute(0, 2, 1, 3)).reshape(
            n_layers, n_blocks, -1)
        n_entries = flat.shape[-1]
        n_rows = math.ceil(n_entries / per_row)
        fpad = n_rows * per_row - n_entries
        r_live = torch.nn.functional.pad(flat, (0, fpad)).view(n_layers, n_blocks, n_rows, per_row)
        r_valid = torch.nn.functional.pad(vflat, (0, fpad)).view(n_layers, n_blocks, n_rows, per_row)
        used = r_valid.any(-1)
        n_live = r_live.sum(-1)
        n_valid = r_valid.sum(-1)
        total = used.sum().item()
        rows[name] = {
            "entries_per_row": per_row,
            "empty": ((n_live == 0) & used).sum().item() / total,
            "partial": ((n_live > 0) & (n_live < n_valid) & used).sum().item() / total,
            "full": ((n_live == n_valid) & used).sum().item() / total,
        }
    total_entries = n_layers * n_heads * t_len
    total_slots = n_layers * t_len
    return {
        "head_slot_holes": 1 - kept.sum().item() / total_entries,
        "token_slot_holes": 1 - (slot_live & valid_b[:, 0]).sum().item() / total_slots,
        "free_layer_blocks": 1 - layer_block_live.sum().item() / (n_layers * n_blocks),
        "free_block_ids": int((~block_live).sum().item()),
        "blocks": n_blocks,
        "blocks_after_head_compaction": compacted_blocks,
        "kept_tokens_per_head": {"min": int(live_per_head.min()), "max": int(live_per_head.max()),
                                 "mean": round(live_per_head.float().mean().item(), 1)},
        "dram_rows": rows,
    }


def similarity_profile(k: torch.Tensor, v: torch.Tensor, start: int) -> dict:
    """Best earlier match (all earlier tokens) per generated token: [L, H, T-start]."""
    unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # noqa: E731
    k, v = unit(k), unit(v)
    t_len = k.shape[2]
    best_k, v_at = [], []
    for layer in range(k.shape[0]):
        sim = k[layer, :, start:] @ k[layer].transpose(1, 2)  # [H, T-start, T]
        rows = torch.arange(start, t_len, device=k.device).view(1, -1, 1)
        cols = torch.arange(t_len, device=k.device).view(1, 1, -1)
        best, arg = sim.masked_fill(cols >= rows, -2.0).max(-1)
        vv = torch.gather(v[layer], 1, arg.unsqueeze(-1).expand(-1, -1, v.shape[-1]))
        best_k.append(best)
        v_at.append((v[layer, :, start:] * vv).sum(-1))
    return {"best_k": torch.stack(best_k).cpu(), "v_at_best": torch.stack(v_at).cpu()}


# ------------------------------------------------------------------ plots


def make_plots(out: Path, sims: dict, sweep: dict, kept_plot: torch.Tensor, layers: list[int],
               plot_layer_idx: int, plot_tau: float, block_size: int, plot_tokens: int, start: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = sorted(float(t) for t in sweep)

    # 1. similarity distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = torch.linspace(-1, 1, 81)
    for ax, key, title in ((axes[0], "best_k", "best earlier K cosine"),
                           (axes[1], "v_at_best", "V cosine at best K match")):
        data = sims[key]
        for idx in sorted({0, len(layers) // 2, len(layers) - 1}):
            ax.hist(data[idx].flatten().numpy(), bins=bins.numpy(), histtype="step", density=True,
                    label=f"layer {layers[idx]}")
        ax.hist(data.flatten().numpy(), bins=bins.numpy(), histtype="stepfilled", alpha=0.25,
                density=True, color="gray", label="all layers")
        ax.set_title(title)
        ax.set_xlabel("cosine similarity")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density (generated tokens x heads)")
    fig.tight_layout()
    fig.savefig(out / "1_similarity_hist.png", dpi=150)
    plt.close(fig)

    # 2. tau sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series = [("head_slot_holes", "head-slot holes"), ("token_slot_holes", "token-slot holes"),
              ("free_layer_blocks", "free (layer, block)")]
    for key, label in series:
        axes[0].plot(taus, [sweep[str(t)][key] for t in taus], marker="o", label=label)
    blocks = sweep[str(taus[0])]["blocks"]
    axes[0].plot(taus, [sweep[str(t)]["free_block_ids"] / blocks for t in taus], marker="o",
                 label="free block ids (all layers)")
    axes[0].plot(taus, [1 - sweep[str(t)]["blocks_after_head_compaction"] / blocks for t in taus],
                 marker="s", linestyle="--", label="saved by per-head compaction")
    axes[0].set_xlabel("tau (K and V cosine threshold)")
    axes[0].set_ylabel("fraction")
    axes[0].set_title("holes vs. merge threshold")
    axes[0].legend(fontsize=8)
    for layout, style in (("NHD", "-"), ("HND", "--")):
        for part in ("empty", "partial"):
            axes[1].plot(taus, [sweep[str(t)]["dram_rows"][layout][part] for t in taus],
                         linestyle=style, marker="o", label=f"{layout} {part} rows")
    axes[1].set_xlabel("tau")
    axes[1].set_ylabel("fraction of DRAM rows")
    axes[1].set_title("DRAM rows: empty = reclaimable, partial = sub-row moves")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "2_tau_sweep.png", dpi=150)
    plt.close(fig)

    # 3. occupancy per (layer, block)
    n_layers, n_heads, t_len = kept_plot.shape
    n_blocks = math.ceil(t_len / block_size)
    pad = n_blocks * block_size - t_len
    live = torch.nn.functional.pad(kept_plot.float(), (0, pad), value=float("nan"))
    occ = live.view(n_layers, n_heads, n_blocks, block_size).nanmean(dim=(1, 3))
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(occ.numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(n_layers), [str(l) for l in layers], fontsize=6)
    ax.set_xlabel("block (logical order)")
    ax.set_ylabel("layer")
    ax.set_title(f"live fraction per (layer, block), tau={plot_tau}  (1 = no holes)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "3_block_occupancy.png", dpi=150)
    plt.close(fig)

    # 4. token x head map for one layer
    t_end = min(t_len, start + plot_tokens)
    grid = kept_plot[plot_layer_idx, :, :t_end].float().numpy()
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.imshow(grid, aspect="auto", cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    for b in range(block_size, t_end, block_size):
        ax.axvline(b - 0.5, color="tab:red", lw=0.4, alpha=0.6)
    ax.axvline(start - 0.5, color="tab:blue", lw=1.2)
    ax.set_xlabel("token position (red lines = block boundaries, blue = end of prompt)")
    ax.set_ylabel("kv_head")
    ax.set_title(f"layer {layers[plot_layer_idx]}: white = kept, black = hole (tau={plot_tau})")
    fig.tight_layout()
    fig.savefig(out / "4_layer_token_head_map.png", dpi=150)
    plt.close(fig)

    # 5. histogram of per-(layer, block) live fraction
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(occ.flatten().numpy(), bins=32, range=(0, 1), color="tab:purple")
    ax.set_xlabel("live fraction of a (layer, block)")
    ax.set_ylabel("count")
    ax.set_title(f"block fragmentation, tau={plot_tau}")
    fig.tight_layout()
    fig.savefig(out / "5_block_fill_hist.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> None:
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    dump = torch.load(dump_dir / "kv.pt", map_location="cpu")
    meta = json.loads((dump_dir / "meta.json").read_text())
    if meta.get("method") != "fullkv":
        print("[note] dump is not fullkv: the cache is already compressed, positions are not contiguous")
    layers = list(dump["layers"])
    rows = list(range(len(layers)))
    if args.layers != "all":
        wanted = [int(x) for x in args.layers.split(",")]
        missing = [l for l in wanted if l not in layers]
        if missing:
            raise SystemExit(f"layers {missing} were not dumped; available: {layers}")
        rows, layers = [layers.index(l) for l in wanted], wanted
    block_size, t_len = dump["block_size"], dump["kv_len"]
    start = min(meta["num_prompt_tokens"], t_len - 1)
    k = dump["key"][rows].to(args.device).float().permute(0, 2, 1, 3).contiguous()  # [L, H, T, D]
    v = dump["value"][rows].to(args.device).float().permute(0, 2, 1, 3).contiguous()
    head_dim = k.shape[-1]
    taus = [float(t) for t in args.taus.split(",")]
    plot_tau = args.plot_tau if args.plot_tau is not None else taus[len(taus) // 2]
    if plot_tau not in taus:
        taus = sorted(taus + [plot_tau])
    print(f"{meta.get('model')}: {len(layers)} layers x {k.shape[1]} kv_heads, {t_len} tokens "
          f"({start} prompt), block_size {block_size}, scope={args.scope}, accumulate={args.accumulate}")

    sims = similarity_profile(k, v, start)
    sweep, kept_plot = {}, None
    for tau in taus:
        kept = simulate(k, v, start, tau, args.tau_v if args.tau_v is not None else tau,
                        args.scope, args.accumulate, block_size)
        sweep[str(tau)] = hole_metrics(kept, block_size, head_dim, args.row_bytes)
        if tau == plot_tau:
            kept_plot = kept.cpu()

    print(f"\n{'tau':>5} {'head-slot':>9} {'tok-slot':>8} {'free L-blk':>10} {'free blk id':>11} "
          f"{'compacted':>10} {'NHD empty/partial':>18} {'HND empty/partial':>18}")
    for tau in taus:
        m = sweep[str(tau)]
        nhd, hnd = m["dram_rows"]["NHD"], m["dram_rows"]["HND"]
        print(f"{tau:>5} {m['head_slot_holes']:>9.3f} {m['token_slot_holes']:>8.3f} "
              f"{m['free_layer_blocks']:>10.3f} {m['free_block_ids']:>5}/{m['blocks']:<5} "
              f"{m['blocks_after_head_compaction']:>4}/{m['blocks']:<5} "
              f"{nhd['empty']:>8.3f}/{nhd['partial']:<8.3f} {hnd['empty']:>8.3f}/{hnd['partial']:<8.3f}")

    out = dump_dir / "holes"
    out.mkdir(exist_ok=True)
    plot_layer = args.plot_layer if args.plot_layer is not None else layers[len(layers) // 2]
    if plot_layer not in layers:
        raise SystemExit(f"--plot-layer {plot_layer} was not dumped; available: {layers}")
    make_plots(out, sims, sweep, kept_plot, layers, layers.index(plot_layer), plot_tau,
               block_size, args.plot_tokens, start)
    report = {
        "settings": {"scope": args.scope, "accumulate": args.accumulate, "tau_v": args.tau_v,
                     "row_bytes": args.row_bytes, "prompt_tokens_kept": start},
        "meta": {k_: meta.get(k_) for k_ in ("model", "method", "problem", "num_prompt_tokens",
                                             "num_output_tokens", "kv_len", "block_size")},
        "sweep": sweep,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}/report.json and 1_..5_*.png")


if __name__ == "__main__":
    main()
