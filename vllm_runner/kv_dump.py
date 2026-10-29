#!/usr/bin/env python3
"""Dump the paged KV cache of one quick-test problem generated with vLLM.

Runs a single request (one problem from data/quick.jsonl) so its blocks are
not reused by other requests, records the request's block table after every
engine step, and after generation copies its K/V out of vLLM's paged cache:

    <out>/kv.pt        key/value [layers, tokens, kv_heads, head_dim] (bf16),
                       in logical token order, plus block_ids/block_size so
                       token t lives in block block_ids[t // block_size],
                       slot t % block_size
    <out>/meta.json    model shape, lengths, layout, settings
    <out>/tokens.json  token ids and decoded pieces for every KV position
    <out>/trace.jsonl  per step: KV length and block table
    <out>/snapshots/   optional mid-generation dumps (--snapshot-every)

With --method triattention the dump is the compacted cache after the last
compression (positions are no longer contiguous); fullkv is the baseline.

Example:
    python vllm_runner/kv_dump.py --model Qwen3-8B --problem 0
    python vllm_runner/analyze_kv_dump.py experiments/kv_dumps/Qwen3-8B/fullkv/problem0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cloudpickle
import torch

from quick_test import PROBLEMS, build_settings
from run_benchmark import (
    MODEL_SPECS,
    REPO_ROOT,
    build_requests,
    configure_logging,
    configure_plugin_env,
    engine_kwargs,
    load_problems,
    sampling_params,
)

DUMP_ROOT = REPO_ROOT / "experiments" / "kv_dumps"
# Worker-side state lives on the model runner under this attribute.
TRACKER_ATTR = "_kv_dump_tracker"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--method", default="fullkv", choices=["fullkv", "triattention"])
    p.add_argument("--problem", type=int, default=0, help="Row of data/quick.jsonl.")
    p.add_argument("--max-length", type=int, default=4096, help="Prompt + generation tokens.")
    p.add_argument(
        "--layers",
        default="all",
        help="Layers to dump: 'all' or comma list such as 0,9,18,27,35.",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Also dump the cache every N KV tokens during generation (0 = final only).",
    )
    p.add_argument("--output-dir", default=None, help="Default: experiments/kv_dumps/<model>/<method>/problem<N>")
    # TriAttention / engine options, same meaning as in quick_test.py.
    p.add_argument("--budget", type=int, default=512)
    p.add_argument(
        "--pruning-mode",
        default="per_head",
        choices=["per_head", "per_layer_per_head", "per_layer"],
    )
    p.add_argument("--divide-length", type=int, default=128)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--protect-prefill", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-decisions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--block-size", type=int, default=16)
    args = p.parse_args()
    args.tensor_parallel_size = 1
    args.max_num_seqs = None
    args.max_num_batched_tokens = None
    return args


# ---------------------------------------------------------------- worker side
# These functions run inside the vLLM worker via LLM.collective_rpc. They
# receive the worker wrapper as their first argument.


def _model_runner(worker_wrapper):
    worker = getattr(worker_wrapper, "worker", worker_wrapper)
    return worker.model_runner


def _split_kv(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V views of one layer's cache, each [num_blocks, block_size, H, D]."""
    from triattention.vllm.runtime.kv_compaction import _split_kv_axes

    return _split_kv_axes(kv_cache)


def _sync(model_runner) -> None:
    """Wait for in-flight kernels that write the cache."""
    device = model_runner.kv_caches[0].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gather(model_runner, block_ids, kv_len, layers):
    """Copy a request's K/V to CPU in logical token order."""
    block_size = int(_split_kv(model_runner.kv_caches[0])[0].shape[1])
    blocks = torch.as_tensor(block_ids, dtype=torch.long)
    keys, values = [], []
    for layer in layers:
        k_cache, v_cache = _split_kv(model_runner.kv_caches[layer])
        idx = blocks.to(k_cache.device)
        k = k_cache.index_select(0, idx).flatten(0, 1)[:kv_len]
        v = v_cache.index_select(0, idx).flatten(0, 1)[:kv_len]
        keys.append(k.to("cpu", copy=True))
        values.append(v.to("cpu", copy=True))
    return torch.stack(keys), torch.stack(values), block_size


def _install_tracker(worker_wrapper, layers, snapshot_every, snapshot_dir):
    model_runner = _model_runner(worker_wrapper)
    num_layers = len(model_runner.kv_caches)
    layers = list(range(num_layers)) if layers is None else layers
    state = {
        "layers": layers,
        "num_layers": num_layers,
        "steps": [],
        "last": {},
        "next_snapshot": {},
    }
    setattr(model_runner, TRACKER_ATTR, state)
    original = model_runner.execute_model

    def execute_model(scheduler_output, *a, **kw):
        out = original(scheduler_output, *a, **kw)
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        for req_id, n_sched in scheduled.items():
            req = model_runner.requests.get(req_id)
            if req is None:
                continue
            kv_len = int(req.num_computed_tokens) + int(n_sched)
            block_ids = list(req.block_ids[0])
            state["last"][req_id] = {"kv_len": kv_len, "block_ids": block_ids}
            state["steps"].append({"req_id": req_id, "kv_len": kv_len, "block_ids": block_ids})
            if snapshot_every > 0:
                target = state["next_snapshot"].setdefault(req_id, snapshot_every)
                if kv_len >= target:
                    _sync(model_runner)
                    k, v, bs = _gather(model_runner, block_ids, kv_len, layers)
                    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"key": k, "value": v, "layers": layers, "block_ids": block_ids,
                         "block_size": bs, "kv_len": kv_len},
                        Path(snapshot_dir) / f"kv_len{kv_len:06d}.pt",
                    )
                    state["next_snapshot"][req_id] = (kv_len // snapshot_every + 1) * snapshot_every
        return out

    model_runner.execute_model = execute_model
    first = model_runner.kv_caches[0]
    return {"num_layers": num_layers, "kv_cache_shape": list(first.shape), "dtype": str(first.dtype)}


def _dump_final(worker_wrapper, out_dir):
    model_runner = _model_runner(worker_wrapper)
    state = getattr(model_runner, TRACKER_ATTR)
    if len(state["last"]) != 1:
        raise RuntimeError(f"expected exactly one tracked request, got {list(state['last'])}")
    (req_id, last), = state["last"].items()
    _sync(model_runner)
    k, v, block_size = _gather(model_runner, last["block_ids"], last["kv_len"], state["layers"])
    torch.save(
        {"key": k, "value": v, "layers": state["layers"], "block_ids": last["block_ids"],
         "block_size": block_size, "kv_len": last["kv_len"]},
        Path(out_dir) / "kv.pt",
    )
    with (Path(out_dir) / "trace.jsonl").open("w") as f:
        prev = None
        for step, rec in enumerate(state["steps"]):
            row = {"step": step, "kv_len": rec["kv_len"], "num_blocks": len(rec["block_ids"])}
            if rec["block_ids"] != prev:
                row["block_ids"] = rec["block_ids"]
                prev = rec["block_ids"]
            f.write(json.dumps(row) + "\n")
    return {
        "req_id": req_id,
        "kv_len": last["kv_len"],
        "num_blocks": len(last["block_ids"]),
        "block_size": block_size,
        "key_shape": list(k.shape),
    }


# ---------------------------------------------------------------- driver side


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    settings["dataset"] = "quick"
    out_dir = Path(args.output_dir) if args.output_dir else (
        DUMP_ROOT / args.model / args.method / f"problem{args.problem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    # Run the engine in this process so the worker can be reached directly;
    # the hooks are also pickled by value in case multiprocessing is forced.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    configure_logging(out_dir)
    configure_plugin_env(settings, args)

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], use_fast=True)
    rows = load_problems(settings)
    if not 0 <= args.problem < len(rows):
        raise SystemExit(f"--problem must be in [0, {len(rows) - 1}] ({PROBLEMS})")
    (request,) = build_requests([rows[args.problem]], tokenizer, settings)

    llm = LLM(**engine_kwargs(settings, args))
    (cache_info,) = llm.collective_rpc(
        _install_tracker, args=(layers, args.snapshot_every, str(out_dir / "snapshots"))
    )
    print(f"[kv-dump] kv cache per layer: {cache_info}")

    (output,) = llm.generate(
        [{"prompt_token_ids": request["prompt_token_ids"]}],
        [sampling_params(settings, request)],
        use_tqdm=False,
    )
    completion = output.outputs[0]
    (dump_info,) = llm.collective_rpc(_dump_final, args=(str(out_dir),))

    token_ids = list(request["prompt_token_ids"]) + list(completion.token_ids)
    kv_len = dump_info["kv_len"]
    # The last sampled token is never fed back, so fullkv holds one fewer.
    expected = len(token_ids) - 1
    if args.method == "fullkv" and kv_len != expected:
        print(f"[kv-dump][warn] kv_len={kv_len} but prompt+output-1={expected}")
    kv_token_ids = token_ids[:kv_len] if args.method == "fullkv" else None
    (out_dir / "tokens.json").write_text(json.dumps({
        "num_prompt_tokens": len(request["prompt_token_ids"]),
        "token_ids": token_ids,
        "pieces": tokenizer.convert_ids_to_tokens(token_ids),
        "kv_token_ids": kv_token_ids,
    }, ensure_ascii=False))
    meta = {
        **{k: settings[k] for k in ("model", "model_path", "method", "budget", "pruning_mode",
                                    "max_length", "block_size", "seed")},
        "problem": args.problem,
        "question": rows[args.problem]["question"],
        "answer": rows[args.problem]["answer"],
        "output_text": completion.text,
        "finish_reason": completion.finish_reason,
        "num_prompt_tokens": len(request["prompt_token_ids"]),
        "num_output_tokens": len(completion.token_ids),
        **cache_info,
        **dump_info,
        "layers_dumped": layers if layers is not None else "all",
        "kv_tensor": "key/value [layers, tokens, kv_heads, head_dim], logical token order, post-RoPE keys",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[kv-dump] {json.dumps(dump_info)}")
    print(f"[kv-dump] wrote {out_dir}")


if __name__ == "__main__":
    main()
