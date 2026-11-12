"""Worker-side hooks that record a request's block table and copy its paged KV out.

Same mechanism as vllm_CoT_analysis/kv_dump.py: they run inside the vLLM
worker via LLM.collective_rpc and receive the worker wrapper first.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

TRACKER_ATTR = "_kv_dump_tracker"


def _model_runner(worker_wrapper):
    worker = getattr(worker_wrapper, "worker", worker_wrapper)
    return worker.model_runner


def _split_kv(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V views of one layer's cache, each [num_blocks, block_size, H, D]."""
    if kv_cache.ndim != 5:
        raise ValueError(f"unexpected kv cache shape {tuple(kv_cache.shape)}")
    if kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    if kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    raise ValueError(f"no K/V axis of size 2 in {tuple(kv_cache.shape)}")


def _sync(model_runner) -> None:
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
        keys.append(k_cache.index_select(0, idx).flatten(0, 1)[:kv_len].to("cpu", copy=True))
        values.append(v_cache.index_select(0, idx).flatten(0, 1)[:kv_len].to("cpu", copy=True))
    return torch.stack(keys), torch.stack(values), block_size


def install_tracker(worker_wrapper, layers):
    model_runner = _model_runner(worker_wrapper)
    num_layers = len(model_runner.kv_caches)
    state = {"layers": list(range(num_layers)) if layers is None else layers,
             "steps": [], "last": {}}
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
            state["steps"].append({"kv_len": kv_len, "block_ids": block_ids})
        return out

    model_runner.execute_model = execute_model
    first = model_runner.kv_caches[0]
    return {"num_layers": num_layers, "kv_cache_shape": list(first.shape), "dtype": str(first.dtype)}


def dump_final(worker_wrapper, out_dir):
    model_runner = _model_runner(worker_wrapper)
    state = getattr(model_runner, TRACKER_ATTR)
    if len(state["last"]) != 1:
        raise RuntimeError(f"expected exactly one tracked request, got {list(state['last'])}")
    (req_id, last), = state["last"].items()
    _sync(model_runner)
    k, v, block_size = _gather(model_runner, last["block_ids"], last["kv_len"], state["layers"])
    torch.save({"key": k, "value": v, "layers": state["layers"], "block_ids": last["block_ids"],
                "block_size": block_size, "kv_len": last["kv_len"]}, Path(out_dir) / "kv.pt")
    with (Path(out_dir) / "trace.jsonl").open("w") as f:
        prev = None
        for step, rec in enumerate(state["steps"]):
            row = {"step": step, "kv_len": rec["kv_len"], "num_blocks": len(rec["block_ids"])}
            if rec["block_ids"] != prev:
                row["block_ids"] = prev = rec["block_ids"]
            f.write(json.dumps(row) + "\n")
    return {"req_id": req_id, "kv_len": last["kv_len"], "num_blocks": len(last["block_ids"]),
            "block_size": block_size, "key_shape": list(k.shape)}
