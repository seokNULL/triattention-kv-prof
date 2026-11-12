#!/usr/bin/env python3
"""Answer one video benchmark item with vLLM (full KV) and dump its paged KV cache.

One request per engine, so its blocks are not shared. After generation the
request's K/V is copied out of vLLM's paged cache into
results/<model>/<dataset>/f<frames>/<item id>/:

    kv.pt        key/value [layers, tokens, kv_heads, head_dim] (bf16, post-RoPE K),
                 logical token order; token t is in block block_ids[t // block_size]
    tokens.json  token ids and pieces per KV position, plus where the video tokens are:
                 vision_group (frame group per position, -1 text, -2 extra vision token
                 such as a trailing newline) and vision_spatial (index within the group)
    meta.json    model shape, item, answer, prediction, video token layout
    trace.jsonl  per step: KV length and block table

The same format as vllm_CoT_analysis/kv_dump.py, so its block_merge.py reads it.

Example (from vllm_MM_analysis/):
    python kv_dump.py --model LLaVA-OneVision-7B --dataset mvbench --item 0
    python kv_dump.py --model Qwen2.5-VL-7B --dataset mvbench --id action_count/3 --num-frames 32
    python kv_dump.py --model InternVL3-8B --dataset videomme --tasks short --item 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cloudpickle

import kv_tracker
from check_vlm import token_runs
from models import (OUTPUT_ROOT, VLM_SPECS, build_prompt, engine_extra, model_path,
                    stop_token_ids, vision_token_id)
from run_benchmark import DATASETS, extract_letter
from video_io import load_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=list(VLM_SPECS))
    p.add_argument("--dataset", default="mvbench", choices=list(DATASETS))
    p.add_argument("--tasks", nargs="+", help="Restrict items as in run_benchmark.py.")
    p.add_argument("--item", type=int, default=0, help="Index into the (filtered) item list.")
    p.add_argument("--id", help="Item id instead of --item, e.g. action_count/3 (see generations.jsonl).")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--layers", default="all", help="'all' or a comma list such as 0,9,18,27.")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Context limit (default: the model's maximum).")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Pixels per frame for Qwen2.5-VL (default 448*448); others resize to a fixed size.")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--output-dir", help="Default: results/<model>/<dataset>/f<frames>/<item id>")
    p.add_argument("--skip-existing", action="store_true", help="Do nothing if kv.pt already exists.")
    return p.parse_args()


def safe_name(item_id: str) -> str:
    return item_id.replace("/", "_")


def vision_layout(token_ids: list[int], vis_id: int, num_frames: int, temporal_patch: int):
    """Frame group and in-group index of every video token (-1 for text tokens).

    One run of video tokens per frame (InternVL: 'Frame i: <img>...</img>') gives
    the groups directly; a single run is split evenly into ceil(frames / temporal_patch)
    groups, and tokens left over past the last full group get group -2.
    """
    runs = token_runs(token_ids, vis_id)
    group = [-1] * len(token_ids)
    spatial = [-1] * len(token_ids)
    n_vision = sum(e - s for s, e in runs)
    if not runs:
        return group, spatial, {"runs": runs, "vision_tokens": 0, "groups": 0, "tokens_per_group": 0}
    if len(runs) > 1:
        for g, (s, e) in enumerate(runs):
            for t in range(s, e):
                group[t], spatial[t] = g, t - s
        sizes = {e - s for s, e in runs}
        per_group = sizes.pop() if len(sizes) == 1 else None
        n_groups = len(runs)
    else:
        n_groups = math.ceil(num_frames / temporal_patch)
        per_group = n_vision // n_groups
        s, e = runs[0]
        for t in range(s, e):
            off = t - s
            g = off // per_group if per_group else -2
            group[t], spatial[t] = (g, off % per_group) if 0 <= g < n_groups else (-2, -1)
    return group, spatial, {"runs": runs, "vision_tokens": n_vision, "groups": n_groups,
                            "tokens_per_group": per_group}


def main() -> None:
    args = parse_args()
    os.environ["VLLM_PLUGINS"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Run the engine in this process so collective_rpc reaches the worker directly.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    cloudpickle.register_pickle_by_value(kv_tracker)

    items, _ = DATASETS[args.dataset].load(args.tasks, None)
    if args.id:
        matches = [it for it in items if it["id"] == args.id]
        if not matches:
            raise SystemExit(f"item id {args.id!r} not found (or its video is missing)")
        item = matches[0]
    else:
        if not 0 <= args.item < len(items):
            raise SystemExit(f"--item must be in [0, {len(items) - 1}]")
        item = items[args.item]
    out_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUT_ROOT / args.model / args.dataset / f"f{args.num_frames}" / safe_name(item["id"]))
    if args.skip_existing and (out_dir / "kv.pt").exists():
        print(f"[kv-dump] {out_dir} already has kv.pt, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    video = load_video(item["video"], args.num_frames, item["bound"])
    print(f"[kv-dump] {item['id']} ({item['task']}): {item['video']} -> {video.shape}")

    from vllm import LLM, SamplingParams

    llm = LLM(model=str(model_path(args.model)), limit_mm_per_prompt={"video": 1}, seed=0,
              enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_memory_utilization,
              **engine_extra(args.model, args.max_model_len, args.max_pixels))
    (cache_info,) = llm.collective_rpc(kv_tracker.install_tracker, args=(layers,))
    print(f"[kv-dump] kv cache per layer: {cache_info}")
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            stop_token_ids=stop_token_ids(args.model, tokenizer))
    prompt = build_prompt(args.model, "video", item["question"], tokenizer)
    (output,) = llm.generate([{"prompt": prompt, "multi_modal_data": {"video": video}}], params,
                             use_tqdm=False)
    (dump_info,) = llm.collective_rpc(kv_tracker.dump_final, args=(str(out_dir),))

    completion = output.outputs[0]
    prompt_ids = list(output.prompt_token_ids)
    token_ids = prompt_ids + list(completion.token_ids)
    kv_len = dump_info["kv_len"]
    if kv_len != len(token_ids) - 1:
        print(f"[kv-dump][warn] kv_len={kv_len} but prompt+output-1={len(token_ids) - 1}")
    vis_id = vision_token_id(args.model, "video", tokenizer)
    temporal_patch = VLM_SPECS[args.model].get("temporal_patch", 1)
    group, spatial, layout = vision_layout(token_ids[:kv_len], vis_id, int(video.shape[0]),
                                           temporal_patch)
    if layout["vision_tokens"] < video.shape[0]:
        print(f"[kv-dump][warn] only {layout['vision_tokens']} video tokens in the prompt ids; "
              "frame groups are not available")
    (out_dir / "tokens.json").write_text(json.dumps({
        "num_prompt_tokens": len(prompt_ids),
        "token_ids": token_ids,
        "pieces": tokenizer.convert_ids_to_tokens(token_ids),
        "kv_token_ids": token_ids[:kv_len],
        "vision_group": group,
        "vision_spatial": spatial,
    }, ensure_ascii=False))
    prediction = extract_letter(completion.text, len(item["candidates"]))
    meta = {
        "model": args.model, "model_path": str(model_path(args.model)), "dataset": args.dataset,
        "id": item["id"], "item_index": items.index(item), "tasks_filter": args.tasks,
        "task": item["task"], "video": item["video"], "bound": item["bound"],
        "num_frames": int(video.shape[0]), "frame_size": list(video.shape[1:3]),
        "question": item["question"], "answer": item["answer"], "prediction": prediction,
        "correct": prediction == item["answer"], "output_text": completion.text,
        "finish_reason": completion.finish_reason,
        "num_prompt_tokens": len(prompt_ids), "num_output_tokens": len(completion.token_ids),
        "vision_token": VLM_SPECS[args.model]["vision_token"]["video"], "vision_token_id": vis_id,
        "temporal_patch": temporal_patch, **{f"vision_{k}": v for k, v in layout.items()},
        **cache_info, **dump_info,
        "layers_dumped": layers if layers is not None else "all",
        "kv_tensor": "key/value [layers, tokens, kv_heads, head_dim], logical token order, post-RoPE keys",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[kv-dump] answer {item['answer']} prediction {prediction} ({completion.text.strip()!r})")
    print(f"[kv-dump] {kv_len} KV tokens, {layout['vision_tokens']} video tokens in "
          f"{layout['groups']} groups of {layout['tokens_per_group']}")
    print(f"[kv-dump] wrote {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
