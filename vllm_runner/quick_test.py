#!/usr/bin/env python3
"""Smoke test: 8 short math problems through vLLM, with or without TriAttention.

Finishes in a minute or two on one GPU. The 4096-token limit and 512-token
budget make TriAttention compress during generation; the script then prints
each answer and the compression events the plugin logged. Prompt, sampling,
seeds and engine settings are the ones ``run_benchmark.py`` uses.

Example:
    python vllm_runner/quick_test.py --model Qwen3-8B --method triattention
    python vllm_runner/quick_test.py --model Qwen3-8B --method fullkv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_benchmark import (
    HF_DEFAULT_TOP_K,
    MODEL_SPECS,
    MODELS_DIR,
    OUTPUT_ROOT,
    build_requests,
    configure_logging,
    configure_plugin_env,
    engine_kwargs,
    load_problems,
    load_runner_defaults,
    make_record,
    sampling_params,
    stats_path_for,
    summarize_compression,
)

PROBLEMS = Path(__file__).resolve().parent / "data" / "quick.jsonl"
# The calibration stats are budget independent; use the aime24 set.
STATS_DATASET = "aime24"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--method", default="triattention", choices=["fullkv", "triattention"])
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--max-length", type=int, default=4096, help="Prompt + generation tokens.")
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
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    return p.parse_args()


def build_settings(args: argparse.Namespace) -> dict:
    runner = load_runner_defaults()["runner_args"]
    model_path = MODELS_DIR / args.model
    if not (model_path / "config.json").exists():
        raise SystemExit(
            f"model not found at {model_path}. "
            f"Run: python scripts/prepare_assets.py --skip-datasets --models {args.model}"
        )
    gen_cfg_path = model_path / "generation_config.json"
    gen_cfg = json.loads(gen_cfg_path.read_text()) if gen_cfg_path.exists() else {}
    tri = args.method == "triattention"
    return {
        "model": args.model,
        "model_path": str(model_path),
        "dataset": "quick",
        "dataset_path": str(PROBLEMS),
        "method": args.method,
        "budget": args.budget if tri else None,
        "stats_path": str(stats_path_for(STATS_DATASET, args.model, args.budget, announce=False))
        if tri
        else None,
        "pruning_mode": args.pruning_mode if tri else None,
        "divide_length": args.divide_length,
        "window_size": args.window_size,
        "protect_prefill": args.protect_prefill,
        "num_samples": 1,
        "max_examples": None,
        "max_length": args.max_length,
        "temperature": float(runner["temperature"]),
        "top_p": float(runner["top_p"]),
        "top_k": int(gen_cfg.get("top_k", HF_DEFAULT_TOP_K)),
        "seed": int(runner["seed"]),
        "use_chat_template": bool(runner.get("use_chat_template", False)),
        "chat_system_prompt": runner.get("chat_system_prompt", ""),
        "block_size": args.block_size,
    }


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, honouring nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    norm = pred.replace(",", "").replace("$", "").strip()
    try:
        return float(norm) == float(gold)
    except ValueError:
        return norm == gold


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    run_dir = OUTPUT_ROOT / "quick" / args.model / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(settings, indent=2))

    log_path = configure_logging(run_dir)
    configure_plugin_env(settings, args)

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], use_fast=True)
    requests = build_requests(load_problems(settings), tokenizer, settings)
    llm = LLM(**engine_kwargs(settings, args))

    t0 = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": r["prompt_token_ids"]} for r in requests],
        [sampling_params(settings, r) for r in requests],
        use_tqdm=True,
    )
    seconds = time.perf_counter() - t0

    records = []
    for req, out in zip(requests, outputs):
        record = make_record(req, out.outputs[0], seconds)
        record["pred"] = last_boxed(record["output"])
        record["correct"] = is_correct(record["pred"], str(record["answer"]))
        records.append(record)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_tokens = sum(r["output_tokens"] for r in records)
    summary = {
        "model": args.model,
        "method": args.method,
        "budget": settings["budget"],
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "output_tokens": out_tokens,
        "seconds": round(seconds, 1),
        "output_tokens_per_second": round(out_tokens / seconds, 1),
        **summarize_compression(log_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'#':>2}  {'gold':>6}  {'pred':>10}  ok  {'tokens':>6}  finish")
    for r in records:
        pred = (r["pred"] or "-")[:10]
        print(
            f"{r['sample_idx']:>2}  {r['answer']:>6}  {pred:>10}  "
            f"{'O' if r['correct'] else 'X':>2}  {r['output_tokens']:>6}  {r['finish_reason']}"
        )
    print(f"\n{json.dumps(summary, indent=2)}")
    if args.method == "triattention" and summary["compression_events"] == 0:
        print(
            "[warn] no compression events: outputs stayed under budget + divide_length, "
            "or the plugin is not active (look for '[TriAttention] Runtime (V2) plugin activated')."
        )
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
