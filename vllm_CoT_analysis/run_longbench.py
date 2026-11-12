#!/usr/bin/env python3
"""LongBench v2 accuracy of a CoT model on vLLM (full KV, or TriAttention).

One draw per question with the same sampling as run_benchmark.py, the
model's chat template (thinking enabled), and the official 0-shot prompt.
Contexts longer than --max-length minus --max-new-tokens are cut in the
middle. Resumable: finished questions in generations.jsonl are skipped.

Example (from vllm_CoT_analysis/):
    python run_longbench.py --model Qwen3-8B --dataset longbench_v2_short --max-examples 10
    python run_longbench.py --model DeepSeek-R1-Distill-Llama-8B --dataset longbench_v2_short

Output: results/<model>/<dataset>/benchmark/<method>/<tag>/
    generations/generations.jsonl   per question: prediction, output, token counts (no context)
    summary.json                    accuracy overall, per domain and per difficulty
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import longbench
from run_benchmark import (
    HF_DEFAULT_TOP_K,
    MODEL_SPECS,
    MODELS_DIR,
    OUTPUT_ROOT,
    configure_logging,
    configure_plugin_env,
    engine_kwargs,
    load_runner_defaults,
    sampling_params,
    stats_path_for,
    summarize_compression,
)
from cli import load_default_budget  # noqa: E402  (scripts/ is on sys.path via run_benchmark)

# TriAttention calibration stats are budget independent; reuse the aime24 set.
STATS_DATASET = "aime24"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--dataset", default="longbench_v2_short", choices=longbench.DATASETS)
    p.add_argument("--method", default="fullkv", choices=["fullkv", "triattention"])
    p.add_argument("--max-examples", type=int, help="Only the first N questions.")
    p.add_argument("--max-length", type=int, default=longbench.MAX_LENGTH,
                   help="Prompt + generation tokens (engine max_model_len).")
    p.add_argument("--max-new-tokens", type=int, default=longbench.MAX_NEW_TOKENS,
                   help="Tokens reserved for generation; the context is cut to fit the rest.")
    p.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budget", type=int, help="TriAttention KV budget (default per model).")
    p.add_argument("--pruning-mode", default="per_head",
                   choices=["per_head", "per_layer_per_head", "per_layer"])
    p.add_argument("--divide-length", type=int, default=128)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--protect-prefill", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-decisions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=16, help="Questions per generate() call.")
    p.add_argument("--run-tag", default=None)
    args = p.parse_args()
    args.tensor_parallel_size = 1
    args.max_num_seqs = None
    args.max_num_batched_tokens = None
    args.prefix_caching = False
    return args


def build_settings(args: argparse.Namespace) -> dict:
    runner = load_runner_defaults()["runner_args"]
    model_path = MODELS_DIR / args.model
    if not (model_path / "config.json").exists():
        raise SystemExit(f"model not found at {model_path}. "
                         f"Run: python scripts/prepare_assets.py --skip-datasets --models {args.model}")
    gen_cfg_path = model_path / "generation_config.json"
    gen_cfg = json.loads(gen_cfg_path.read_text()) if gen_cfg_path.exists() else {}
    tri = args.method == "triattention"
    budget = (args.budget or load_default_budget(args.model)) if tri else None
    return {
        "model": args.model, "model_path": str(model_path), "dataset": args.dataset,
        "method": args.method, "budget": budget,
        "stats_path": str(stats_path_for(STATS_DATASET, args.model, budget, announce=False)) if tri else None,
        "pruning_mode": args.pruning_mode if tri else None,
        "divide_length": args.divide_length, "window_size": args.window_size,
        "protect_prefill": args.protect_prefill,
        "max_length": args.max_length, "max_new_tokens": args.max_new_tokens,
        "chat_template": args.chat_template,
        "temperature": float(runner["temperature"]), "top_p": float(runner["top_p"]),
        "top_k": int(gen_cfg.get("top_k", HF_DEFAULT_TOP_K)), "seed": int(runner["seed"]),
        "block_size": args.block_size,
    }


def build_request(row: dict, tokenizer, settings: dict) -> dict:
    ids, prompt, removed = longbench.build_prompt_ids(
        row, tokenizer, settings["max_length"] - settings["max_new_tokens"], settings["chat_template"])
    return {"row": row, "prompt": prompt, "prompt_token_ids": ids, "context_tokens_removed": removed,
            "max_tokens": settings["max_length"] - len(ids), "seed": settings["seed"] + row["index"]}


def make_record(req: dict, completion) -> dict:
    row = req["row"]
    prediction = longbench.extract_answer(completion.text)
    return {
        **{k: row[k] for k in ("index", "_id", "domain", "sub_domain", "difficulty", "length",
                               "question", "answer")},
        "prediction": prediction, "correct": prediction == row["answer"],
        "output": completion.text, "finish_reason": completion.finish_reason,
        "prefill_tokens": len(req["prompt_token_ids"]), "output_tokens": len(completion.token_ids),
        "context_tokens_removed": req["context_tokens_removed"],
    }


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    tag = "full" if args.method == "fullkv" else f"budget_{settings['budget']}"
    tag += f"_{args.run_tag}" if args.run_tag else ""
    run_dir = OUTPUT_ROOT / args.model / args.dataset / "benchmark" / args.method / tag
    gen_path = run_dir / "generations" / "generations.jsonl"
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(settings, indent=2))
    print(f"[longbench] output: {run_dir}")
    log_path = configure_logging(run_dir)
    configure_plugin_env(settings, args)

    rows = longbench.load_rows(args.dataset, args.max_examples)
    done = {}
    if gen_path.exists():
        for line in gen_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["index"]] = rec
    pending = [r for r in rows if r["index"] not in done]
    print(f"[longbench] {len(rows)} questions, {len(done)} done, {len(pending)} to run")

    if pending:
        from transformers import AutoTokenizer
        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], use_fast=True)
        llm = LLM(**engine_kwargs(settings, args))
        with gen_path.open("a", encoding="utf-8") as out:
            for start in range(0, len(pending), args.chunk_size):
                chunk = [build_request(r, tokenizer, settings) for r in pending[start:start + args.chunk_size]]
                t0 = time.perf_counter()
                outputs = llm.generate([{"prompt_token_ids": r["prompt_token_ids"]} for r in chunk],
                                       [sampling_params(settings, r) for r in chunk], use_tqdm=True)
                for req, o in zip(chunk, outputs):
                    rec = make_record(req, o.outputs[0])
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[rec["index"]] = rec
                out.flush()
                acc = sum(r["correct"] for r in done.values()) / len(done)
                print(f"[longbench] {min(start + args.chunk_size, len(pending))}/{len(pending)} "
                      f"in {time.perf_counter() - t0:.0f}s, running accuracy {acc:.3f}")

    records = [done[r["index"]] for r in rows if r["index"] in done]
    summary = {**longbench.summarize(records), **summarize_compression(log_path),
               "model": args.model, "dataset": args.dataset, "method": args.method}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n{args.model} {args.dataset}: accuracy {summary['accuracy']:.3f} ({summary['items']} questions, "
          f"no answer {summary['no_answer']}, hit length limit {summary['hit_length_limit']}, "
          f"context cut {summary['truncated_context']})")
    for domain, s in summary["per_domain"].items():
        print(f"  {domain:<40} {s['accuracy']:.3f}  ({s['items']})")
    print(f"wrote {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
