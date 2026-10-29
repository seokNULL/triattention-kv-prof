#!/usr/bin/env python3
"""Run AIME24 / AIME25 / MATH-500 with vLLM, optionally with TriAttention.

Uses the same models, datasets, prompt, sampling and seeds as the HF path
(``scripts/cli.py run-one``) and grades with the same evaluator, but generates
with vLLM's paged KV cache. ``--method triattention`` turns on the runtime
plugin in ``triattention/vllm`` (installed as a ``vllm.general_plugins`` entry
point); ``--method fullkv`` disables it.

Example:
    python vllm_runner/run_vllm.py --model Qwen3-8B --dataset aime24 \\
        --method triattention --budget 2048 --num-samples 1 --max-examples 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from cli import (  # noqa: E402
    DATASETS,
    MODEL_SPECS,
    MODELS_DIR,
    dataset_max_length,
    load_default_budget,
    load_runner_defaults,
    resolve_dataset_path,
    resolve_num_samples,
    stats_path_for,
)

OUTPUT_ROOT = REPO_ROOT / "experiments" / "vllm_outputs"
EVAL_SCRIPT = REPO_ROOT / "triattention" / "evaluation" / "eval_math_multi.py"
# Same per-draw seed formula as scripts/worker.py.
RUN_SEED_STRIDE = 1_000_000
# HF generate() falls back to top_k=50 when the model's generation_config
# does not set one; mirror that so both paths sample identically.
HF_DEFAULT_TOP_K = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--method", required=True, choices=["fullkv", "triattention"])
    p.add_argument("--budget", type=int, help="KV budget (default: defaults.yaml per model).")
    p.add_argument("--num-samples", type=int, help="Draws per problem (default: runner_defaults.yaml).")
    p.add_argument("--max-examples", type=int, help="Only the first N problems.")
    p.add_argument("--max-length", type=int, help="Prompt + generation tokens (default per dataset).")

    tri = p.add_argument_group("TriAttention runtime")
    tri.add_argument(
        "--pruning-mode",
        default="per_head",
        choices=["per_head", "per_layer_per_head", "per_layer"],
    )
    tri.add_argument("--stats-path", help="Calibration stats (default: shipped file).")
    tri.add_argument("--divide-length", type=int, default=128, help="Compress every N tokens.")
    tri.add_argument("--window-size", type=int, default=128, help="Recent tokens always kept.")
    tri.add_argument(
        "--protect-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Never evict prompt tokens (HF path default: allow_prefill_compression=False).",
    )
    tri.add_argument(
        "--log-decisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log each TriAttention compression decision.",
    )

    s = p.add_argument_group("sampling")
    s.add_argument("--temperature", type=float, default=None)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--top-k", type=int, default=None, help="Default: model generation_config, else 50.")
    s.add_argument("--seed", type=int, default=None)

    e = p.add_argument_group("vLLM engine")
    e.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    e.add_argument("--tensor-parallel-size", type=int, default=1)
    e.add_argument("--block-size", type=int, default=16)
    e.add_argument("--max-num-seqs", type=int, default=None)
    e.add_argument("--max-num-batched-tokens", type=int, default=None)
    e.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Requests per generate() call; finished chunks are saved so reruns resume.",
    )

    o = p.add_argument_group("output")
    o.add_argument("--run-tag", default=None, help="Suffix for the output folder.")
    o.add_argument("--output-root", default=str(OUTPUT_ROOT))
    o.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict:
    defaults = load_runner_defaults()
    runner = defaults["runner_args"]
    budget = None
    stats_path = None
    if args.method == "triattention":
        budget = args.budget or load_default_budget(args.model)
        stats_path = Path(args.stats_path) if args.stats_path else stats_path_for(
            args.dataset, args.model, budget
        )
        if not stats_path.exists():
            raise SystemExit(f"stats file not found: {stats_path}")
    model_path = MODELS_DIR / args.model
    if not (model_path / "config.json").exists():
        raise SystemExit(
            f"model not found at {model_path}. "
            f"Run: python scripts/prepare_assets.py --skip-datasets --models {args.model}"
        )
    gen_cfg_path = model_path / "generation_config.json"
    gen_cfg = json.loads(gen_cfg_path.read_text()) if gen_cfg_path.exists() else {}
    return {
        "model": args.model,
        "model_path": str(model_path),
        "dataset": args.dataset,
        "dataset_path": str(resolve_dataset_path(args.dataset)),
        "method": args.method,
        "budget": budget,
        "stats_path": str(stats_path) if stats_path else None,
        "pruning_mode": args.pruning_mode if args.method == "triattention" else None,
        "divide_length": args.divide_length,
        "window_size": args.window_size,
        "protect_prefill": args.protect_prefill,
        "num_samples": args.num_samples or resolve_num_samples(runner, args.dataset),
        "max_examples": args.max_examples,
        "max_length": args.max_length or dataset_max_length(args.dataset, defaults),
        "temperature": args.temperature if args.temperature is not None else float(runner["temperature"]),
        "top_p": args.top_p if args.top_p is not None else float(runner["top_p"]),
        "top_k": args.top_k if args.top_k is not None else int(gen_cfg.get("top_k", HF_DEFAULT_TOP_K)),
        "seed": args.seed if args.seed is not None else int(runner["seed"]),
        "use_chat_template": bool(runner.get("use_chat_template", False)),
        "chat_system_prompt": runner.get("chat_system_prompt", ""),
        "block_size": args.block_size,
    }


def run_dir_for(settings: dict, args: argparse.Namespace) -> Path:
    tag = "full" if settings["method"] == "fullkv" else f"budget_{settings['budget']}"
    if settings["method"] == "triattention" and settings["pruning_mode"] != "per_head":
        tag += f"_{settings['pruning_mode']}"
    if args.run_tag:
        tag += f"_{args.run_tag}"
    return (
        Path(args.output_root)
        / settings["dataset"]
        / settings["model"]
        / f"sample{settings['num_samples']}"
        / settings["method"]
        / tag
    )


def configure_plugin_env(settings: dict, args: argparse.Namespace) -> None:
    """Must run before vLLM is imported: the plugin reads these at load time."""
    if settings["method"] == "fullkv":
        os.environ["ENABLE_TRIATTENTION"] = "0"
        return
    os.environ["ENABLE_TRIATTENTION"] = "1"
    env = {
        "KV_BUDGET": settings["budget"],
        "SPARSE_STATS_PATH": settings["stats_path"],
        "MODEL_PATH": settings["model_path"],
        "PRUNING_MODE": settings["pruning_mode"],
        "DIVIDE_LENGTH": settings["divide_length"],
        "WINDOW_SIZE": settings["window_size"],
        "PROTECT_PREFILL": str(settings["protect_prefill"]).lower(),
        "LOG_DECISIONS": str(args.log_decisions).lower(),
    }
    for key, value in env.items():
        os.environ[f"TRIATTN_RUNTIME_{key}"] = str(value)


def load_problems(settings: dict) -> list[dict]:
    from triattention.common.prompt_utils import extract_question_from_record

    rows = []
    with open(settings["dataset_path"], encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["question"] = extract_question_from_record(row)
            row["index"] = index
            rows.append(row)
            if settings["max_examples"] and len(rows) >= settings["max_examples"]:
                break
    return rows


def build_requests(rows: list[dict], tokenizer, settings: dict) -> list[dict]:
    """One request per (problem, draw), tokenized exactly like scripts/worker.py."""
    from triattention.common.prompt_utils import build_prompt

    requests = []
    for row in rows:
        prompt = build_prompt(
            tokenizer,
            row["question"],
            use_chat_template=settings["use_chat_template"],
            system_prompt=settings["chat_system_prompt"],
        )
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        max_tokens = settings["max_length"] - len(ids)
        if max_tokens <= 0:
            raise SystemExit(f"prompt of problem {row['index']} exceeds max_length")
        for draw in range(settings["num_samples"]):
            requests.append(
                {
                    "row": row,
                    "prompt": prompt,
                    "prompt_token_ids": ids,
                    "draw_idx": draw,
                    "max_tokens": max_tokens,
                    "seed": settings["seed"] + draw * RUN_SEED_STRIDE + row["index"],
                }
            )
    return requests


def load_done(path: Path) -> set[tuple[int, int]]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((rec["sample_idx"], rec["draw_idx"]))
    return done


def make_record(req: dict, completion, seconds: float) -> dict:
    record = dict(req["row"])
    record.update(
        {
            "prompt": req["prompt"],
            "output": completion.text,
            "prefill_tokens": len(req["prompt_token_ids"]),
            "output_tokens": len(completion.token_ids),
            "total_tokens": len(req["prompt_token_ids"]) + len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "chunk_generation_seconds": seconds,
            "sample_idx": req["row"]["index"],
            "draw_idx": req["draw_idx"],
        }
    )
    return record


def generate(settings: dict, args: argparse.Namespace, gen_path: Path) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], use_fast=True)
    requests = build_requests(load_problems(settings), tokenizer, settings)
    done = load_done(gen_path)
    pending = [r for r in requests if (r["row"]["index"], r["draw_idx"]) not in done]
    print(
        f"[vllm-runner] {len(requests)} requests, {len(requests) - len(pending)} already done, "
        f"{len(pending)} to run"
    )
    if not pending:
        return {}

    engine_kwargs = {
        "model": settings["model_path"],
        "dtype": "bfloat16",
        "max_model_len": settings["max_length"],
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": settings["block_size"],
        # Required by the TriAttention runtime; kept for fullkv so both
        # methods run on the same engine configuration.
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "trust_remote_code": True,
        "seed": settings["seed"],
        "disable_log_stats": False,
    }
    if args.max_num_seqs:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**engine_kwargs)

    total_out, total_seconds = 0, 0.0
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    with gen_path.open("a", encoding="utf-8") as fout:
        for start in range(0, len(pending), args.chunk_size):
            chunk = pending[start : start + args.chunk_size]
            params = [
                SamplingParams(
                    n=1,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                    max_tokens=r["max_tokens"],
                    seed=r["seed"],
                )
                for r in chunk
            ]
            prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in chunk]
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=True)
            seconds = time.perf_counter() - t0
            chunk_out = 0
            for req, out in zip(chunk, outputs):
                completion = out.outputs[0]
                chunk_out += len(completion.token_ids)
                fout.write(json.dumps(make_record(req, completion, seconds), ensure_ascii=False) + "\n")
            fout.flush()
            total_out += chunk_out
            total_seconds += seconds
            print(
                f"[vllm-runner] chunk {start // args.chunk_size + 1}: {len(chunk)} requests, "
                f"{chunk_out} output tokens in {seconds:.1f}s ({chunk_out / seconds:.1f} tok/s)"
            )
    return {
        "generated_requests": len(pending),
        "output_tokens": total_out,
        "generation_seconds": round(total_seconds, 2),
        "output_tokens_per_second": round(total_out / total_seconds, 2) if total_seconds else 0.0,
    }


def evaluate(settings: dict, run_dir: Path, gen_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--base_dir",
        str(gen_dir),
        "--dataset",
        settings["dataset"],
        "--exp_name",
        "eval",
        "--output_dir",
        str(run_dir),
        "--num_samples",
        str(settings["num_samples"]),
    ]
    print(f"[vllm-runner] eval: {' '.join(cmd)}")
    env = dict(os.environ, PYTHONPATH=f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    run_dir = run_dir_for(settings, args)
    gen_dir = run_dir / "generations"
    gen_path = gen_dir / "generations.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(settings, indent=2))
    print(f"[vllm-runner] output: {run_dir}")
    print(json.dumps(settings, indent=2))

    configure_plugin_env(settings, args)
    stats = generate(settings, args, gen_path)
    if stats:
        (run_dir / "throughput.json").write_text(json.dumps(stats, indent=2))
        print(f"[vllm-runner] throughput: {stats}")
    if not args.skip_eval:
        evaluate(settings, run_dir, gen_dir)


if __name__ == "__main__":
    main()
