#!/usr/bin/env python3
"""Video benchmark accuracy (full KV) of a vision-language model on vLLM.

Greedy decoding, one video per prompt, multiple-choice answers scored by the
option letter. Resumable: finished items in generations.jsonl are skipped.

Example (from vllm_MM_analysis/):
    python run_benchmark.py --model LLaVA-OneVision-7B --limit-per-task 10   # 200 items
    python run_benchmark.py --model Qwen2.5-VL-7B                            # all 4000
    python run_benchmark.py --model InternVL3-8B --tasks "Action Count" "Moving Count"
    python run_benchmark.py --model LLaVA-OneVision-7B --dataset videomme --tasks short

Output: results/<model>/<dataset>/benchmark/f<frames>[_<tag>]/
    generations.jsonl   one line per item (answer, prediction, prompt/vision token counts)
    summary.json        accuracy overall and per task (Video-MME: per video length)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mvbench
import videomme
from models import (OUTPUT_ROOT, VLM_SPECS, build_prompt, engine_extra, model_path,
                    stop_token_ids, vision_token_id)
from video_io import load_video

DATASETS = {"mvbench": mvbench, "videomme": videomme}
LETTERS = "ABCDEFGH"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, choices=list(VLM_SPECS))
    p.add_argument("--dataset", default="mvbench", choices=list(DATASETS))
    p.add_argument("--tasks", nargs="+",
                   help="MVBench task names, or Video-MME lengths (short medium long). Default: all.")
    p.add_argument("--limit-per-task", type=int, help="First N items of each task.")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16, help="Videos decoded per generate call.")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Context limit (default: the model's maximum).")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Pixels per frame for Qwen2.5-VL (default 448*448); others resize to a fixed size.")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tag", help="Suffix for the run folder.")
    return p.parse_args()


def run_dir_for(args: argparse.Namespace) -> Path:
    name = f"f{args.num_frames}" + (f"_{args.tag}" if args.tag else "")
    return OUTPUT_ROOT / args.model / args.dataset / "benchmark" / name


def extract_letter(text: str, num_options: int) -> str | None:
    """First option letter in the answer, e.g. 'B', '(B)', 'B. text', 'Answer: B'."""
    letters = LETTERS[:num_options]
    text = text.strip()
    for pattern in (rf"^\(?([{letters}])(?![A-Za-z])", rf"\(([{letters}])\)",
                    rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])"):
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def summarize(records: list[dict], missing: dict, args: argparse.Namespace) -> dict:
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r["correct"])
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "model": args.model, "dataset": args.dataset, "num_frames": args.num_frames,
        "items": len(records), "accuracy": mean([r["correct"] for r in records]),
        "task_mean_accuracy": mean([mean(v) for v in by_task.values()]),
        "unparsed": sum(r["prediction"] is None for r in records),
        "mean_prompt_tokens": mean([r["prompt_tokens"] for r in records]),
        "mean_vision_tokens": mean([r["vision_tokens"] for r in records]),
        "per_task": {t: {"items": len(v), "accuracy": mean(v)} for t, v in by_task.items()},
        "missing_videos": {t: n for t, n in missing.items() if n},
    }


def main() -> None:
    args = parse_args()
    os.environ["VLLM_PLUGINS"] = ""  # keep the text-only TriAttention plugin out
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    items, missing = DATASETS[args.dataset].load(args.tasks, args.limit_per_task)
    for task, n in missing.items():
        if n:
            print(f"[warn] {task}: {n} videos not found, skipped")
    run_dir = run_dir_for(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_path = run_dir / "generations.jsonl"
    done = load_done(gen_path)
    # Same-video questions (Video-MME has 3 per video) land in one batch and are decoded once.
    todo = sorted((it for it in items if it["id"] not in done), key=lambda it: it["video"])
    print(f"{len(items)} items, {len(done)} already done, {len(todo)} to run -> {run_dir}")

    if todo:
        from vllm import LLM, SamplingParams

        llm = LLM(model=str(model_path(args.model)), limit_mm_per_prompt={"video": 1}, seed=0,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enforce_eager=args.enforce_eager,
                  **engine_extra(args.model, args.max_model_len, args.max_pixels))
        config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
        print(f"max_model_len {getattr(config, 'max_model_len', '?')}")
        tokenizer = llm.get_tokenizer()
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                stop_token_ids=stop_token_ids(args.model, tokenizer))
        vis_id = vision_token_id(args.model, "video", tokenizer)
        decode = lambda key: load_video(key[0], args.num_frames, key[1])  # noqa: E731

        t0 = time.time()
        too_long: dict[str, str] = {}
        with ThreadPoolExecutor(8) as pool, gen_path.open("a") as out:
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start:start + args.batch_size]
                keys = list(dict.fromkeys((it["video"], it["bound"]) for it in batch))
                decoded = dict(zip(keys, pool.map(_safe(decode), keys)))
                videos = [decoded[(it["video"], it["bound"])] for it in batch]
                ok = [(it, v) for it, v in zip(batch, videos) if not isinstance(v, Exception)]
                for it, v in zip(batch, videos):
                    if isinstance(v, Exception):
                        print(f"[warn] {it['id']}: {v}")
                prompts = [{"prompt": build_prompt(args.model, "video", it["question"], tokenizer),
                            "multi_modal_data": {"video": v}} for it, v in ok]
                outputs = generate_each_on_error(llm, prompts, params)
                for (it, v), o in zip(ok, outputs):
                    if isinstance(o, Exception):
                        too_long[it["id"]] = str(o)
                        print(f"[warn] {it['id']} skipped: {o}")
                        continue
                    text = o.outputs[0].text
                    pred = extract_letter(text, len(it["candidates"]))
                    rec = {"id": it["id"], "task": it["task"], "answer": it["answer"],
                           "prediction": pred, "correct": pred == it["answer"], "output": text,
                           "frames": int(v.shape[0]), "prompt_tokens": len(o.prompt_token_ids),
                           "vision_tokens": sum(t == vis_id for t in o.prompt_token_ids)}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[it["id"]] = rec
                out.flush()
                n = min(start + args.batch_size, len(todo))
                acc = sum(r["correct"] for r in done.values()) / max(len(done), 1)
                print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s, running accuracy {acc:.3f}")

    records = [done[it["id"]] for it in items if it["id"] in done]
    summary = summarize(records, missing, args)
    if todo and too_long:
        summary["skipped_errors"] = too_long
        print(f"[warn] {len(too_long)} items skipped (prompt too long for max_model_len?); "
              "see skipped_errors in summary.json. Lower --num-frames / --max-pixels or raise "
              "--max-model-len, then re-run to fill them in.")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n{args.model} {args.dataset}: accuracy {summary['accuracy']:.3f} "
          f"({summary['items']} items, task mean {summary['task_mean_accuracy']:.3f}, "
          f"unparsed {summary['unparsed']}, vision tokens/item {summary['mean_vision_tokens']:.0f})")
    for task, s in summary["per_task"].items():
        print(f"  {task:<26} {s['accuracy']:.3f}  ({s['items']})")
    print(f"wrote {run_dir / 'summary.json'}")


def generate_each_on_error(llm, prompts, params) -> list:
    """llm.generate on the batch; if it is rejected, retry one by one and return the errors in place."""
    if not prompts:
        return []
    try:
        return llm.generate(prompts, params, use_tqdm=False)
    except ValueError as e:
        if len(prompts) == 1:
            return [e]
        return [o for p in prompts for o in generate_each_on_error(llm, [p], params)]


def _safe(fn):
    def wrapped(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001 - report and skip unreadable videos
            return e
    return wrapped


if __name__ == "__main__":
    main()
