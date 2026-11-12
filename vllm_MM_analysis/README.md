# vllm_MM_analysis

KV-cache redundancy of video / image (multimodal) models on vLLM 0.18.1
(`kv-prof-vllm` env). Text CoT analysis lives in `../vllm_CoT_analysis/`.
Run every command from this folder. Outputs go to `results/<model>/...`.

Models: `LLaVA-OneVision-7B` (1D RoPE), `Qwen2.5-VL-7B` (M-RoPE), `InternVL3-8B` (1D RoPE).
The TriAttention plugin is disabled here (`VLLM_PLUGINS=""`).

## 0. Download (add `--ca-bundle <pem>` behind the company proxy)

```bash
python prepare.py --skip-datasets                       # the 3 models
huggingface-cli login                                   # MVBench is gated: accept its terms on HF first
python prepare.py --skip-models --datasets mvbench      # json + video archives, extracted in place
# Video-MME is very large: fetch the questions and some video chunks
python prepare.py --skip-models --datasets videomme --patterns "*.parquet" "videos_chunked_01.zip"
```

## 1. Does the model run?

```bash
python check_vlm.py --model LLaVA-OneVision-7B          # synthetic video + image
```

## 2. Benchmark accuracy (full KV)

```bash
python run_benchmark.py --model LLaVA-OneVision-7B --limit-per-task 10   # 200 items, quick
python run_benchmark.py --model LLaVA-OneVision-7B                       # all 4000
python run_benchmark.py --model LLaVA-OneVision-7B --num-frames 32        # frame-count sweep
python run_benchmark.py --model LLaVA-OneVision-7B --dataset videomme --tasks short
```

16 frames per video by default (`--num-frames`). Prints overall and per-task
accuracy; writes `results/<model>/mvbench/benchmark/f16/{generations.jsonl,summary.json}`.
Re-running resumes. Tasks whose videos are not found are reported and skipped.
Video-MME items are grouped by video length (short / medium / long) instead of task.

## 3. KV dump and block-merge opportunity

```bash
bash run_all.sh                                    # item 0 of mvbench, 3 models, 16 frames
ITEMS=0-9 FRAMES="16 32 64" bash run_all.sh        # more items and a frame-count sweep
ITEMS=0-4 DATASETS=videomme TASKS=short bash run_all.sh
bash opportunity.sh --min-match 0.5                # all dumps; filters: --model --dataset --frames --item
```

`kv_dump.py` answers one item and dumps its KV in the same format as
`../vllm_CoT_analysis/kv_dump.py`, plus the frame group of every video token
(`tokens.json`: `vision_group`, `vision_spatial`). `opportunity.sh` runs
`../vllm_CoT_analysis/block_merge.py` unchanged on each dump, then
`mm_breakdown.py` splits the holes into video and text tokens
(`opportunity/modality.json`, `results/opportunity_summary.csv`).

Note: with `--scope block` a video token is compared only with the 15 other
slots of its block, i.e. spatially adjacent patches of the same frame. Redundancy
between frames (same patch, next frame) is outside that scope.
