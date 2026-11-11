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
```

## 1. Does the model run?

```bash
python check_vlm.py --model LLaVA-OneVision-7B          # synthetic video + image
```

## 2. Benchmark accuracy (full KV)

```bash
python run_benchmark.py --model LLaVA-OneVision-7B --limit-per-task 10   # 200 items, quick
python run_benchmark.py --model LLaVA-OneVision-7B                       # all 4000
```

16 frames per video by default (`--num-frames`). Prints overall and per-task
accuracy; writes `results/<model>/mvbench/benchmark/f16/{generations.jsonl,summary.json}`.
Re-running resumes. Tasks whose videos are not found are reported and skipped.
