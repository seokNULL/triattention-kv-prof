# vllm_MM_analysis

KV-cache redundancy of video / image (multimodal) models on vLLM 0.18.1
(`kv-prof-vllm` env). Text CoT analysis lives in `../vllm_CoT_analysis/`.
Run every command from this folder. Outputs go to `results/<model>/...`.

Models: `LLaVA-OneVision-7B` (1D RoPE), `Qwen2.5-VL-7B` (M-RoPE).
The TriAttention plugin is disabled here (`VLLM_PLUGINS=""`).

## Step 1: does the model run?

```bash
python prepare_models.py --models LLaVA-OneVision-7B     # add --ca-bundle <pem> behind the proxy
python check_vlm.py --model LLaVA-OneVision-7B           # synthetic video + image
python check_vlm.py --model LLaVA-OneVision-7B --video clip.mp4 --image photo.jpg
```

Prints each answer, the prompt length and the positions of the vision tokens;
saves `results/<model>/check/check_vlm.json`.
