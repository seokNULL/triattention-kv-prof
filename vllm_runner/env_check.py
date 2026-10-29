#!/usr/bin/env python3
"""Check that this env can run vllm_runner/run_benchmark.py.

Reports package versions, whether the TriAttention vLLM plugin is
registered, and which models, datasets and calibration stats are present.
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cli import DATASETS, MODEL_SPECS, MODELS_DIR, REPO_ROOT as _ROOT, stats_path_for  # noqa: E402

# The plugin patches GPUModelRunner._prepare_inputs(scheduler_output,
# num_scheduled_tokens) (vLLM >= 0.12) and calls
# BlockTable.compute_slot_mapping(req_indices, positions) and
# commit_slot_mapping, which vLLM 0.19 replaced.
MIN_VLLM = (0, 12, 0)
MAX_VLLM_EXCLUSIVE = (0, 19, 0)


def version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def parse(v: str) -> tuple[int, ...]:
    parts = []
    for piece in v.split("+")[0].split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def main() -> int:
    ok = True
    print("== packages")
    for pkg in ("vllm", "torch", "transformers", "triton", "triattention"):
        print(f"  {pkg:<13} {version(pkg) or 'NOT INSTALLED'}")
    vllm_v = version("vllm")
    if vllm_v is None or not MIN_VLLM <= parse(vllm_v) < MAX_VLLM_EXCLUSIVE:
        print("  [fail] vllm >=0.12,<0.19 is required (tested: 0.18.1)")
        ok = False

    print("== TriAttention vLLM plugin")
    plugins = [ep for ep in md.entry_points(group="vllm.general_plugins") if ep.name == "triattention"]
    if plugins:
        print(f"  registered: {plugins[0].value}")
    else:
        print("  [fail] not registered; run `pip install -e . --no-deps` in the repo root")
        ok = False

    print("== assets (model / dataset / stats)")
    for model in MODEL_SPECS:
        model_ok = (MODELS_DIR / model / "config.json").exists()
        for dataset in DATASETS:
            data_ok = (_ROOT / "data" / f"{dataset}.jsonl").exists()
            stats_ok = stats_path_for(dataset, model, 2048, announce=False).exists()
            flags = " ".join(
                f"{name}={'ok' if good else 'MISSING'}"
                for name, good in (("model", model_ok), ("data", data_ok), ("stats", stats_ok))
            )
            print(f"  {model:<30} {dataset:<8} {flags}")
    print("== result:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
