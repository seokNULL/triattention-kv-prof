#!/usr/bin/env python3
"""Download the models and datasets that ``scripts/cli.py run-one`` expects.

Models go to ``experiments/models/<name>`` and datasets to
``data/<name>.jsonl``. TriAttention calibration stats need no preparation:
``cli.py`` falls back to the files shipped in ``triattention/calibration``.

Examples:
    python scripts/prepare_assets.py                              # everything
    python scripts/prepare_assets.py --models Qwen3-8B --datasets aime24
    python scripts/prepare_assets.py --ca-bundle ~/ca-bundle.pem  # corporate CA
    python scripts/prepare_assets.py --skip-models \\
        --local-dataset-dir dataset    # convert dataset/AIME24, dataset/MATH500, ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cli import DATASETS, MODEL_SPECS, MODELS_DIR, REPO_ROOT

DATA_DIR = REPO_ROOT / "data"

DATASET_SOURCES = {
    "aime24": "HuggingFaceH4/aime_2024",
    "aime25": "MathArena/aime_2025",
    "math500": "HuggingFaceH4/MATH-500",
}
EXPECTED_ROWS = {"aime24": 30, "aime25": 30, "math500": 500}
# Folder names accepted by --local-dataset-dir, compared case-insensitively
# with "-" and "_" removed.
LOCAL_ALIASES = {
    "aime24": {"aime24", "aime2024"},
    "aime25": {"aime25", "aime2025"},
    "math500": {"math500", "math"},
}
MODEL_FILE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]


def set_ca_bundle(path: str) -> None:
    bundle = str(Path(path).expanduser().resolve())
    if not Path(bundle).is_file():
        raise SystemExit(f"CA bundle not found: {bundle}")
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ[key] = bundle


def model_is_complete(path: Path) -> bool:
    return (path / "config.json").exists() and any(path.glob("*.safetensors"))


def download_model(name: str, force: bool) -> None:
    from huggingface_hub import snapshot_download

    target = MODELS_DIR / name
    if model_is_complete(target) and not force:
        print(f"[skip] model {name} already at {target}")
        return
    print(f"[download] model {MODEL_SPECS[name]} -> {target}")
    snapshot_download(
        repo_id=MODEL_SPECS[name],
        local_dir=str(target),
        allow_patterns=MODEL_FILE_PATTERNS,
    )
    if not model_is_complete(target):
        raise SystemExit(f"model download incomplete: {target}")


def find_local_source(root: Path, name: str) -> Path:
    for entry in sorted(root.iterdir()):
        key = entry.stem.lower().replace("-", "").replace("_", "")
        if key in LOCAL_ALIASES[name]:
            return entry
    raise SystemExit(
        f"no folder/file for {name} under {root} "
        f"(accepted names: {sorted(LOCAL_ALIASES[name])}, any case)"
    )


def load_rows(name: str, local_root: Path | None) -> list[dict]:
    from datasets import load_dataset

    if local_root is None:
        print(f"[download] dataset {DATASET_SOURCES[name]}")
        dd = load_dataset(DATASET_SOURCES[name])
    else:
        src = find_local_source(local_root, name)
        print(f"[convert] dataset {src}")
        if src.is_dir():
            dd = load_dataset(str(src))
        else:
            fmt = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}.get(
                src.suffix
            )
            if fmt is None:
                raise SystemExit(f"unsupported dataset file: {src}")
            dd = load_dataset(fmt, data_files=str(src))
    split = next((s for s in ("test", "train") if s in dd), next(iter(dd)))
    return [dict(row) for row in dd[split]]


def prepare_dataset(name: str, local_root: Path | None, force: bool) -> None:
    out = DATA_DIR / f"{name}.jsonl"
    if out.exists() and not force:
        print(f"[skip] dataset {name} already at {out}")
        return
    rows = load_rows(name, local_root)
    for row in rows:
        # Same normalisation as triattention/evaluation/data_loader.py.
        if name.startswith("aime") and "question" not in row and "problem" in row:
            row["question"] = row.pop("problem")
    if not rows or "answer" not in rows[0]:
        fields = list(rows[0]) if rows else []
        raise SystemExit(f"{name}: expected an 'answer' field, got {fields}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    note = "" if len(rows) == EXPECTED_ROWS[name] else (
        f"  [warn] expected {EXPECTED_ROWS[name]} rows"
    )
    print(f"[ok] {out} ({len(rows)} rows){note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument(
        "--local-dataset-dir",
        help="Convert already-downloaded datasets from this folder instead of "
        "fetching them (e.g. dataset/AIME24, dataset/MATH500).",
    )
    parser.add_argument("--ca-bundle", help="PEM bundle for TLS-intercepting proxies.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ca_bundle:
        set_ca_bundle(args.ca_bundle)
    if not args.skip_models:
        for name in args.models or MODEL_SPECS:
            download_model(name, args.force)
    if not args.skip_datasets:
        local_root = None
        if args.local_dataset_dir:
            local_root = Path(args.local_dataset_dir).expanduser()
            if not local_root.is_absolute():
                local_root = (Path.cwd() / local_root).resolve()
            if not local_root.is_dir():
                raise SystemExit(f"not a directory: {local_root}")
        for name in args.datasets or DATASETS:
            prepare_dataset(name, local_root, args.force)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
