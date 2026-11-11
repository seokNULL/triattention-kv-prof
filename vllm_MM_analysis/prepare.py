#!/usr/bin/env python3
"""Download the vision-language models and video benchmarks.

Models go to experiments/models/<name>, datasets to data/<dataset>.
MVBench is gated: accept its terms on the Hugging Face page and log in
(``huggingface-cli login`` or ``HF_TOKEN``) first.

Example (from vllm_MM_analysis/):
    python prepare.py --models LLaVA-OneVision-7B --skip-datasets
    python prepare.py --skip-models --datasets mvbench --ca-bundle /path/to/company-ca.pem
    python prepare.py            # all models and datasets
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import mvbench
from models import MODELS_DIR, VLM_SPECS

MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py"]
DATASETS = {"mvbench": mvbench}


def set_ca_bundle(path: str) -> None:
    bundle = str(Path(path).expanduser().resolve())
    if not Path(bundle).is_file():
        raise SystemExit(f"CA bundle not found: {bundle}")
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ[key] = bundle


def download_model(name: str, force: bool) -> None:
    from huggingface_hub import snapshot_download

    target = MODELS_DIR / name
    if (target / "config.json").exists() and any(target.glob("*.safetensors")) and not force:
        print(f"[skip] model {name} already at {target}")
        return
    print(f"[download] model {VLM_SPECS[name]['repo']} -> {target}")
    snapshot_download(repo_id=VLM_SPECS[name]["repo"], local_dir=str(target),
                      allow_patterns=MODEL_PATTERNS)


def extract_archives(root: Path, delete: bool) -> None:
    """Unzip every archive under root next to itself, once."""
    for archive in sorted(root.rglob("*.zip")):
        marker = archive.with_name(f".{archive.name}.extracted")
        if marker.exists():
            continue
        print(f"[extract] {archive.relative_to(root)}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(archive.parent)
        marker.touch()
        if delete:
            archive.unlink()


def download_dataset(name: str, patterns: list[str] | None, delete_zips: bool) -> None:
    from huggingface_hub import snapshot_download

    module = DATASETS[name]
    print(f"[download] dataset {module.HF_REPO} -> {module.ROOT}")
    snapshot_download(repo_id=module.HF_REPO, repo_type="dataset", local_dir=str(module.ROOT),
                      allow_patterns=patterns or ["json/*", "video/*"])
    extract_archives(module.ROOT, delete_zips)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", choices=list(VLM_SPECS))
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS))
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument("--patterns", nargs="+",
                   help="Dataset files to fetch (default: json/* video/*), e.g. 'video/clevrer*'.")
    p.add_argument("--delete-zips", action="store_true", help="Remove archives after extracting.")
    p.add_argument("--ca-bundle", help="PEM bundle for TLS-intercepting proxies.")
    p.add_argument("--force", action="store_true", help="Download models even if present.")
    args = p.parse_args()

    if args.ca_bundle:
        set_ca_bundle(args.ca_bundle)
    if not args.skip_models:
        for name in args.models or VLM_SPECS:
            download_model(name, args.force)
    if not args.skip_datasets:
        for name in args.datasets or DATASETS:
            download_dataset(name, args.patterns, args.delete_zips)


if __name__ == "__main__":
    main()
