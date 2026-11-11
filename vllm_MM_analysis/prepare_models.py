#!/usr/bin/env python3
"""Download the vision-language models to experiments/models/<name>.

Example (from vllm_MM_analysis/):
    python prepare_models.py --models LLaVA-OneVision-7B
    python prepare_models.py --ca-bundle /path/to/company-ca.pem
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from models import MODELS_DIR, VLM_SPECS

FILE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", choices=list(VLM_SPECS), default=list(VLM_SPECS))
    p.add_argument("--ca-bundle", help="PEM bundle for TLS-intercepting proxies.")
    p.add_argument("--force", action="store_true", help="Download even if present.")
    args = p.parse_args()

    if args.ca_bundle:
        bundle = str(Path(args.ca_bundle).expanduser().resolve())
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ[key] = bundle
    from huggingface_hub import snapshot_download

    for name in args.models:
        target = MODELS_DIR / name
        if (target / "config.json").exists() and any(target.glob("*.safetensors")) and not args.force:
            print(f"[skip] {name} already at {target}")
            continue
        print(f"[download] {VLM_SPECS[name]['repo']} -> {target}")
        snapshot_download(repo_id=VLM_SPECS[name]["repo"], local_dir=str(target),
                          allow_patterns=FILE_PATTERNS)


if __name__ == "__main__":
    main()
