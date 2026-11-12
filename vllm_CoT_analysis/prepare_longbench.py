#!/usr/bin/env python3
"""Download LongBench v2 and write data/longbench_v2_<short|medium|long>.jsonl.

Example (from vllm_CoT_analysis/):
    python prepare_longbench.py
    python prepare_longbench.py --ca-bundle /path/to/company-ca.pem
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import longbench

FIELDS = ["_id", "domain", "sub_domain", "difficulty", "length", "question",
          "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ca-bundle", help="PEM bundle for TLS-intercepting proxies.")
    args = p.parse_args()
    if args.ca_bundle:
        bundle = str(Path(args.ca_bundle).expanduser().resolve())
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ[key] = bundle
    from datasets import load_dataset

    data = load_dataset(longbench.HF_REPO)
    rows = [r for split in data.values() for r in split]
    longbench.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for length in longbench.LENGTHS:
        subset = [{k: r[k] for k in FIELDS} for r in rows if r["length"] == length]
        path = longbench.DATA_DIR / f"longbench_v2_{length}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(subset)} questions")


if __name__ == "__main__":
    main()
