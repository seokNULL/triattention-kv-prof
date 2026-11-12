#!/usr/bin/env python3
"""Download LongBench v2 and write data/longbench_v2_<short|medium|long>.jsonl.

Fetches the dataset files with huggingface_hub (no ``datasets`` package, whose
recent releases need a newer huggingface_hub than vLLM 0.18 allows) and reads
whichever of .json / .jsonl / .parquet the repo provides. A copy downloaded
elsewhere can be given with --local-file.

Example (from vllm_CoT_analysis/):
    python prepare_longbench.py
    python prepare_longbench.py --ca-bundle /path/to/company-ca.pem
    python prepare_longbench.py --local-file ~/Downloads/data.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import longbench

FIELDS = ["_id", "domain", "sub_domain", "difficulty", "length", "question",
          "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context"]
RAW_DIR = longbench.DATA_DIR / "longbench_v2_raw"


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or not text.lstrip().startswith("["):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def download() -> list[Path]:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=longbench.HF_REPO, repo_type="dataset", local_dir=str(RAW_DIR),
                      allow_patterns=["*.json", "*.jsonl", "*.parquet"])
    files = [p for p in sorted(RAW_DIR.rglob("*")) if p.suffix in {".json", ".jsonl", ".parquet"}
             and ".cache" not in p.parts]
    if not files:
        raise SystemExit(f"no data files downloaded into {RAW_DIR}")
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ca-bundle", help="PEM bundle for TLS-intercepting proxies.")
    p.add_argument("--local-file", help="Use this LongBench v2 data file instead of downloading.")
    args = p.parse_args()
    if args.ca_bundle:
        bundle = str(Path(args.ca_bundle).expanduser().resolve())
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ[key] = bundle

    files = [Path(args.local_file).expanduser()] if args.local_file else download()
    rows = []
    for path in files:
        loaded = [r for r in read_rows(path) if isinstance(r, dict) and "context" in r]
        print(f"{path}: {len(loaded)} questions")
        rows.extend(loaded)
    if not rows:
        raise SystemExit("no LongBench v2 questions found in " + ", ".join(map(str, files)))
    missing = [f for f in FIELDS if f not in rows[0]]
    if missing:
        raise SystemExit(f"unexpected format: fields {missing} missing; found {sorted(rows[0])}")

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
