"""LongBench v2 (THUDM/LongBench-v2): 503 four-choice questions over long contexts.

Stored as one dataset per context length, data/longbench_v2_<short|medium|long>.jsonl
(written by prepare_longbench.py). Prompt and answer extraction follow the
official 0-shot setting ("The correct answer is (X)"); a context longer than
the input budget is cut in the middle, keeping its first and last halves, as
the official evaluation does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
HF_REPO = "THUDM/LongBench-v2"
LENGTHS = ["short", "medium", "long"]
DATASETS = [f"longbench_v2_{length}" for length in LENGTHS]
# Prompt + generation; the input is cut to MAX_LENGTH - MAX_NEW_TOKENS tokens.
MAX_LENGTH = 32768
MAX_NEW_TOKENS = 8192

TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)"."""


def is_longbench(dataset: str) -> bool:
    return dataset in DATASETS


def dataset_path(dataset: str) -> Path:
    path = DATA_DIR / f"{dataset}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python prepare_longbench.py")
    return path


def load_rows(dataset: str, max_examples: int | None = None) -> list[dict]:
    rows = []
    with dataset_path(dataset).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["index"] = len(rows)
                rows.append(row)
                if max_examples and len(rows) >= max_examples:
                    break
    return rows


def build_prompt_ids(row: dict, tokenizer, max_input_tokens: int, chat_template: bool):
    """Token ids of the prompt, the context cut in the middle to fit max_input_tokens.

    Returns (prompt_ids, prompt_text, number of context tokens removed).
    """
    def render(context: str) -> str:
        text = TEMPLATE.format(context=context, **{k: row[k] for k in (
            "question", "choice_A", "choice_B", "choice_C", "choice_D")})
        if chat_template:
            text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                 tokenize=False, add_generation_prompt=True)
        return text

    context_ids = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
    overhead = len(tokenizer(render(""), add_special_tokens=False)["input_ids"])
    budget = max_input_tokens - overhead
    removed = max(0, len(context_ids) - budget)
    if removed:
        half = budget // 2
        context = (tokenizer.decode(context_ids[:half], skip_special_tokens=True)
                   + tokenizer.decode(context_ids[len(context_ids) - (budget - half):],
                                      skip_special_tokens=True))
    else:
        context = row["context"]
    text = render(context)
    ids = tokenizer(text, add_special_tokens=not chat_template)["input_ids"]
    return ids, text, removed


def extract_answer(text: str) -> str | None:
    """Letter after 'The correct answer is', from the final answer (after </think>) if present."""
    answer_part = text.split("</think>")[-1].replace("*", "")
    for part in (answer_part, text.replace("*", "")):
        matches = re.findall(r"The correct answer is \(?([A-D])\)?", part)
        if matches:
            return matches[-1]
    return None


def summarize(records: list[dict]) -> dict:
    """Accuracy overall and per domain / difficulty."""
    def acc(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else 0.0

    def by(key):
        groups: dict[str, list] = {}
        for r in records:
            groups.setdefault(r[key], []).append(r)
        return {k: {"items": len(v), "accuracy": acc(v)} for k, v in sorted(groups.items())}

    n = max(len(records), 1)
    return {
        "items": len(records),
        "accuracy": acc(records),
        "no_answer": sum(r["prediction"] is None for r in records),
        "truncated_context": sum(r["context_tokens_removed"] > 0 for r in records),
        "hit_length_limit": sum(r["finish_reason"] == "length" for r in records),
        "mean_prompt_tokens": round(sum(r["prefill_tokens"] for r in records) / n, 1),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in records) / n, 1),
        "per_domain": by("domain"),
        "per_difficulty": by("difficulty"),
    }
