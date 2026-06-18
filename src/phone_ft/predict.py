"""Run the fine-tuned model on new notes and pull out phone numbers.

Usage:
    uv run predict "Cust called from (415) 555-0132, acct 5567891234, owes $42.50"
    uv run predict            # runs a few built-in demo notes

The model finds the spans from CONTEXT; we then normalize the matched digits
into a canonical form — the practical model-plus-regex hybrid described in the
guide (the model decides *which* digit-runs are phones; simple code cleans them).
"""

from __future__ import annotations

import os
import re
import sys

from transformers import pipeline

from . import MODEL_DIR

DEMO_NOTES = [
    "Customer called from (415) 555-0132 regarding order #B284917.",
    "No callback wanted. Account 5567891234, refund of $42.50 issued on 06/18/2026.",
    "Reach them at 415.555.0188 or the home line 555-0177, acct AC-998120.",
    "Verified id, zip 94107, phone +1 628-555-0143, order A1029384.",
]


def normalize(text: str) -> str:
    """Reduce a matched span to its digits for a canonical, comparable form."""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"+1 ({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 7:
        return f"{digits[0:3]}-{digits[3:]}"
    return text.strip()


def extract(ner, note: str):
    """Return the list of phone spans the model found in a single note."""
    spans = ner(note)
    phones = []
    for span in spans:
        # aggregation_strategy="simple" groups B-/I- tokens into one entity.
        group = span.get("entity_group") or span.get("entity") or ""
        if group.endswith("PHONE"):
            raw = note[span["start"]:span["end"]]
            phones.append((raw, normalize(raw), float(span["score"])))
    return phones


def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(
            f"No fine-tuned model at '{MODEL_DIR}'.\n"
            "Run `uv run generate-data` then `uv run train` first."
        )

    ner = pipeline(
        "token-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        aggregation_strategy="simple",
    )

    notes = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_NOTES

    for note in notes:
        print("\nNOTE:", note)
        phones = extract(ner, note)
        if not phones:
            print("  (no phone numbers found)")
        for raw, norm, score in phones:
            print(f"  phone: {raw!r:30} -> {norm:24} (conf {score:.2f})")


if __name__ == "__main__":
    main()
