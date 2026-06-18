"""Classify call transcripts as voicemail / not with the fine-tuned model.

Usage:
    uv run vm-predict "Hi, you've reached Jamie. Leave a message after the tone."
    uv run vm-predict            # runs a few built-in demo transcripts

Designed for the stated goal: classify at high volume, cheaply and fast. The
fine-tuned DistilBERT is ~100 MB, runs on CPU in milliseconds, and costs nothing
per call.
"""

from __future__ import annotations

import os
import sys

from transformers import pipeline

from . import MODEL_DIR

DEMO = [
    "Hi, you've reached Jamie. I'm not able to take your call right now. "
    "Please leave your name and number after the tone. [beep] Hey Jamie, it's Pat "
    "from Acme about your order, call me back at (415) 555-0132.",

    "Agent: Thank you for calling Northwind, this is Alex, how can I help you? "
    "Customer: Hi, I'm calling about the invoice. Agent: Sure, what's your account number?",

    "Thank you for calling Globex. For billing, press 1. For technical support, "
    "press 2. To speak with a representative, press 0.",            # IVR (hard negative)

    "You've reached Taylor— oh, hello? Sorry, I just grabbed the phone. "
    "Yes, this is Taylor speaking. Customer: Hi, calling about the refund.",  # picked up live
]


def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(
            f"No fine-tuned model at '{MODEL_DIR}'.\n"
            "Run `uv run vm-generate-data` then `uv run vm-train` first."
        )

    clf = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        truncation=True,
        max_length=512,
    )

    notes = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO

    for note in notes:
        result = clf(note)[0]
        flag = "VOICEMAIL" if result["label"] == "voicemail" else "not voicemail"
        print(f"\n[{flag}]  (conf {result['score']:.2f})")
        print(f"  {note[:110]}{'...' if len(note) > 110 else ''}")


if __name__ == "__main__":
    main()
