"""Generate a labeled, synthetic dataset of customer-service notes.

Real projects spend most of their effort HERE — acquiring and labeling data.
When real labeled data is scarce, a legitimate way to bootstrap is to generate
it: drop *known* phone numbers (and known look-alike distractors such as account
numbers, order IDs, dates, and dollar amounts) into realistic note templates.
Because we placed every entity ourselves, the labels are perfect and free.

We emit NATURAL text plus character spans for the phone numbers:
    {"text": "Left voicemail at (415) 555-0132 on 06/18/2026.",
     "spans": [[18, 32]]}                 # [start, end) of each phone number

Training on raw text (rather than pre-split tokens) makes the data look exactly
like what the model sees at inference time — no train/serve skew.

We deliberately include:
  * every messy phone format (parens / no parens, dashes / dots / spaces / none,
    a leading +1, and 7-digit local numbers), and
  * HARD NEGATIVES — account numbers, order IDs, ZIPs, dates, amounts — so the
    model learns from *context* which digit-runs are phones, not just "long
    number = phone" (which is exactly where a naive regex fails).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from . import DATA_DIR

# --------------------------------------------------------------------------- #
# Low-level random pieces
# --------------------------------------------------------------------------- #

def _digits(n: int) -> str:
    return "".join(random.choice("0123456789") for _ in range(n))


def _area() -> str:
    # Plausible-looking US area/exchange codes (no leading 0/1).
    return random.choice("23456789") + _digits(2)


# --------------------------------------------------------------------------- #
# Entity generators. Phone returns the literal text as it would appear in a
# note (internal spaces are fine — it's natural text now). Distractors return
# look-alike strings that are NOT phones.
# --------------------------------------------------------------------------- #

def gen_phone() -> str:
    """A phone number in one of many inconsistent real-world formats."""
    a, p, l = _area(), _area(), _digits(4)
    return random.choice([
        f"({a}) {p}-{l}",     # (415) 555-0132
        f"({a}) {p}{l}",      # (415) 5550132
        f"{a}-{p}-{l}",       # 415-555-0132
        f"{a}.{p}.{l}",       # 415.555.0132
        f"{a} {p} {l}",       # 415 555 0132
        f"{a}{p}{l}",         # 4155550132   (10 straight)
        f"+1 {a}-{p}-{l}",    # +1 415-555-0132
        f"+1{a}{p}{l}",       # +14155550132
        f"{p}-{l}",           # 555-0132     (7-digit local)
        f"{p}{l}",            # 5550132      (7 straight)
    ])


def gen_account() -> str:
    return random.choice([
        f"#{_digits(random.randint(8, 12))}",
        _digits(random.randint(8, 12)),
        f"AC-{_digits(6)}",
    ])


def gen_order() -> str:
    return f"#{random.choice('ABCDEFGH')}{_digits(random.randint(6, 8))}"


def gen_date() -> str:
    m, d, y = random.randint(1, 12), random.randint(1, 28), random.randint(22, 26)
    sep = random.choice(["/", "-"])
    return f"{m:02d}{sep}{d:02d}{sep}20{y}"


def gen_amount() -> str:
    return f"${random.randint(1, 4999)}.{_digits(2)}"


def gen_zip() -> str:
    return _digits(5)


SLOTS = {
    "PHONE": gen_phone,
    "ACCT": gen_account,
    "ORDER": gen_order,
    "DATE": gen_date,
    "AMOUNT": gen_amount,
    "ZIP": gen_zip,
}

# --------------------------------------------------------------------------- #
# Note templates written as NATURAL text (punctuation attached as in real life).
# Some templates contain NO phone at all, so the model learns not to invent one
# from an account number or amount.
# --------------------------------------------------------------------------- #

TEMPLATES = [
    "Customer called from {PHONE} regarding order {ORDER}.",
    "Spoke with caller, best callback number is {PHONE}, account {ACCT}.",
    "Cust requested refund of {AMOUNT} on acct {ACCT}; reach them at {PHONE}.",
    "Left voicemail at {PHONE} on {DATE}, no answer.",
    "Updated billing for account {ACCT}. Confirm by phone {PHONE}.",
    "New number on file: {PHONE}. Previous order {ORDER} shipped {DATE}.",
    "Caller verified identity, phone {PHONE}, zip {ZIP}.",
    "Follow up needed. Account {ACCT}, order {ORDER}, amount {AMOUNT}.",
    "Reached customer at {PHONE}; they will mail check for {AMOUNT}.",
    "Please text {PHONE} after {DATE} to reschedule.",
    "No phone provided. Account {ACCT}, order {ORDER}, total {AMOUNT}.",
    "Customer in zip {ZIP}, account {ACCT}, disputes charge of {AMOUNT}.",
    "Two contacts given: {PHONE} and {PHONE}. Acct {ACCT}.",
    "Wrong number, do not call {PHONE}. Correct one is {PHONE}.",
    "Shipment for order {ORDER} delayed until {DATE}; notified at {PHONE}.",
    "Account {ACCT} closed on {DATE}. Refund {AMOUNT} issued.",
    "cust says call cell {PHONE} not the home line {PHONE}",
    "billing q only, acct {ACCT}, no callback requested",
]

_SLOT_RE = re.compile(r"\{([A-Z]+)\}")


def render(template: str):
    """Expand one template into {'text', 'spans'} with exact phone char-spans."""
    text = ""
    spans = []
    last = 0
    for m in _SLOT_RE.finditer(template):
        text += template[last:m.start()]           # literal text before the slot
        value = SLOTS[m.group(1)]()
        start = len(text)
        text += value
        if m.group(1) == "PHONE":
            spans.append([start, len(text)])
        last = m.end()
    text += template[last:]                          # trailing literal
    return {"text": text, "spans": spans}


def build(n: int):
    return [render(random.choice(TEMPLATES)) for _ in range(n)]


def _write(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic phone-NER data.")
    parser.add_argument("--n-train", type=int, default=1600)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DATA_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Generate splits independently so there is NO leakage between them.
    for name, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        rows = build(n)
        path = os.path.join(args.out, f"{name}.jsonl")
        _write(path, rows)
        print(f"  wrote {n:>5} examples -> {path}")

    # Show one example so the format is concrete.
    sample = build(1)[0]
    print("\nExample row:")
    print("  text  :", sample["text"])
    print("  spans :", sample["spans"], "->",
          [sample["text"][s:e] for s, e in sample["spans"]])
    print("\nNext: uv run train")


if __name__ == "__main__":
    main()
