"""Generate a labeled, synthetic dataset of call transcripts.

As with the phone example, acquiring/labeling data is the real work. Here we
bootstrap with synthetic transcripts so you can run end-to-end today; in
production you'd mix in (anonymized) real transcripts.

Each example is one transcript plus a single label:
    {"text": "Hi, you've reached Jamie...", "label": "voicemail"}
    {"text": "Agent: Thanks for calling Acme...", "label": "not_voicemail"}

The interesting part is the HARD NEGATIVES — things that are automated or sound
voicemail-ish but are NOT voicemails. Without them the model learns a lazy rule
("automated/short = voicemail") and fails exactly there. We include:
  * IVR / phone menus  ("press 1 for billing")          -> automated, not a VM
  * hold / queue messages ("please remain on the line") -> automated, not a VM
  * live calls that OPEN with voicemail-ish phrasing but then a human picks up
  * very short live calls

The voicemail "tell" is almost always at the very start, which is convenient:
even when transcripts get long, truncation to the first tokens keeps the signal.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from . import DATA_DIR

NAMES = ["Jamie", "Pat", "Alex", "Taylor", "Jordan", "Morgan", "Casey",
         "Riley", "Sam", "Chris", "Dana", "Lee", "Robin", "Quinn"]
COMPANIES = ["Acme", "Northwind", "Globex", "Initech", "Umbrella",
             "Contoso", "Cygnus", "Vertex", "Stark Supply", "Wayne Logistics"]
REASONS = ["your recent order", "the invoice", "your appointment",
           "the warranty claim", "your account balance", "the delivery delay",
           "the refund request", "scheduling a callback", "your service plan",
           "the replacement part"]


def _name() -> str:
    return random.choice(NAMES)


def _company() -> str:
    return random.choice(COMPANIES)


def _reason() -> str:
    return random.choice(REASONS)


def _phone() -> str:
    d = lambda n: "".join(random.choice("0123456789") for _ in range(n))
    a = random.choice("23456789") + d(2)
    return random.choice([f"({a}) {d(3)}-{d(4)}", f"{a}-{d(3)}-{d(4)}", f"{a}{d(3)}{d(4)}"])


# --------------------------------------------------------------------------- #
# POSITIVES — actual voicemails
# --------------------------------------------------------------------------- #

def vm_personal_greeting() -> str:
    """A personal greeting, a beep, and (usually) a message left."""
    name, caller, company, reason = _name(), _name(), _company(), _reason()
    greet = random.choice([
        f"Hi, you've reached {name}. I'm not able to take your call right now. "
        f"Please leave your name, number, and a brief message after the tone.",
        f"Hello, this is {name}. Sorry I missed you. Leave a message and I'll call you back.",
        f"You've reached the voicemail of {name}. I can't get to the phone, "
        f"so please record your message after the beep.",
    ])
    beep = random.choice(["[beep]", "BEEP", "*beep*", "[tone]"])
    if random.random() < 0.8:                      # most leave a message
        msg = random.choice([
            f"Hey {name}, it's {caller} from {company} calling about {reason}. "
            f"Give me a call back at {_phone()} when you get a chance. Thanks!",
            f"Hi {name}, this is {caller} regarding {reason}. You can reach me at "
            f"{_phone()}. Talk soon, bye.",
            f"{name}, {caller} here. Just following up on {reason} — no rush, "
            f"call me back whenever. My number is {_phone()}.",
        ])
        return f"{greet} {beep} {msg}"
    return f"{greet} {beep}"                        # hang-up, no message


def vm_carrier() -> str:
    """A generic carrier/automated voicemail prompt."""
    name = _name()
    return random.choice([
        f"The person you are trying to reach, {name}, is not available. "
        f"Please leave a message after the tone. [beep] "
        f"Hi {name}, it's {_name()}, call me back at {_phone()}.",
        "Your call has been forwarded to an automated voice messaging system. "
        f"The person you are calling is unavailable. At the tone, please record your "
        f"message. [tone] Hey, it's {_name()} from {_company()}, give me a ring.",
        f"The wireless customer you are calling is not available right now. "
        f"Please try your call again later, or leave a message after the tone. [beep] "
        f"Hi, this is {_name()} about {_reason()}.",
    ])


# --------------------------------------------------------------------------- #
# NEGATIVES — including HARD negatives
# --------------------------------------------------------------------------- #

def live_call() -> str:
    """A normal two-party live conversation (clear negative)."""
    agent, company, reason = _name(), _company(), _reason()
    turns = [
        f"Agent: Thank you for calling {company}, this is {agent}, how can I help you today?",
        f"Customer: Hi, I'm calling about {reason}.",
        "Agent: I can help with that. Can I get your account number to pull up your details?",
        f"Customer: Sure, it's {''.join(random.choice('0123456789') for _ in range(8))}.",
        "Agent: Thanks, I see your account here. Let me take a look.",
        random.choice([
            "Customer: Great, I appreciate it.",
            "Customer: How long will this take?",
            "Customer: Okay, and will there be any extra charge?",
        ]),
        "Agent: No problem, I've gone ahead and taken care of that for you.",
        "Customer: Perfect, thanks so much. Have a good day.",
    ]
    return " ".join(turns)


def ivr_menu() -> str:
    """HARD NEGATIVE: an automated phone menu — automated, but not a voicemail."""
    company = _company()
    return random.choice([
        f"Thank you for calling {company}. Your call may be recorded for quality and "
        f"training. For billing, press 1. For technical support, press 2. To speak with "
        f"a representative, press 0. To repeat this menu, press 9.",
        f"Welcome to {company}. If you know your party's extension, you may dial it at "
        f"any time. For sales, press 1. For support, press 2. Para español, marque tres.",
        f"You've reached {company} automated services. For order status, press 1. "
        f"For returns, press 2. For all other inquiries, please stay on the line.",
    ])


def hold_message() -> str:
    """HARD NEGATIVE: queue / hold message — automated, but not a voicemail."""
    wait = random.randint(2, 20)
    return random.choice([
        "Your call is important to us. Please remain on the line and the next available "
        f"representative will be with you shortly. Your estimated wait time is {wait} minutes.",
        "All of our agents are currently assisting other customers. Please stay on the "
        f"line and your call will be answered in the order it was received.",
        f"Thank you for your patience. You are number {random.randint(2, 9)} in the queue. "
        "Please continue to hold.",
    ])


def picked_up_after_prompt() -> str:
    """HARD NEGATIVE: opens with voicemail-ish phrasing, but a human picks up live."""
    name, caller, company, reason = _name(), _name(), _company(), _reason()
    return random.choice([
        f"You've reached {name}— oh, hello? Sorry, I just grabbed the phone. "
        f"Yes, this is {name} speaking. Customer: Hi {name}, it's {caller} from {company} "
        f"about {reason}. {name}: Oh great, perfect timing, go ahead.",
        f"Hi, you've reached {name}, please leave a— wait, hold on, I'm here! "
        f"Hello? Customer: Hey, is this {name}? {name}: Yes it is, sorry about that, "
        f"what can I do for you? Customer: Calling about {reason}.",
    ])


def short_live() -> str:
    """A very short live call (negative)."""
    name, company, reason = _name(), _company(), _reason()
    return random.choice([
        f"Hello? Customer: Hi, is this {name}? {name}: Yes it is. "
        f"Customer: Great, I'm calling from {company} about {reason}. Is now a good time? "
        f"{name}: Sure, go ahead.",
        f"{name}: Hello, this is {name}. Customer: Hey {name}, quick question about "
        f"{reason}. {name}: Of course, what's up?",
    ])


# (generator, label, sampling weight)
GENERATORS = [
    (vm_personal_greeting, "voicemail", 28),
    (vm_carrier, "voicemail", 17),
    (live_call, "not_voicemail", 22),
    (ivr_menu, "not_voicemail", 11),       # hard negative
    (hold_message, "not_voicemail", 8),    # hard negative
    (picked_up_after_prompt, "not_voicemail", 9),  # hard negative
    (short_live, "not_voicemail", 5),
]
_FUNCS = [g for g, _, _ in GENERATORS]
_LABELS = [lab for _, lab, _ in GENERATORS]
_WEIGHTS = [w for _, _, w in GENERATORS]


def build(n: int):
    rows = []
    for _ in range(n):
        idx = random.choices(range(len(GENERATORS)), weights=_WEIGHTS, k=1)[0]
        rows.append({"text": _FUNCS[idx](), "label": _LABELS[idx]})
    return rows


def _write(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic voicemail-classification data.")
    parser.add_argument("--n-train", type=int, default=1400)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DATA_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    for name, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        rows = build(n)
        path = os.path.join(args.out, f"{name}.jsonl")
        _write(path, rows)
        n_vm = sum(1 for r in rows if r["label"] == "voicemail")
        print(f"  wrote {n:>5} examples ({n_vm} voicemail / {n - n_vm} not) -> {path}")

    sample = build(2)
    print("\nExample rows:")
    for r in sample:
        print(f"  [{r['label']:>13}] {r['text'][:90]}...")
    print("\nNext: uv run vm-train")


if __name__ == "__main__":
    main()
