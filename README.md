# ft — Fine-Tuning, explained with worked examples

📖 **Read the guide:** open [`docs/index.html`](docs/index.html) in a browser, or serve it locally (see [Viewing the guide](#viewing-the-guide)).

A small, self-contained learning project about **fine-tuning a language model**.
It pairs a from-scratch written guide with **two complete, runnable examples** that
fine-tune **DistilBERT** for two different *task shapes*.

- **`docs/index.html`** — the guide: what fine-tuning is, when you *can* and
  *can't* do it, why closed frontier models (ChatGPT/Claude) can't be fine-tuned,
  the difference between encoder / encoder-decoder / decoder-only models, how to
  navigate Hugging Face, and why **getting good data is the hard part**.
- **`src/phone_ft/`** — Example A: **token classification** (extract phone numbers).
- **`src/voicemail_clf/`** — Example B: **sequence classification** (is this a voicemail?).

## The two use cases

| | Example A — phone numbers | Example B — voicemail |
| --- | --- | --- |
| Package | `src/phone_ft/` | `src/voicemail_clf/` |
| Task shape | **Token classification** (NER) | **Sequence classification** |
| Label | A tag per token (BIO) | One label per transcript |
| HF model class | `AutoModelForTokenClassification` | `AutoModelForSequenceClassification` |
| Question | "Which spans are phone numbers?" | "Is this whole transcript a voicemail?" |

Both use the same base model (`distilbert-base-uncased`) and the same `uv` +
Transformers + `Trainer` stack — the differences are the task shape, the head, and
the data. That's the lesson: **once you know the pattern, new tasks are mostly new
data, not new code.**

> **Example A — phone numbers.** Agent notes contain phone numbers in inconsistent
> formats — `(415) 555-0132`, `415-555-0132`, `4155550132`, `415.555.0132`, `+1 …`,
> even 7-digit local numbers — mixed with account numbers, order IDs, dates, and
> amounts. A regex grabs the account numbers too; a fine-tuned model uses **context**
> to pull out *only* phone numbers.
>
> **Example B — voicemail.** Given a call transcript, decide whether it's a voicemail
> or a live conversation, **at high volume, cheaply and fast**. The training data
> deliberately includes **hard negatives** — IVR phone menus, hold messages, and live
> calls that *open* with a voicemail greeting — so the model can't cheat with an
> "automated = voicemail" shortcut.

## Quick start (uses [`uv`](https://docs.astral.sh/uv/) only)

```bash
# install uv once (macOS/Linux):
#   curl -LsSf https://astral.sh/uv/install.sh | sh

# Example A — phone numbers (token classification)
uv run phone-generate-data                                              # 1. labeled synthetic data -> data/phone/*.jsonl
uv run phone-train                                                      # 2. fine-tune -> ./models/phone-ner
uv run phone-predict "call me at (415) 555-0132, acct 5567891234"      # 3. try it

# Example B — voicemail (sequence classification)
uv run vm-generate-data                                                 # 1. labeled synthetic data -> data/voicemail/*.jsonl
uv run vm-train                                                         # 2. fine-tune -> ./models/voicemail-clf
uv run vm-predict "Hi, you've reached Jamie. Leave a message after the tone."   # 3. try it
```

You never run `pip install` — `uv run` reads `pyproject.toml`, builds the
environment on first use, and caches it. The first run downloads PyTorch + the base
model (a few hundred MB); after that it's instant. Each example trains in a few
minutes on a laptop CPU.

### What each command does

**Example A — phone numbers** (`src/phone_ft/`)

| Command | What it does |
| --- | --- |
| `uv run phone-generate-data` | Builds notes with phones in every format **plus hard negatives** (account numbers, dates, amounts), perfectly span-labeled. Writes `data/phone/{train,val,test}.jsonl`. |
| `uv run phone-train` | Tokenizes raw text, aligns labels to sub-word tokens via char offsets, adds a token-classification head to DistilBERT, fine-tunes, reports precision/recall/F1. Saves to `./models/phone-ner`. |
| `uv run phone-predict "..."` | Extracts and normalizes phone numbers from a note. No argument → demo notes. |

**Example B — voicemail** (`src/voicemail_clf/`)

| Command | What it does |
| --- | --- |
| `uv run vm-generate-data` | Builds synthetic transcripts — voicemails plus **hard negatives** (IVR menus, hold messages, calls that pick up after a voicemail greeting). Writes `data/voicemail/{train,val,test}.jsonl`. |
| `uv run vm-train` | Tokenizes (truncating to 512 tokens), adds a 2-way sequence-classification head, fine-tunes, reports accuracy/precision/recall/F1 **and a confusion matrix** (classes are imbalanced). Saves to `./models/voicemail-clf`. |
| `uv run vm-predict "..."` | Classifies a transcript as `voicemail` / `not voicemail`. No argument → demo transcripts. |

## Project layout

```
ft/
├── docs/
│   ├── index.html        # the written guide
│   └── .nojekyll
├── src/
│   ├── phone_ft/         # Example A — token classification
│   │   ├── generate_data.py  train.py  predict.py
│   └── voicemail_clf/    # Example B — sequence classification
│       ├── generate_data.py  train.py  predict.py
├── pyproject.toml        # uv-managed; defines the phone-* and vm-* commands
└── README.md
```

Generated data (`data/`) and fine-tuned models (`models/`) are git-ignored —
regenerate them with the `*-generate-data` and `*-train` commands.

## Viewing the guide

`docs/index.html` is fully self-contained (no external assets) — just **open the file
in your browser** (double-click it, or open `file://` to it). No server needed.

The `docs/` layout with a `.nojekyll` file is also the standard structure for serving
the guide from GitHub Pages later (Settings → Pages → Source: `docs/`), should you
ever want to.

## Notes

- Training runs fine on a laptop CPU (or Apple-Silicon MPS); DistilBERT is small, so
  we do **full** fine-tuning. For large LLMs you'd use LoRA/PEFT instead — same idea,
  fewer weights updated.
- The synthetic data exists so you can run end-to-end *today*. In a real project you'd
  mix in (anonymized) real notes/transcripts — that's the work that actually
  determines quality.
