# ft — Fine-Tuning, explained with a worked example

📖 **Read the guide live:** <https://curtisalexander.github.io/ft/>

A small, self-contained learning project about **fine-tuning a language model**.

- **`index.html`** — a from-scratch guide: what fine-tuning is, when you *can* and
  *can't* do it, why closed frontier models (ChatGPT/Claude) can't be fine-tuned,
  the difference between encoder / encoder-decoder / decoder-only models, how to
  navigate Hugging Face, and why **getting good data is the hard part**. This is the
  GitHub Pages page.
- **`src/phone_ft/`** — a complete, runnable example that fine-tunes
  **DistilBERT** into a phone-number extractor for messy customer-service notes.

> **Scenario.** Customer-service agents type free-text notes containing phone
> numbers in inconsistent formats — `(415) 555-0132`, `415-555-0132`,
> `4155550132`, `415.555.0132`, `+1 …`, even 7-digit local numbers — mixed in with
> account numbers, order IDs, dates, and dollar amounts. A plain regex grabs the
> account numbers too. A fine-tuned model uses **context** to pull out *only* the
> phone numbers.

## Why an encoder model?

Pulling spans out of text is a **labeling** problem, which **encoder-only** models
(BERT/DistilBERT) with a token-classification head do best — far less data and
compute than a big generative model. See the guide for the full reasoning.

## Quick start (uses [`uv`](https://docs.astral.sh/uv/) only)

```bash
# install uv once (macOS/Linux):
#   curl -LsSf https://astral.sh/uv/install.sh | sh

uv run generate-data                       # 1. make labeled synthetic data -> data/*.jsonl
uv run train                               # 2. fine-tune -> ./phone-ner-model  (downloads torch + base model on first run)
uv run predict "call me at (415) 555-0132, acct 5567891234"   # 3. try it
```

You never run `pip install` — `uv run` reads `pyproject.toml`, builds the
environment on first use, and caches it.

### What each command does

| Command | File | What it does |
| --- | --- | --- |
| `uv run generate-data` | `generate_data.py` | Builds realistic notes with phones in every format **plus hard negatives** (account numbers, dates, amounts), perfectly BIO-labeled. Writes `data/train.jsonl`, `val.jsonl`, `test.jsonl`. |
| `uv run train` | `train.py` | Tokenizes, aligns labels to sub-words, adds a classification head to DistilBERT, fine-tunes, and reports precision/recall/F1 on the held-out test split. |
| `uv run predict "..."` | `predict.py` | Loads `./phone-ner-model` and extracts (and normalizes) phone numbers from any note. With no argument, runs built-in demo notes. |

## Publishing the guide on GitHub Pages

1. Create a GitHub repo named **`ft`** and push this folder.
2. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
   pick your default branch and `/ (root)`.
3. `index.html` is served at `https://<you>.github.io/ft/`.

## Notes

- Training runs fine on a laptop CPU (or Apple-Silicon MPS) in a few minutes;
  DistilBERT is small, so we do **full** fine-tuning. For large LLMs you'd use
  LoRA/PEFT instead — same idea, fewer weights updated.
- The synthetic data exists so you can run end-to-end *today*. In a real project
  you'd mix in (anonymized) real agent notes — that's the work that actually
  determines quality.
