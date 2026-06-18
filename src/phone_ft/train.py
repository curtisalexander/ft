"""Fine-tune DistilBERT into a phone-number extractor (token classification).

This is the core of the example. It:
  1. loads the BIO-tagged data produced by `generate-data`,
  2. tokenizes it and ALIGNS each word-level label to its sub-word tokens
     (the step beginners most often get wrong),
  3. adds a fresh classification HEAD on top of the pretrained DistilBERT body
     and continues training the whole thing (full fine-tuning), and
  4. reports precision / recall / F1 on a held-out test split.

Run with:  uv run train
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from . import BASE_MODEL, DATA_DIR, ID2LABEL, LABEL2ID, LABELS, MODEL_DIR


def tokenize_and_align_labels(batch, tokenizer):
    """Tokenize raw text and assign each token a BIO label from char spans.

    We tokenize the natural note text and ask the tokenizer for each token's
    character offsets. A token is part of a phone number when its start offset
    falls inside one of that note's phone spans. The first such token of a span
    gets B-PHONE, the rest get I-PHONE; everything else is O. Special tokens
    ([CLS], [SEP], pad) get -100 so the loss ignores them.

    Tokenizing the same raw text the model will see at inference avoids any
    train/serve skew.
    """
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        return_offsets_mapping=True,
    )
    all_labels = []
    for i, spans in enumerate(batch["spans"]):
        offsets = tokenized["offset_mapping"][i]
        seq_ids = tokenized.sequence_ids(i)
        label_ids = []
        active_span = None                       # the phone span we're inside
        for j, (start, end) in enumerate(offsets):
            if seq_ids[j] is None or start == end:   # special token / empty
                label_ids.append(-100)
                active_span = None
                continue
            here = next(((s, e) for s, e in spans if s <= start < e), None)
            if here is None:
                label_ids.append(LABEL2ID["O"])
                active_span = None
            elif here == active_span:            # continuing the same phone
                label_ids.append(LABEL2ID["I-PHONE"])
            else:                                # first token of a new phone
                label_ids.append(LABEL2ID["B-PHONE"])
                active_span = here
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    tokenized.pop("offset_mapping")
    return tokenized


def build_metrics():
    """Return a compute_metrics fn using seqeval (entity-level P/R/F1)."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_row, label_row in zip(preds, labels):
            seq_labels, seq_preds = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:                   # ignore special / continuation tokens
                    continue
                seq_labels.append(ID2LABEL[l])
                seq_preds.append(ID2LABEL[p])
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
            "_report": classification_report(true_labels, true_preds, zero_division=0),
        }

    return compute_metrics


def make_training_args(output_dir: str, epochs: float, batch_size: int) -> TrainingArguments:
    """Build TrainingArguments, tolerant of the eval_strategy rename across versions."""
    common = dict(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=25,
        save_total_limit=1,
        report_to=[],  # no wandb/etc.
    )
    try:
        return TrainingArguments(eval_strategy="epoch", save_strategy="no", **common)
    except TypeError:
        # Older transformers used `evaluation_strategy`.
        return TrainingArguments(evaluation_strategy="epoch", save_strategy="no", **common)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for phone NER.")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--out", default=MODEL_DIR)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        path = os.path.join(args.data, f"{split}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run `uv run generate-data` first.")

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ONE line that both ADDS a new 3-way classification head and keeps the
    # pretrained body — exactly the "add a head + adjust the body" idea.
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    raw = load_dataset(
        "json",
        data_files={
            "train": os.path.join(args.data, "train.jsonl"),
            "validation": os.path.join(args.data, "val.jsonl"),
            "test": os.path.join(args.data, "test.jsonl"),
        },
    )

    tokenized = raw.map(
        lambda b: tokenize_and_align_labels(b, tokenizer),
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    trainer = Trainer(
        model=model,
        args=make_training_args(args.out, args.epochs, args.batch_size),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_metrics(),
    )

    print("\nFine-tuning...\n")
    trainer.train()

    print("\nEvaluating on held-out TEST split (data the model never trained on):")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"  precision : {test_metrics['eval_precision']:.4f}")
    print(f"  recall    : {test_metrics['eval_recall']:.4f}")
    print(f"  f1        : {test_metrics['eval_f1']:.4f}")
    if "eval__report" in test_metrics:
        print("\n" + test_metrics["eval__report"])

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved fine-tuned model -> {args.out}")
    print('Next: uv run predict "called from (415) 555-0132, acct 5567891234"')


if __name__ == "__main__":
    main()
