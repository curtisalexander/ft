"""Fine-tune DistilBERT to classify a transcript as voicemail / not.

This is SEQUENCE classification: one label for the whole transcript. Compared to
the phone example (token classification), the head is simpler (one prediction per
document) and labeling is cheaper (one label per example, no spans).

It:
  1. loads the labeled transcripts produced by `vm-generate-data`,
  2. tokenizes each transcript (truncating to the first 512 tokens — the
     voicemail signal lives at the start),
  3. adds a 2-way classification HEAD on top of DistilBERT and fine-tunes, and
  4. reports accuracy / precision / recall / F1 and a confusion matrix on a
     held-out test split (precision/recall matter because classes are imbalanced).

Run with:  uv run vm-train
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from . import BASE_MODEL, DATA_DIR, ID2LABEL, LABEL2ID, LABELS, MODEL_DIR, POSITIVE_ID

MAX_LEN = 512


def preprocess(batch, tokenizer):
    enc = tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)
    enc["labels"] = [LABEL2ID[label] for label in batch["label"]]
    return enc


def build_metrics():
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=POSITIVE_ID, zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision,   # of calls flagged voicemail, how many were
            "recall": recall,         # of real voicemails, how many we caught
            "f1": f1,
            "_confusion": cm.tolist(),
        }

    return compute_metrics


def make_training_args(output_dir: str, epochs: float, batch_size: int) -> TrainingArguments:
    common = dict(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=25,
        save_total_limit=1,
        report_to=[],
    )
    try:
        return TrainingArguments(eval_strategy="epoch", save_strategy="no", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", save_strategy="no", **common)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for voicemail classification.")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--out", default=MODEL_DIR)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        path = os.path.join(args.data, f"{split}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run `uv run vm-generate-data` first.")

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ONE line that adds a fresh 2-way classification head on the pretrained body.
    model = AutoModelForSequenceClassification.from_pretrained(
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
        lambda b: preprocess(b, tokenizer),
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    trainer = Trainer(
        model=model,
        args=make_training_args(args.out, args.epochs, args.batch_size),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_metrics(),
    )

    print("\nFine-tuning...\n")
    trainer.train()

    print("\nEvaluating on held-out TEST split (data the model never trained on):")
    m = trainer.evaluate(tokenized["test"])
    print(f"  accuracy  : {m['eval_accuracy']:.4f}")
    print(f"  precision : {m['eval_precision']:.4f}")
    print(f"  recall    : {m['eval_recall']:.4f}")
    print(f"  f1        : {m['eval_f1']:.4f}")
    if "eval__confusion" in m:
        (tn, fp), (fn, tp) = m["eval__confusion"]
        print("\n  confusion matrix:")
        print(f"                     pred not_vm   pred voicemail")
        print(f"    actual not_vm      {tn:>8}        {fp:>8}")
        print(f"    actual voicemail   {fn:>8}        {tp:>8}")

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved fine-tuned model -> {args.out}")
    print('Next: uv run vm-predict "Hi, you\'ve reached Jamie, leave a message after the tone."')


if __name__ == "__main__":
    main()
