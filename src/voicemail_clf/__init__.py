"""Fine-tuning example: classify whether a call transcript is a voicemail.

A small, self-contained demonstration of fine-tuning an encoder model
(DistilBERT) for SEQUENCE classification (one label for the whole transcript).

Contrast with the phone example, which is TOKEN classification (a label per
token). Same base model and tooling; different task shape and head.
"""

__version__ = "0.1.0"

# Binary labels. Order matters: index 1 ("voicemail") is the positive class.
LABELS = ["not_voicemail", "voicemail"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
POSITIVE_ID = LABEL2ID["voicemail"]

# Where things live, relative to the repo root (namespaced per use case).
DATA_DIR = "data/voicemail"
MODEL_DIR = "models/voicemail-clf"
BASE_MODEL = "distilbert-base-uncased"
