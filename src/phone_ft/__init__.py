"""Fine-tuning example: extract phone numbers from customer-service notes.

A small, self-contained demonstration of fine-tuning an encoder model
(DistilBERT) for token classification (NER-style span extraction).
"""

__version__ = "0.1.0"

# The label scheme (BIO tagging) shared by every module.
LABELS = ["O", "B-PHONE", "I-PHONE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Where things live, relative to the repo root. Each use case is namespaced so
# the two examples never collide.
DATA_DIR = "data/phone"
MODEL_DIR = "models/phone-ner"
BASE_MODEL = "distilbert-base-uncased"
