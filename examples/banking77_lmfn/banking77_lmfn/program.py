"""The task: classify a bank customer's message into one of 77 intents."""

from typing import Literal

import lmcc
from datasets import load_dataset

import lmfn

DATASET = "mteb/banking77"
LABELS = tuple(sorted(set(load_dataset(DATASET, split="test")["label_text"])))
Intent = Literal[LABELS]


@lmfn.ai
def classify(text: str) -> Intent:
    """Classify the bank customer's message by what they need."""


# Compact on purpose: at scale every token is paid for. The categories are
# written once (the enum placeholder), the reply is one short line.
ADAPTERS = {
    "compact": lmcc.adapter(name="classify_compact", messages=[
        lmcc.system("{instruction}\n\nReply with one line:\nIntent: {answer}"),
        lmcc.user("{text}"),
    ]),
}
