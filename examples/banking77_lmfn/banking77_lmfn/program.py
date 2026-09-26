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
    # For the trained student: no category list, ~9x fewer prompt tokens.
    # The intent is still read by the kernel (forgiving, reported, strict-able);
    # only what the model is told changes (lmcc §5 descriptions).
    "short": lmcc.adapter(name="classify_short", messages=[
        lmcc.system("{instruction}\nIntent: {answer}"),
        lmcc.user("{text}"),
    ], formats={"enum": {"describe": "<intent>"}}),
    "compact": lmcc.adapter(name="classify_compact", messages=[
        lmcc.system("{instruction}\n\nReply with one line:\nIntent: {answer}"),
        lmcc.user("{text}"),
    ]),
}


# The distilled students' layout is saved beside them as lmcc artifacts
# (student/adapter.json, student/signature.json) and rendered by
# student_prompt.py with lmcc alone; check_prompts.py proves that those files
# are ADAPTERS["short"] and this function's signature.
