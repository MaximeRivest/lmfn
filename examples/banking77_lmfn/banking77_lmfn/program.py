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


def _read_intent(capture, field):
    from lmcc import core
    where = f"field {field.name!r}"
    try:
        return core.read_value(field.shape, capture.text, where=where)
    except lmcc.Refusal:          # the kernel's forgiving read (lmcc §7a): case, quotes, a period
        return core.forgive_value(field.shape, capture.text, where=where)


# The intent written by name, without listing the 77 choices in the prompt:
# for a model that has learned them (the distilled student).
intent_by_name = lmcc.make_format(write=lambda v: str(v), read=_read_intent,
                                  describe=lambda: "<intent>", accepts=("enum",))


# Compact on purpose: at scale every token is paid for. The categories are
# written once (the enum placeholder), the reply is one short line.
ADAPTERS = {
    # For the trained student: no category list, ~9x fewer prompt tokens.
    "short": lmcc.adapter(name="classify_short", messages=[
        lmcc.system("{instruction}\nIntent: {answer}"),
        lmcc.user("{text}"),
    ], formats={"enum": intent_by_name}),
    "compact": lmcc.adapter(name="classify_compact", messages=[
        lmcc.system("{instruction}\n\nReply with one line:\nIntent: {answer}"),
        lmcc.user("{text}"),
    ]),
}
