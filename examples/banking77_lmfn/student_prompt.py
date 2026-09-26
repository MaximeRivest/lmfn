"""The prompt a distilled banking77 student sees, from its saved calling convention.

student/adapter.json (the layout) and student/signature.json (the task: its
instruction and the 77 intents) are lmcc artifacts, plain data, saved beside
the weights they were trained with. This module needs lmcc alone — no lmfn,
no lm15, no dataset — so the training and serving scripts can run in any
environment where lmcc (standard library only) is installed; no script keeps
its own copy of the prompt. check_prompts.py proves the files, the lmfn
program and the training data agree.
"""

import json
import os

import lmcc

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.join(HERE, "student", "adapter.json")
SIGNATURE = os.path.join(HERE, "student", "signature.json")
_plan = None


def plan() -> lmcc.Plan:
    """The student's plan: its adapter bound to its signature (data-only: an
    empty registry suffices)."""
    global _plan
    if _plan is None:
        adapter = lmcc.load(json.load(open(ADAPTER)), registry=lmcc.Registry())
        _plan = adapter.bind(lmcc.signature_from_dict(json.load(open(SIGNATURE))), {},
                             registry=lmcc.Registry())
    return _plan


def messages(text: str) -> list[dict]:
    """The chat messages for one customer message: system, then user."""
    r = plan().render(text=text)
    return [{"role": "system", "content": r.system}] + [
        {"role": m["role"], "content": "".join(p["text"] for p in m["parts"])} for m in r.messages]
