"""Offline proof that the banking77 students were trained on what lmcc says.

For each training set: every prompt the model was trained on equals, byte
for byte, what lmcc renders today from the adapter, and every recorded
teacher answer reads back through lmcc to a valid intent. Also: the saved
student/adapter.json is exactly ADAPTERS["short"] and student/signature.json
exactly the lmfn function's signature. No network, no model.

    python examples/banking77_lmfn/check_prompts.py
"""

import json
import sys

import polars as pl

import lmcc
import lmfn
import student_prompt
from banking77_lmfn.program import ADAPTERS, LABELS, classify
from student_prompt import messages as student_messages

DATA = "/home/maxime/Projects/primeintellect/data"
SETS = {"banking77-teacher": "compact", "banking77-teacher-short": "short"}


def rendered(adapter, text):
    with lmfn.configure(model="gpt-4.1-mini"):          # the name only picks capabilities
        r = classify.with_adapter(adapter).render(text)
    return [{"role": "system", "content": r.system}] + [
        {"role": m.role, "content": m.parts[0].text} for m in r.messages]


def main() -> int:
    failures = 0
    if json.load(open(student_prompt.ADAPTER)) != ADAPTERS["short"].dump():
        print("student/adapter.json differs from ADAPTERS['short']: the student's layout moved")
        failures += 1
    if json.load(open(student_prompt.SIGNATURE)) != lmcc.signature_to_dict(classify.signature):
        print("student/signature.json differs from classify's signature: the task moved")
        failures += 1
    for name, key in SETS.items():
        rows = pl.read_parquet(f"{DATA}/{name}/train.parquet").to_dicts()
        adapter = ADAPTERS[key]
        with lmfn.configure(model="gpt-4.1-mini"):
            plan = classify.with_adapter(adapter).plan()
        same = read = repaired = 0
        for row in rows:
            text = row["prompt"][-1]["content"]
            got = student_messages(text) if key == "short" else rendered(adapter, text)
            same += got == row["prompt"]
            reading = plan.read(row["completion"][0]["content"])
            read += reading.values["answer"] in LABELS
            repaired += bool(reading.repairs)
        ok = same == read == len(rows)
        failures += not ok
        print(f"{name:26} {len(rows)} rows · prompt identical {same} · answer read {read} "
              f"(repaired {repaired}) · {'OK' if ok else 'MISMATCH'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
