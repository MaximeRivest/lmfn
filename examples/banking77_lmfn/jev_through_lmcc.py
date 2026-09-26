"""Jev on the 200 test questions, with lmcc writing the request and reading the reply.

A  "same question": the request of the 2026-09-24 zero-shot run (jev_label.py:
   the customer message alone as the state, the instruction as the judgment's
   question), now rendered by an lmcc adapter and read by lmcc, probabilities
   included. It should reproduce jev_test200_predictions.json.
B  "lmfn default": classify.using(model="jev-latest"), what lmfn does with no
   adapter given (lmfn.judgment_adapter: no system prompt, which lm15 refuses
   for Jev; the input alone as the state; the docstring as the question).

Measured 2026-09-26 (lmcc 0.8.3, lm15 1.0.1), 200 test questions:
   A  accuracy 78.5% (Sept 24: 78.5%), same pick 99%, probabilities 200/200
   B  accuracy 78.5%, same pick 99%, probabilities 200/200, median diff 0.00
   Control, same day: Jev against itself agrees on 195/200 picks (max
   probability difference 0.15); direct lm15 against the lmcc path 197/200
   (median 0.00, max 0.16). The differences are Jev's, not lmcc's.

    python examples/banking77_lmfn/jev_through_lmcc.py [A|B ...]
Needs TYPESAFE_API_KEY. About 200 x 1,000 input tokens per mode ($0.01).
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import lm15
import lmcc
import lmcc_lm15
import lmcc_std

lmcc_std.install()                             # the json_object reader

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = json.load(open(os.path.join(HERE, "jev_test200_predictions.json")))
LABELS = list(TEST[0]["probs"])              # the 77 keys, as the earlier run declared them
QUESTION = "Classify the bank customer's message by what they need."   # jev_label.py
router = lm15.LMRouter()


def plan_a() -> lmcc.Plan:
    sig = lmcc.signature(QUESTION, inputs={"text": str},
                         outputs={"intent": lmcc.field({"enum": LABELS}, desc=QUESTION)})
    adapter = lmcc.adapter(name="jev_same_question", messages=[lmcc.user("{text}")],
                           reader={"kind": "json_object", "probabilities": "required"})
    return adapter.bind(sig, {"native_structured_output": True})


def run_a(row):
    plan = PLAN_A
    request = lmcc_lm15.request(plan.render(text=row["text"]), model="jev-latest")
    reading = lmcc_lm15.read(plan, router.complete(request))
    return reading.values["intent"], reading.probabilities["intent"], reading.measured_by["intent"]


def run_b(row):
    from banking77_lmfn.program import classify
    result = classify.using(model="jev-latest").call(row["text"])
    return result.value, result.probabilities.get("answer", {}), result.measured_by.get("answer")


def with_retries(fn, row):
    for attempt in range(6):
        try:
            return fn(row)
        except lm15.errors.RateLimitError:
            time.sleep(0.5 * 2 ** attempt)
    return fn(row)


def evaluate(mode: str) -> dict:
    fn = {"A": run_a, "B": run_b}[mode]
    t0 = time.time()
    with ThreadPoolExecutor(32) as pool:
        out = list(pool.map(lambda r: with_retries(fn, r), TEST))
    wall = time.time() - t0
    n = len(TEST)
    same_pick = sum(p == r["pred"] for (p, _, _), r in zip(out, TEST))
    correct = sum(p == r["label"] for (p, _, _), r in zip(out, TEST))
    earlier = sum(r["pred"] == r["label"] for r in TEST)
    diffs = [max(abs(probs.get(k, 0.0) - r["probs"][k]) for k in LABELS)
             for (_, probs, _), r in zip(out, TEST) if probs]
    return {"mode": mode, "rows": n, "accuracy": correct / n, "earlier_accuracy": earlier / n,
            "same_pick_as_earlier": same_pick / n,
            "rows_with_probabilities": len(diffs),
            "max_prob_diff_vs_earlier": max(diffs) if diffs else None,
            "median_prob_diff_vs_earlier": sorted(diffs)[len(diffs) // 2] if diffs else None,
            "measured_by": sorted({m for _, _, m in out if m}), "seconds": round(wall, 1)}


if __name__ == "__main__":
    PLAN_A = plan_a()
    for mode in sys.argv[1:] or ["A", "B"]:
        print(json.dumps(evaluate(mode)), flush=True)
