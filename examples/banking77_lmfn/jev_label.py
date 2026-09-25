"""Label banking77 training questions with Jev's full 77-way distribution.

Same question as the zero-shot Jev eval (one `choice` over the 77 intent
names, the customer message as the state). Keeps the probability of every
intent, not just the pick: those distributions are the soft targets the
student is distilled on (full_distill.py).

    python jev_label.py IN.json OUT.jsonl [INFLIGHT]

IN.json: {"labels": [...77 names...], "train": [{"text", "label"}, ...]}
Run with an lm15 that has `judgments` (lm15-dev) and TYPESAFE_API_KEY set.
"""

import asyncio
import json
import sys
import time

from lm15 import AsyncLMRouter, Config, Message, Request, choice, judgments

QUESTION = "Classify the bank customer's message by what they need."


async def main(src: str, dst: str, inflight: int) -> None:
    data = json.load(open(src))
    labels, rows = data["labels"], data["train"]
    answers = judgments(intent=choice(QUESTION, {label: None for label in labels}))
    cfg = Config(response_format=answers, probabilities="required")
    sem = asyncio.Semaphore(inflight)
    out: list[dict | None] = [None] * len(rows)
    retries = 0

    async with AsyncLMRouter() as router:
        async def one(i: int) -> None:
            nonlocal retries
            async with sem:
                for attempt in range(8):
                    try:
                        r = await router.complete(Request(model="jev-latest",
                                                          messages=[Message.user(rows[i]["text"])],
                                                          config=cfg))
                        p = r.probabilities["intent"]
                        out[i] = {**rows[i], "probs": [float(p[label]) for label in labels],
                                  "pred": r.data["intent"]}
                        return
                    except Exception as e:          # 429 / transient: back off and retry
                        retries += 1
                        if attempt == 7:
                            print("gave up:", i, repr(e)[:200], file=sys.stderr)
                            return
                        await asyncio.sleep(0.5 * 2 ** attempt)

        t0 = time.time()
        await asyncio.gather(*(one(i) for i in range(len(rows))))
        wall = time.time() - t0

    done = [r for r in out if r is not None]
    with open(dst, "w") as f:
        for r in done:
            f.write(json.dumps(r) + "\n")
    agree = sum(r["pred"] == r["label"] for r in done) / len(done)
    print(json.dumps({"rows": len(rows), "labelled": len(done), "retries": retries,
                      "seconds": round(wall, 1), "rows_per_s": round(len(done) / wall),
                      "jev_matches_human_label": round(agree, 3)}))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 128))
