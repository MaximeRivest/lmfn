"""Throughput and quality of the 77-way classifier served by vLLM
(`--runner pooling --convert classify`), same prompts as the generating
student. Batched /classify requests, prompts tokenized once by the client.

    python bench_classifier.py [ROWS]
"""

import asyncio
import json
import sys
import time

import httpx
from datasets import load_dataset
from transformers import AutoTokenizer

URL = "http://127.0.0.1:8012"
STUDENT = "/home/maxime/Projects/primeintellect/outputs/banking77-sft-0.8b-short/hf_step_150"
SYSTEM = "Classify the bank customer's message by what they need.\nIntent: <intent>"
tok = AutoTokenizer.from_pretrained(STUDENT)
LABELS = tuple(sorted(set(load_dataset("mteb/banking77", split="test")["label_text"])))


def prompt(t):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Intent:"


async def classify(ids, batch, inflight):
    sem = asyncio.Semaphore(inflight)
    out = [None] * len(ids)
    async with httpx.AsyncClient(base_url=URL, timeout=600) as client:
        async def one(start):
            async with sem:
                for attempt in range(5):
                    try:
                        r = await client.post("/classify", json={"model": "classifier",
                                                                 "input": ids[start:start + batch]})
                        break
                    except httpx.TransportError:
                        if attempt == 4:
                            raise
                        await asyncio.sleep(0.2 * (attempt + 1))
                r.raise_for_status()
                for k, d in enumerate(r.json()["data"]):
                    out[start + k] = (d["label"], max(d["probs"]))
        t0 = time.time()
        await asyncio.gather(*(one(s) for s in range(0, len(ids), batch)))
        return out, time.time() - t0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    test = load_dataset("mteb/banking77", split="test").shuffle(seed=0).select(range(200))
    tids = [tok.encode(prompt(t)) for t in test["text"]]
    preds, _ = asyncio.run(classify(tids, 64, 4))
    acc = sum(p == g for (p, _), g in zip(preds, test["label_text"])) / len(preds)
    rows = load_dataset("mteb/banking77", split="train")["text"]
    bulk = [tok.encode(prompt(rows[i % len(rows)])) for i in range(n)]
    asyncio.run(classify(bulk[:2048], 256, 8))                     # warm up
    best = {}
    for batch, inflight in ((256, 8), (512, 8), (1024, 4), (1024, 8)):
        _, secs = asyncio.run(classify(bulk, batch, inflight))
        best[f"batch{batch}x{inflight}"] = round(n / secs)
        print(json.dumps({"batch": batch, "inflight": inflight, "rows_per_s": round(n / secs)}), flush=True)
    top = max(best.values())
    print(json.dumps({"accuracy_200_test": round(acc, 3), "rows": n, "best_rows_per_s": top,
                      "hours_per_100M_rows_one_gpu": round(1e8 / top / 3600, 1)}))


if __name__ == "__main__":
    main()
