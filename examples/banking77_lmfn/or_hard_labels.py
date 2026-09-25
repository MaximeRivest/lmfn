"""Hard-label eval of any OpenRouter model on the same 200 banking77 test questions.

Same instruction as or_logprob_targets.py; the answer is constrained to the 77
intent names with a JSON schema enum (structured outputs), so every answer is
a valid intent. No probabilities: accuracy and agreement only.

    python or_hard_labels.py MODEL off|low|medium|high [INFLIGHT]
"""

import asyncio
import json
import os
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = json.load(open("/home/maxime/Projects/primeintellect/data/banking77-jev/questions.json"))["labels"]
TEST = json.load(open(os.path.join(HERE, "jev_test200_predictions.json")))
SYSTEM = ("Classify the bank customer's message by what they need. Answer with exactly one intent name "
          "from this list and nothing else:\n" + "\n".join(LABELS))
SCHEMA = {"type": "json_schema", "json_schema": {"name": "intent", "strict": True, "schema": {
    "type": "object", "properties": {"intent": {"type": "string", "enum": LABELS}},
    "required": ["intent"], "additionalProperties": False}}}
OUT = "/home/maxime/Projects/primeintellect/outputs/or-hard"


async def main(model, effort, inflight):
    key = os.environ["OPENROUTER_API_KEY"]
    out = [None] * len(TEST)
    sem = asyncio.Semaphore(inflight)
    reasoning = {"enabled": False} if effort == "off" else {"effort": effort}
    async with httpx.AsyncClient(timeout=300, headers={"Authorization": f"Bearer {key}"}) as c:
        async def one(k):
            body = {"model": model, "max_tokens": 4000 if effort != "off" else 50, "response_format": SCHEMA,
                    "reasoning": reasoning, "provider": {"require_parameters": True},
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TEST[k]["text"]}]}
            if effort == "off":
                body["temperature"] = 0
            async with sem:
                for attempt in range(5):
                    t0 = time.time()
                    r = await c.post("https://openrouter.ai/api/v1/chat/completions", json=body)
                    try:
                        j = r.json(); msg = j["choices"][0]["message"]
                        pick = json.loads(msg["content"])["intent"]
                        u = j["usage"]
                        out[k] = {"text": TEST[k]["text"], "label": TEST[k]["label"], "pick": pick,
                                  "secs": time.time() - t0, "cost": u.get("cost"),
                                  "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}
                        return
                    except Exception:
                        await asyncio.sleep(2 ** attempt)
                print("failed", k, r.status_code, r.text[:300], file=sys.stderr)
        t0 = time.time()
        await asyncio.gather(*(one(k) for k in range(len(TEST))))
        wall = time.time() - t0
    done = [r for r in out if r]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{model.replace('/', '_')}__{effort}.jsonl", "w") as f:
        for r in done:
            f.write(json.dumps(r) + "\n")
    jev = {t["text"]: t["pred"] for t in TEST}
    n = len(done)
    secs = sorted(r["secs"] for r in done)
    print(json.dumps({"model": model, "reasoning": effort, "answered": f"{n}/{len(TEST)}",
                      "valid_intent": sum(r["pick"] in LABELS for r in done),
                      "accuracy_vs_human": round(sum(r["pick"] == r["label"] for r in done) / n, 3),
                      "agrees_with_jev_pick": round(sum(r["pick"] == jev[r["text"]] for r in done) / n, 3),
                      "mean_reasoning_tokens": round(sum(r["reasoning_tokens"] for r in done) / n),
                      "median_secs": round(secs[n // 2], 2), "p90_secs": round(secs[int(0.9 * n)], 2),
                      "wall_s": round(wall, 1), "cost_usd_200": round(sum(r["cost"] or 0 for r in done), 3),
                      "cost_per_1000_rows": round(sum(r["cost"] or 0 for r in done) / n * 1000, 2)}, indent=1))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16))
