"""Compare OpenRouter models that allow distillation on the same 200 banking77 questions.

Every request sets provider.enforce_distillable_text = true, so it only reaches
models whose authors allow training on their output (OpenRouter's flag).
Reasoning is turned off where the endpoint allows it, else set to the lowest
effort; the answer is constrained to the 77 intents by a JSON-schema enum when
the model supports structured outputs, else read from plain text.

    python or_compare.py MODEL [MODEL ...]
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
EFFORT = os.environ.get("EFFORT")          # e.g. xhigh / high: force a reasoning level
OUT = "/home/maxime/Projects/primeintellect/outputs/or-distillable" + (f"-{EFFORT}" if EFFORT else "")
URL = "https://openrouter.ai/api/v1/chat/completions"


def body(model, text, reasoning, structured):
    quick = reasoning in (None, {"enabled": False})          # None: model has no reasoning setting
    b = {"model": model, "max_tokens": 50 if quick else (32000 if EFFORT else 4000),
         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
         "provider": {"enforce_distillable_text": True, "require_parameters": True}}
    if reasoning is not None:
        b["reasoning"] = reasoning
    if quick:
        b["temperature"] = 0
    if structured:
        b["response_format"] = SCHEMA
    return b


def parse(content, structured):
    if content is None:
        return None
    if structured:
        try:
            return json.loads(content)["intent"]
        except Exception:
            pass
    s = content.strip().strip("`\"' .").split()[0] if content.strip() else ""
    if s in LABELS:
        return s
    hits = [l for l in LABELS if l in content]
    return max(hits, key=len) if hits else content.strip()[:60]


async def pick_mode(c, model):
    """First (reasoning, structured) combination the model accepts."""
    modes = ({"effort": e} for e in (["xhigh", "high"] if EFFORT == "xhigh" else [EFFORT])) if EFFORT else \
            ({"enabled": False}, None, {"effort": "minimal"}, {"effort": "low"})
    for reasoning in modes:
        for structured in (True, False):
            r = await c.post(URL, json=body(model, TEST[0]["text"], reasoning, structured))
            if r.status_code == 200 and r.json().get("choices"):
                return reasoning, structured
    return None, None     # structured None = nothing accepted


async def run(c, model):
    for attempt in range(4):                                  # a 429 during probing is transient
        reasoning, structured = await pick_mode(c, model)
        if structured is not None:
            break
        await asyncio.sleep(15)
    if structured is None:
        return {"model": model, "error": "no accepted mode"}
    sem = asyncio.Semaphore(int(os.environ.get("INFLIGHT", 16)))
    out = [None] * len(TEST)

    async def one(k):
        async with sem:
            for attempt in range(5):
                t0 = time.time()
                try:
                    r = await c.post(URL, json=body(model, TEST[k]["text"], reasoning, structured))
                    j = r.json()
                    if r.status_code == 200 and j.get("choices"):
                        u = j.get("usage") or {}
                        out[k] = {"text": TEST[k]["text"], "label": TEST[k]["label"],
                                  "pick": parse(j["choices"][0]["message"].get("content"), structured),
                                  "secs": time.time() - t0, "cost": u.get("cost") or 0,
                                  "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                                  "provider": j.get("provider")}
                        return
                except Exception:
                    pass
                await asyncio.sleep(2 ** attempt)

    t0 = time.time()
    await asyncio.gather(*(one(k) for k in range(len(TEST))))
    wall = time.time() - t0
    done = [r for r in out if r]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{model.replace('/', '_')}.jsonl", "w") as f:
        for r in done:
            f.write(json.dumps(r) + "\n")
    n = len(done)
    secs = sorted(r["secs"] for r in done)
    return {"model": model, "reasoning": json.dumps(reasoning) if reasoning else "none (no setting)", "structured": structured,
            "answered": n, "valid": sum(r["pick"] in LABELS for r in done),
            "accuracy": round(sum(r["pick"] == r["label"] for r in done) / len(TEST), 3),
            "mean_reasoning_tokens": round(sum(r["reasoning_tokens"] for r in done) / max(n, 1)),
            "median_s": round(secs[n // 2], 2) if n else None,
            "cost_per_1000": round(sum(r["cost"] for r in done) / max(n, 1) * 1000, 3),
            "providers": sorted({r["provider"] for r in done if r["provider"]}), "wall_s": round(wall, 1)}


async def main(models):
    key = os.environ["OPENROUTER_API_KEY"]
    async with httpx.AsyncClient(timeout=300, headers={"Authorization": f"Bearer {key}"}) as c:
        sem = asyncio.Semaphore(5)

        async def guarded(m):
            async with sem:
                res = await run(c, m)
                print(json.dumps(res), flush=True)
                return res
        results = await asyncio.gather(*(guarded(m) for m in models))
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
