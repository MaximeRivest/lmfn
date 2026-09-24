"""How fast can the distilled classifier run on one RTX 3090? Three ways to call
the same vLLM server (the fine-tuned Qwen3.5-0.8B, served as "student"):

  A  chat       what lmfn does today: chat request, the model writes "Intent: <label>"
  B  prefill    the prompt already ends with "Intent:", the model writes only the label
  C  prefill+choice   as B, and decoding is restricted to the 77 valid labels
  E  batched ids      as B, but prompts are tokenized once by the client and sent
                      64 per request as token ids (how a batch job should call it)
  D  short+choice     as C, with the category list removed from the prompt (the
                      distilled model has learned it; the 77 choices still bound it)

Accuracy on the 200 test questions the other runs used; throughput on 4,000
training messages, 256 requests in flight.

    python examples/banking77_lmfn/bench_serving.py
"""

import asyncio
import json
import time

import httpx
from datasets import load_dataset
from transformers import AutoTokenizer

import lmfn
from banking77_lmfn.program import ADAPTERS, LABELS, classify

import os

URL = "http://127.0.0.1:8011/v1"
# BENCH_ADAPTER=short for the student trained on the short layout (no category list)
fn = classify.with_adapter(ADAPTERS[os.environ.get("BENCH_ADAPTER", "compact")])
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")


def messages(text):
    with lmfn.configure(model="gpt-4.1-mini"):          # the name only picks capabilities
        r = fn.render(text)
    return [{"role": "system", "content": r.system}] + [
        {"role": m.role, "content": m.parts[0].text} for m in r.messages]


def short_messages(text):
    return [{"role": "system", "content": "Classify the bank customer's message by what they need.\n\n"
                                          "Reply with one line:\nIntent: <intent>"},
            {"role": "user", "content": text}]


def raw_prompt(text, short=False):
    """The chat template exactly as the student was trained on (thinking off),
    then the start of its answer."""
    p = tok.apply_chat_template(short_messages(text) if short else messages(text), tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return p + "Intent:"


def request(mode, text):
    if mode == "A":
        return "/chat/completions", {"model": "student", "messages": messages(text),
                                     "max_tokens": 32, "temperature": 0,
                                     "chat_template_kwargs": {"enable_thinking": False}}
    body = {"model": "student", "prompt": raw_prompt(text, short=mode == "D"), "max_tokens": 12,
            "temperature": 0, "stop": ["\n", "<|im_end|>"]}
    if mode in ("C", "D"):
        body["structured_outputs"] = {"choice": [" " + label for label in LABELS]}
    return "/completions", body


def label_of(mode, data):
    choice = data["choices"][0]
    text = choice["message"]["content"] if mode == "A" else choice["text"]
    text = text.strip()
    if mode == "A":
        text = text.removeprefix("Intent:").strip()
    return text, data.get("usage", {})


async def run(mode, texts, concurrency=256):
    sem = asyncio.Semaphore(concurrency)
    out = [None] * len(texts)
    usage = {"prompt": 0, "completion": 0}
    async with httpx.AsyncClient(base_url=URL, timeout=600,
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        async def one(i, text):
            async with sem:
                path, body = request(mode, text)
                for attempt in range(5):          # a dropped connection is retried, as in production
                    try:
                        resp = await client.post(path, json=body)
                        break
                    except httpx.TransportError:
                        if attempt == 4:
                            raise
                        await asyncio.sleep(0.2 * (attempt + 1))
                resp.raise_for_status()
                out[i], u = label_of(mode, resp.json())
                usage["prompt"] += u.get("prompt_tokens", 0)
                usage["completion"] += u.get("completion_tokens", 0)
        t0 = time.time()
        await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
        return out, time.time() - t0, usage


async def run_batched(ids, batch=int(os.environ.get("BENCH_BATCH", 64)), inflight=int(os.environ.get("BENCH_INFLIGHT", 16))):
    """Mode E: lists of token-id prompts per /completions request."""
    sem = asyncio.Semaphore(inflight)
    out = [None] * len(ids)
    usage = {"completion": 0}
    async with httpx.AsyncClient(base_url=URL, timeout=600) as client:
        async def one(start):
            chunk = ids[start:start + batch]
            async with sem:
                for attempt in range(5):
                    try:
                        resp = await client.post("/completions", json={
                            "model": "student", "prompt": chunk, "max_tokens": 12,
                            "temperature": 0, "stop": ["\n", "<|im_end|>"]})
                        break
                    except httpx.TransportError:
                        if attempt == 4:
                            raise
                        await asyncio.sleep(0.2 * (attempt + 1))
                resp.raise_for_status()
                data = resp.json()
                for c in data["choices"]:
                    out[start + c["index"]] = c["text"].strip()
                usage["completion"] += data["usage"]["completion_tokens"]
        t0 = time.time()
        await asyncio.gather(*(one(s) for s in range(0, len(ids), batch)))
        return out, time.time() - t0, usage


def main_batched(n):
    test = load_dataset("mteb/banking77", split="test").shuffle(seed=0).select(range(200))
    rows = load_dataset("mteb/banking77", split="train")["text"]
    bulk = [rows[i % len(rows)] for i in range(n)]
    ids = lambda texts: [tok.encode(raw_prompt(t)) for t in texts]      # noqa: E731
    asyncio.run(run_batched(ids(bulk[:256])))
    preds, _, _ = asyncio.run(run_batched(ids(test["text"])))
    acc = sum(p == t for p, t in zip(preds, test["label_text"])) / len(preds)
    t0 = time.time()
    bulk_ids = ids(bulk)
    tok_secs = time.time() - t0
    _, secs, usage = asyncio.run(run_batched(bulk_ids))
    rps = n / secs
    print(json.dumps({"mode": "E", "accuracy": round(acc, 3), "rows": n, "rows_per_s": round(rps),
                      "client_tokenize_rows_per_s": round(n / tok_secs),
                      "out_tokens_per_row": round(usage["completion"] / n, 1),
                      "hours_per_100M_rows": round(1e8 / rps / 3600, 1)}), flush=True)


def main():
    test = load_dataset("mteb/banking77", split="test").shuffle(seed=0).select(range(200))
    bulk = load_dataset("mteb/banking77", split="train").shuffle(seed=1).select(range(4000))["text"]
    results = []
    import sys
    if sys.argv[1:2] == ["E"]:
        return main_batched(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    for mode in sys.argv[1:] or ["A", "B", "C", "D"]:
        asyncio.run(run(mode, bulk[:64]))                          # warm up (prefix cache, graphs)
        preds, _, _ = asyncio.run(run(mode, test["text"]))
        acc = sum(p == t for p, t in zip(preds, test["label_text"])) / len(preds)
        valid = sum(p in LABELS for p in preds) / len(preds)
        _, secs, usage = asyncio.run(run(mode, bulk))
        rps = len(bulk) / secs
        results.append({"mode": mode, "accuracy": round(acc, 3), "valid_label": round(valid, 3),
                        "rows_per_s": round(rps), "out_tokens_per_row": round(usage["completion"] / len(bulk), 1),
                        "hours_per_100M_rows": round(1e8 / rps / 3600, 1)})
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
