---
title: "Distill a big model into a small one — a data scientist's notebook"
rat:
  project: ..
  python:
    requires: ">=3.11"
    dependencies: ["-e .", "-e examples/banking77_lmfn", "polars"]
---

# Distill a big model into a small one

**The situation.** A table of 100,000 customer messages. You want each one
classified into one of 77 intents. A big model (DeepSeek V4.1 Flash, thinking)
does it well but costs money per row; you want a 0.8B model that does it for
almost nothing, at 100 million rows.

**The flow.** One typed function → the big model labels a sample → a small
model learns those answers (not the big model's thinking) → the small model
runs at scale. Same function, same prompt layout, from the first cell to
production.

## 1. The data

```python
import polars as pl
from datasets import load_dataset

df = pl.from_arrow(load_dataset("mteb/banking77", split="train").data.table)
df = df.select("text", "label_text").with_row_index("id")
print(df.shape)
df.sample(5, seed=1)
```

```output
(9993, 3)
shape: (5, 3)
┌──────┬─────────────────────────────────┬────────────────────────────────┐
│ id   ┆ text                            ┆ label_text                     │
│ ---  ┆ ---                             ┆ ---                            │
│ u32  ┆ str                             ┆ str                            │
╞══════╪═════════════════════════════════╪════════════════════════════════╡
│ 8107 ┆ Card delivery services? Where?  ┆ order_physical_card            │
│ 7463 ┆ So, I am in the middle of purc… ┆ failed_transfer                │
│ 1000 ┆ What are the currency types th… ┆ fiat_currency_support          │
│ 7456 ┆ What could possibly be the cau… ┆ failed_transfer                │
│ 1845 ┆ How much does a transfer cost?  ┆ top_up_by_bank_transfer_charge │
└──────┴─────────────────────────────────┴────────────────────────────────┘
```

## 2. The task is one function

The type is the whole specification: 77 allowed answers, nothing else.

```python
import lmfn
from banking77_lmfn.program import classify, ADAPTERS, LABELS

print(len(LABELS), "intents, e.g.", LABELS[:4])
print(classify.__doc__)
```

## 3. What the model is asked, exactly

The compact layout: the category list once, one line back. Every token is
paid for at scale, so the prompt is short and the answer is a few tokens.

```python
from lmfn_verifiers import EndpointRouter           # any OpenAI-compatible server
prime = EndpointRouter("https://api.pinference.ai/api/v1", "PRIME_API_KEY")   # Prime Inference
fn = classify.with_adapter(ADAPTERS["compact"])
teacher = fn.using(model="deepseek/deepseek-v4.1-flash", router=prime,
                   capabilities={"instruct": True, "stop_sequences": True})
request = teacher.render("I am still waiting on my card?")
print(request.system[:400], "…")
print("user:", request.messages[0].parts[0].text)
```

## 4. The big model labels a sample

Run through the lmfn harness (verifiers), with the teacher's reasoning
**dropped before recording**: it still thinks, and we still pay for it, but
what we keep is the answer only. Results of the run (2,000 random training
rows, done before this notebook):

```python
import glob, json

teacher_runs = sorted(glob.glob("/tmp/b77_label/**/traces.jsonl", recursive=True))
rows = [json.loads(l) for l in open(teacher_runs[-1])]
answered = [r for r in rows if r["ok"] and r["traces"][0]["info"].get("lmfn")]
agree = sum(r["traces"][0]["rewards"]["agrees"]["score"] for r in answered) / len(answered)
tok_in = sum(r["num_input_tokens"] for r in rows); tok_out = sum(r["num_output_tokens"] for r in rows)
print(f"rows sent {len(rows)}, answered {len(answered)} (the rest: provider 504s, retried in production)")
print(f"teacher agrees with the dataset's human label: {agree:.1%}")
print(f"cost: ${tok_in * 0.30e-6 + tok_out * 1.20e-6:.2f} for {len(rows)} rows "
      f"→ ${(tok_in * 0.30e-6 + tok_out * 1.20e-6) / len(rows) * 1e6:,.0f} per million rows")
```

One recorded teacher reply — the answer only, no thinking:

```python
t = answered[0]["traces"][0]
reply = [n["message"] for n in t["nodes"] if n["message"]["role"] == "assistant"][0]
print({k: v for k, v in reply.items() if v})
```

## 5. The training set

`lmfn_verifiers.sft_rows` turns those traces into prompt/completion rows:
the exact request the teacher answered, and its answer.

```python
import lmfn_verifiers as lv
sft = lv.sft_rows(teacher_runs[-1])
print(len(sft), "training rows")
print("completion:", sft[0]["completion"])
pl.DataFrame({"intent": [r["completion"][0]["content"] for r in sft]}) \
  .group_by("intent").len().sort("len", descending=True).head(5)
```

## 6. The small model learns them

prime-rl SFT, full fine-tune of Qwen3.5-0.8B on one RTX 3090 (7.4 GB),
150 steps of 16 examples. Live dashboard: **http://100.86.49.54:7788**
(run `banking77-sft-0.8b`).

```python
import re
log = "/home/maxime/Projects/primeintellect/outputs/banking77-sft-0.8b/logs/attempt_1/trainer.log"
steps = [(int(m[0]), float(m[1])) for m in re.findall(r"Step (\d+) \|.*?Loss ([\d.]+)", open(log).read())]
loss = pl.DataFrame(steps, schema=["step", "loss"], orient="row")
print(f"{len(steps)} steps logged; loss {steps[0][1]:.3f} → {steps[-1][1]:.3f}")
loss.with_columns((pl.col("step") // 25 * 25).alias("bucket")).group_by("bucket").agg(pl.col("loss").mean()).sort("bucket")
```

## 7. Before and after, on 200 questions neither model saw

Same function, same layout, same 200 test questions; the student answers
greedily in one line.

```python
def score(pattern):
    f = sorted(glob.glob(pattern, recursive=True))
    if not f:
        return None
    rs = [json.loads(l) for l in open(f[-1])]
    ok = [r["traces"][0] for r in rs if r["ok"]]
    return {"accuracy": sum((t["rewards"].get("agrees") or {}).get("score", 0) for t in ok) / len(ok),
            "readable": sum(t["metrics"].get("readable", 0) for t in ok) / len(ok), "rows": len(ok)}

results = pl.DataFrame([
    {"model": "DeepSeek V4.1 Flash (teacher, thinking)", **(score("/tmp/b77_teacher/**/traces.jsonl") or {})},
    {"model": "Qwen3.5-0.8B, untrained", **(score("/tmp/b77_base/**/traces.jsonl") or {})},
    {"model": "Qwen3.5-0.8B, distilled", **(score("/tmp/b77_student/**/traces.jsonl") or {})},
])
results
```

## 8. At scale

The distilled model is served by vLLM on one 3090; `lmfn` calls it like any
model. On the full table:

```python
import time, concurrent.futures as cf

local = EndpointRouter("http://127.0.0.1:8011/v1", "local")
student = fn.using(model="student", router=local, max_tokens=16, temperature=0,
                   capabilities={"instruct": True, "stop_sequences": True})
sample = df.sample(1000, seed=7)

def label(text):
    """One row: the value, or None when the reply is not one of the 77 intents
    (at scale a bad row is recorded, never a crash)."""
    try:
        return student(text)
    except lmcc.Refusal:
        return None

import lmcc
t0 = time.time()
with cf.ThreadPoolExecutor(32) as pool:
    labels = list(pool.map(label, sample["text"]))
secs = time.time() - t0
out = sample.with_columns(pl.Series("predicted", labels, dtype=pl.Utf8))
print(f"{len(out)} rows in {secs:.1f}s → {len(out) / secs:.0f} rows/s on one 3090 "
      f"→ 100M rows in {1e8 / (len(out) / secs) / 3600:.0f} h")
print(f"unreadable (not a valid intent): {out['predicted'].is_null().mean():.1%}")
print(f"agreement with the human labels: {(out['predicted'] == out['label_text']).fill_null(False).mean():.1%}")
out.select("text", "label_text", "predicted").head(8)
```
