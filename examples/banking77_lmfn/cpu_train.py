"""Train Ettin-17M on banking77's human labels on a CPU (the 91.5% recipe), timed.

Recipe identical to `soft_vs_hard.py MODEL 6 1e-4 human SEED all` on the GPU:
9,493 rows (the 500 validation rows held out), batch 32, AdamW lr 1e-4,
weight decay 0.01, 5% warmup then cosine, grad clip 1.0, max 128 tokens,
6 passes, cross-entropy on the human label. Differences are only in how the
arithmetic runs: PyTorch SDPA attention instead of flash-attention, and the
precision chosen with --dtype.

    python cpu_train.py [--threads N] [--dtype fp32|bf16] [--compile]
                        [--group-length] [--bench-steps K] [--seed S]

--bench-steps K times K steps (after 5 warm-up steps) and stops: for choosing
settings. --group-length batches rows of similar length (less padding); it
changes which rows share a batch, so it is reported as a recipe variant.
"""

import argparse
import json
import math
import os
import random
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

T_START = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ettin-17m"))
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--compile", action="store_true")
ap.add_argument("--group-length", action="store_true")
ap.add_argument("--bench-steps", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--out", default="results")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
EPOCHS, BATCH, LR, WARMUP, MAXLEN, VAL_ROWS, SPLIT_SEED = a.epochs, 32, 1e-4, 0.05, 128, 500, 0
timings = {}


def tick(name, t0):
    timings[name] = round(time.time() - t0, 2)


# ---- data: same split as the GPU runs ----------------------------------------
t0 = time.time()
q = json.load(open(f"{a.data}/questions.json"))
LABELS = q["labels"]
rows = [json.loads(line) for line in open(f"{a.data}/train.jsonl")]
test = [json.loads(line) for line in open(f"{a.data}/test.jsonl")]
subset = set(q["deepseek_subset"])
pool = [r for r in rows if r["text"] not in subset]
random.Random(SPLIT_SEED).shuffle(pool)
val_texts = {r["text"] for r in pool[:VAL_ROWS]}
train = [r for r in rows if r["text"] not in val_texts]
y_train = torch.tensor([LABELS.index(r["label"]) for r in train])
y_test = torch.tensor([LABELS.index(r["label"]) for r in test])

torch.manual_seed(a.seed)
random.seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
enc = lambda rs: [tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"] for r in rs]
train_ids, test_ids = enc(train), enc(test)
tick("data_and_tokenize", t0)

t0 = time.time()
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(LABELS),
                                                           attn_implementation="sdpa", dtype=torch.float32)
model.train()
fwd = torch.compile(model) if a.compile else model
tick("load_model", t0)
ac = (lambda: torch.autocast("cpu", dtype=torch.bfloat16)) if a.dtype == "bf16" else (lambda: torch.autocast("cpu", enabled=False))


def epoch_batches():
    order = list(range(len(train_ids)))
    random.shuffle(order)
    if a.group_length:                         # sort within chunks of 50 batches, then shuffle the batches
        chunk = BATCH * 50
        order = [i for c in range(0, len(order), chunk)
                 for i in sorted(order[c:c + chunk], key=lambda j: len(train_ids[j]))]
        bs = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        random.shuffle(bs)
        return bs
    return [order[i:i + BATCH] for i in range(0, len(order), BATCH)]


def pad(idx, ids):
    return tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt")


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(train_ids) / BATCH)
warm = max(1, int(WARMUP * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))


def step(b):
    e = pad(b, train_ids)
    with ac():
        logits = fwd(**e).logits.float()
    loss = F.cross_entropy(logits, y_train[b])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    return loss.item()


cfg = {"threads": torch.get_num_threads(), "dtype": a.dtype, "compile": a.compile, "group_length": a.group_length,
       "cpu_capability": torch.backends.cpu.get_cpu_capability(), "torch": torch.__version__}

if a.bench_steps:
    bs = epoch_batches()
    for b in bs[:5]:
        step(b)                                 # warm-up (and compile)
    t0 = time.time()
    for b in bs[5:5 + a.bench_steps]:
        step(b)
    per = (time.time() - t0) / a.bench_steps
    print(json.dumps({**cfg, "sec_per_step": round(per, 4), "rows_per_s": round(BATCH / per, 1),
                      "projected_train_min": round(per * steps / 60, 1)}))
    raise SystemExit

t0 = time.time()
epoch_s = []
for ep in range(EPOCHS):
    te = time.time()
    losses = [step(b) for b in epoch_batches()]
    epoch_s.append(round(time.time() - te, 1))
    print(f"epoch {ep + 1}: loss {statistics.mean(losses):.4f}, {epoch_s[-1]} s", flush=True)
tick("train", t0)

t0 = time.time()
model.eval()
with torch.no_grad():
    order = sorted(range(len(test_ids)), key=lambda i: len(test_ids[i]))
    logits = torch.empty(len(test_ids), len(LABELS))
    for i in range(0, len(order), 256):
        idx = order[i:i + 256]
        logits[idx] = model(**pad(idx, test_ids)).logits.float()
tick("evaluate_3076", t0)
p = logits.softmax(1)
res = {**cfg, "seed": a.seed, "train_rows": len(train_ids), "steps": steps,
       "accuracy": round(float((p.argmax(1) == y_test).float().mean()), 4),
       "top3": round(float((p.topk(3, 1).indices == y_test[:, None]).any(1).float().mean()), 4),
       "epoch_seconds": epoch_s, "test_rows_per_s": round(len(test_ids) / timings["evaluate_3076"])}

t0 = time.time()
lat = []
with torch.no_grad():
    for r in test[:220]:
        ts = time.time()
        model(**tok(r["text"], return_tensors="pt")).logits.softmax(1).argmax()
        lat.append((time.time() - ts) * 1000)
lat = sorted(lat[20:])
res["one_message_ms_p50"] = round(statistics.median(lat), 2)
res["one_message_ms_p90"] = round(lat[int(0.9 * len(lat))], 2)
tick("latency_test", t0)

t0 = time.time()
model.save_pretrained(f"{a.out}/model")
tok.save_pretrained(f"{a.out}/model")
tick("save", t0)
res["timings_s"] = timings
res["total_wall_s"] = round(time.time() - T_START, 1)
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/cpu_run.json", "w"), indent=1)
print(json.dumps(res, indent=1))
