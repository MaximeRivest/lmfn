"""Distill Jev's 77-way distributions into ModernBERT (an encoder), whole model trained.

Same data, split, loss and test as full_distill.py (the Qwen3.5-0.8B run), so
the numbers compare directly: KL(Jev || student) on Jev's full distribution,
500 held-out train rows for validation, the same 200 test questions. The input
is the customer message alone (an encoder needs no instruction). After
training, throughput and one-row latency are measured in plain PyTorch.

    python modernbert_distill.py base|large|HF_MODEL_ID [EPOCHS] [LR]

Any ModernBERT-architecture encoder works, e.g. jhu-clsp/ettin-encoder-17m.
"""

import json
import math
import os
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

T_START = time.time()
size = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BASE = size if "/" in size else f"answerdotai/ModernBERT-{size}"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/maxime/Projects/primeintellect/data/banking77-jev"
TEST = os.path.join(HERE, "jev_test200_predictions.json")
BATCH, WARMUP, MAXLEN, VAL_ROWS, SEED = 32, 0.05, 128, 500, 0
LR = float(sys.argv[3]) if len(sys.argv) > 3 else {"base": 5e-5, "large": 3e-5}.get(size, 1e-4)
OUT = f"/home/maxime/Projects/primeintellect/outputs/banking77-{BASE.split('/')[-1].lower()}-e{EPOCHS}"
DEVICE = "cuda:0"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
timings = {}


def tick(name, t0):
    timings[name] = round(time.time() - t0, 1)
    print(f"[{name}] {timings[name]} s", flush=True)


# ---- data: identical split to full_distill.py ---------------------------------
t0 = time.time()
q = json.load(open(f"{DATA}/questions.json"))
LABELS = q["labels"]
rows = [json.loads(line) for line in open(f"{DATA}/train.jsonl")]
subset = set(q["deepseek_subset"])
rng = random.Random(SEED)
pool = [r for r in rows if r["text"] not in subset]
rng.shuffle(pool)
val = pool[:VAL_ROWS]
val_texts = {r["text"] for r in val}
train = [r for r in rows if r["text"] not in val_texts]
test = json.load(open(TEST))


def targets(rs):
    p = torch.tensor([[r["probs"][l] for l in LABELS] if isinstance(r["probs"], dict) else r["probs"]
                      for r in rs], dtype=torch.float32)
    return p / p.sum(1, keepdim=True)


tok = AutoTokenizer.from_pretrained(BASE)
enc_ids = lambda rs: [tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"] for r in rs]
train_ids, val_ids, test_ids = enc_ids(train), enc_ids(val), enc_ids(test)
train_p, val_p, test_jev = targets(train), targets(val), targets(test)
test_gold = torch.tensor([LABELS.index(r["label"]) for r in test])
tick("data_and_tokenize", t0)

# ---- model ------------------------------------------------------------------
t0 = time.time()
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), attn_implementation="flash_attention_2", dtype=torch.float32).to(DEVICE)
    attn = "flash_attention_2"
except Exception as e:                       # no flash-attn: PyTorch's own attention
    print("flash_attention_2 unavailable:", repr(e)[:150])
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), attn_implementation="sdpa", dtype=torch.float32).to(DEVICE)
    attn = "sdpa"
model.train()
n_params = sum(p.numel() for p in model.parameters())
tick("load_model", t0)


def batches(ids, n, shuffle, bs=BATCH):
    order = list(range(n))
    if shuffle:
        random.shuffle(order)
    for i in range(0, n, bs):
        b = order[i:i + bs]
        yield b, tok.pad({"input_ids": [ids[j] for j in b]}, return_tensors="pt").to(DEVICE)


def logits_of(enc):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**enc).logits.float()


@torch.no_grad()
def predict(ids):
    model.eval()
    out = torch.cat([logits_of(e) for _, e in batches(ids, len(ids), False, 128)])
    model.train()
    return out.cpu()


def kl(logits, p):
    return float(F.kl_div(F.log_softmax(logits, 1), p, reduction="batchmean"))


# ---- train ------------------------------------------------------------------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(train) / BATCH)
warm = max(1, int(WARMUP * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
random.seed(SEED)
torch.cuda.reset_peak_memory_stats()
history = []
t_train = time.time()
for epoch in range(EPOCHS):
    t0 = time.time()
    run = 0.0
    for k, (b, enc) in enumerate(batches(train_ids, len(train_ids), True), 1):
        loss = F.kl_div(F.log_softmax(logits_of(enc), 1), train_p[b].to(DEVICE), reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item()
    torch.cuda.synchronize()
    ep_s = time.time() - t0
    history.append({"epoch": epoch + 1, "train_kl": round(run / k, 4),
                    "val_kl": round(kl(predict(val_ids), val_p), 4), "seconds": round(ep_s, 1)})
    print(history[-1], flush=True)
timings["train_total"] = round(time.time() - t_train, 1)
timings["train_steps"] = steps
peak = torch.cuda.max_memory_allocated() / 1e9

# ---- evaluate: the same 200 test questions ------------------------------------
t0 = time.time()
logits = predict(test_ids)
probs = F.softmax(logits, 1)
conf, pred = probs.max(1)


def ece(probs, gold, bins=15):
    conf, pred = probs.max(1)
    right = (pred == gold).float()
    e = 0.0
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += float(m.float().mean() * (conf[m].mean() - right[m].mean()).abs())
    return e


oc = conf.argsort(descending=True)
results = {
    "model": BASE, "lr": LR, "attention": attn, "epochs": EPOCHS, "train_rows": len(train),
    "params_M": round(n_params / 1e6),
    "accuracy_vs_human": round(float((pred == test_gold).float().mean()), 3),
    "top3_vs_human": round(float((probs.topk(3, 1).indices == test_gold[:, None]).any(1).float().mean()), 3),
    "agrees_with_jev_pick": round(float((pred == test_jev.argmax(1)).float().mean()), 3),
    "kl_to_jev_test": round(kl(logits, test_jev), 4),
    "mean_confidence": round(float(conf.mean()), 3), "ece": round(ece(probs, test_gold), 3),
    **{f"acc_top_{int(f * 100)}pct_confident": round(float((pred[oc[:int(f * 200)]] == test_gold[oc[:int(f * 200)]]).float().mean()), 3)
       for f in (0.8, 0.5)},
    "history": history, "peak_train_gpu_memory_gb": round(peak, 1),
}
tick("evaluate_200", t0)

t0 = time.time()
model.to(torch.bfloat16).save_pretrained(f"{OUT}/model")
tok.save_pretrained(f"{OUT}/model")
json.dump(LABELS, open(f"{OUT}/model/labels.json", "w"))
tick("save_bf16", t0)
timings["total_wall_to_saved_model"] = round(time.time() - T_START, 1)

# ---- serving speed, plain PyTorch, bf16 --------------------------------------
model.eval()
texts = [r["text"] for r in rows] * 2                  # 19,986 real banking77 messages
t0 = time.time()
ids = [tok(t, truncation=True, max_length=MAXLEN)["input_ids"] for t in texts]
speed = {"tokenize_rows_per_s": round(len(ids) / (time.time() - t0))}
order = sorted(range(len(ids)), key=lambda i: len(ids[i]))   # a batch job sorts by length
with torch.no_grad():
    for bs in (256, 1024):
        for rep in range(2):                                 # first pass warms up kernels
            torch.cuda.synchronize(); t0 = time.time()
            for i in range(0, len(ids), bs):
                e = tok.pad({"input_ids": [ids[j] for j in order[i:i + bs]]}, return_tensors="pt").to(DEVICE)
                model(**e).logits.float().softmax(1).max(1)
            torch.cuda.synchronize()
        speed[f"rows_per_s_batch{bs}"] = round(len(ids) / (time.time() - t0))
    lat = []
    for t in [r["text"] for r in test]:
        t0 = time.time()
        e = tok(t, return_tensors="pt").to(DEVICE)
        model(**e).logits.float().softmax(1).max(1)
        torch.cuda.synchronize()
        lat.append((time.time() - t0) * 1000)
    lat = sorted(lat[20:])
    speed["one_row_latency_ms_p50"] = round(statistics.median(lat), 1)
    speed["one_row_latency_ms_p90"] = round(lat[int(0.9 * len(lat))], 1)
best = max(v for k, v in speed.items() if k.startswith("rows_per_s_batch"))
speed["hours_per_100M_rows_one_gpu"] = round(1e8 / best / 3600, 1)
results["timings_s"] = timings
results["serving_pytorch_bf16"] = speed
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=1))
