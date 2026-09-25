"""Distill Jev's 77-way distributions into the ORIGINAL Qwen3.5-0.8B, whole model trained.

The 248k-word output layer is not used: the hidden state of the last prompt
token (after "Intent:") goes into a new 77-way linear head. Every weight of
the backbone and the head is trained (no freezing, no LoRA), from the base
checkpoint, not the earlier distilled student.

Loss: KL(Jev || student) over the 77 intents, i.e. soft cross-entropy on
Jev's full distribution (Jev reports probabilities rounded to 0.01; each row
is renormalized to sum to 1). Matching a known target distribution is a
supervised objective: an RL reward for "be close to this distribution" would
optimize the same quantity through sampled picks, with far noisier gradients.

Same prompt and the same 200 test questions as every earlier banking77 run.
Every phase is timed.

    python full_distill.py VARIANT      # VARIANT = sub1894 | all
"""

import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

T_START = time.time()
BASE = "Qwen/Qwen3.5-0.8B"
JEV_TRAIN = "/home/maxime/Projects/primeintellect/data/banking77-jev/train.jsonl"
SUBSET = "/home/maxime/Projects/primeintellect/data/banking77-jev/questions.json"  # labels, train rows, DeepSeek's 1,894 texts
TEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_test200_predictions.json")
SYSTEM = "Classify the bank customer's message by what they need.\nIntent: <intent>"
DEVICE = "cuda:0"                              # pick the card with CUDA_VISIBLE_DEVICES
EPOCHS, BATCH, LR_BODY, LR_HEAD, WARMUP = 3, 32, 3e-5, 3e-4, 0.05
VAL_ROWS, SEED = 500, 0

variant = sys.argv[1]
OUT = f"/home/maxime/Projects/primeintellect/outputs/banking77-full-jev-{variant}"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
timings = {}


def tick(name, t0):
    timings[name] = round(time.time() - t0, 1)
    print(f"[{name}] {timings[name]} s", flush=True)


# ---- data -------------------------------------------------------------------
t0 = time.time()
LABELS = json.load(open(SUBSET))["labels"]
rows = [json.loads(line) for line in open(JEV_TRAIN)]
subset = set(json.load(open(SUBSET))["deepseek_subset"])
rng = random.Random(SEED)
pool = [r for r in rows if r["text"] not in subset]
rng.shuffle(pool)
val = pool[:VAL_ROWS]                           # held out from every variant
val_texts = {r["text"] for r in val}
train = [r for r in rows if r["text"] in subset] if variant == "sub1894" else \
        [r for r in rows if r["text"] not in val_texts]
test = json.load(open(TEST))

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"                       # last position = the "Intent:" token for every row


def prompt(text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "Intent:"


def targets(rs):
    p = torch.tensor([[r["probs"][l] for l in LABELS] if isinstance(r["probs"], dict) else r["probs"]
                      for r in rs], dtype=torch.float32)
    return p / p.sum(1, keepdim=True)


def encode(rs):
    return [tok(prompt(r["text"]))["input_ids"] for r in rs]


train_ids, val_ids, test_ids = encode(train), encode(val), encode(test)
train_p, val_p = targets(train), targets(val)
test_jev = targets(test)
test_gold = torch.tensor([LABELS.index(r["label"]) for r in test])
tick("data_and_tokenize", t0)

# ---- model: original weights, new 77-way head, everything trainable ----------
t0 = time.time()
lm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(DEVICE)
body = lm.model
head = torch.nn.Linear(lm.config.hidden_size, len(LABELS), device=DEVICE)
with torch.no_grad():                           # start from the base model's own output rows
    first = [tok.encode(" " + l)[0] for l in LABELS]   # for each intent's first token
    head.weight.copy_(lm.lm_head.weight[first])
    head.bias.zero_()
lm.lm_head = torch.nn.Identity()                # the 248k-word output layer is gone from the graph
body.train()
n_body = sum(p.numel() for p in body.parameters())
tick("load_model", t0)


def batches(ids, n, shuffle):
    order = list(range(n))
    if shuffle:
        random.shuffle(order)
    for i in range(0, n, BATCH):
        b = order[i:i + BATCH]
        enc = tok.pad({"input_ids": [ids[j] for j in b]}, return_tensors="pt").to(DEVICE)
        yield b, enc


def logits_of(enc):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = body(**enc).last_hidden_state[:, -1, :]
    return head(h.float())


@torch.no_grad()
def predict(ids):
    body.eval()
    out = torch.cat([logits_of(enc) for _, enc in batches(ids, len(ids), False)])
    body.train()
    return out.cpu()


def kl(logits, p):
    return float(F.kl_div(F.log_softmax(logits, 1), p, reduction="batchmean"))


# ---- train ------------------------------------------------------------------
opt = torch.optim.AdamW([{"params": body.parameters(), "lr": LR_BODY},
                         {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
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
        torch.nn.utils.clip_grad_norm_(list(body.parameters()) + list(head.parameters()), 1.0)
        opt.step(); sched.step()
        run += loss.item()
    torch.cuda.synchronize()
    ep_s = time.time() - t0
    vk = kl(predict(val_ids), val_p)
    history.append({"epoch": epoch + 1, "train_kl": round(run / k, 4), "val_kl": round(vk, 4),
                    "seconds": round(ep_s, 1)})
    print(history[-1], flush=True)
timings["train_total"] = round(time.time() - t_train, 1)
timings["train_steps"] = steps
peak = torch.cuda.max_memory_allocated() / 1e9

# ---- evaluate on the 200 test questions -------------------------------------
t0 = time.time()
logits = predict(test_ids)
probs = F.softmax(logits, 1)
conf, pred = probs.max(1)
jev_pick = test_jev.argmax(1)


def ece(probs, gold, bins=15):
    conf, pred = probs.max(1)
    right = (pred == gold).float()
    e = 0.0
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += float(m.float().mean() * (conf[m].mean() - right[m].mean()).abs())
    return e


order_c = conf.argsort(descending=True)
keep = {f"acc_top_{int(f * 100)}pct_confident":
        round(float((pred[order_c[:int(f * 200)]] == test_gold[order_c[:int(f * 200)]]).float().mean()), 3)
        for f in (0.8, 0.5)}
results = {
    "variant": variant, "train_rows": len(train), "val_rows": len(val), "epochs": EPOCHS,
    "trainable_params_M": round((n_body + sum(p.numel() for p in head.parameters())) / 1e6),
    "accuracy_vs_human": round(float((pred == test_gold).float().mean()), 3),
    "top3_vs_human": round(float((probs.topk(3, 1).indices == test_gold[:, None]).any(1).float().mean()), 3),
    "agrees_with_jev_pick": round(float((pred == jev_pick).float().mean()), 3),
    "jev_accuracy_same_200": round(float((jev_pick == test_gold).float().mean()), 3),
    "kl_to_jev_test": round(kl(logits, test_jev), 4),
    "mean_confidence": round(float(conf.mean()), 3),
    "ece": round(ece(probs, test_gold), 3), "jev_ece_same_200": round(ece(test_jev, test_gold), 3),
    **keep, "history": history, "peak_gpu_memory_gb": round(peak, 1),
}
tick("evaluate_200", t0)

t0 = time.time()
torch.save({"body": {k: v.to(torch.bfloat16) for k, v in body.state_dict().items()},
            "head": head.state_dict(), "labels": LABELS, "system": SYSTEM, "base": BASE},
           f"{OUT}/model.pt")
tick("save_bf16", t0)
timings["total_wall"] = round(time.time() - T_START, 1)
results["timings_s"] = timings
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print(json.dumps(results, indent=1))
