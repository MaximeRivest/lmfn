"""Does training on Jev's full distribution beat training on Jev's top pick alone?

One controlled change: the target is either Jev's 77-way distribution ("soft")
or a one-hot on Jev's top choice ("hard"). Same encoder, data, split, loss
(KL; with a one-hot target it is plain cross-entropy), optimizer, passes and
seed. Several seeds per setting, so seed noise can be told apart from the
effect.

Evaluated on the FULL banking77 test set (3,076 questions, Jev-labelled too),
not just the 200-question subset: ±0.75 points standard error instead of ±2.8.
Confidence quality is reported raw and after temperature scaling on the 500
validation rows against their human labels (the standard fix for an
overconfident classifier, applied the same way to both).

    python soft_vs_hard.py MODEL_ID EPOCHS LR TARGET SEED ROWS
      TARGET = soft | hard | human | synth | human_synth     ROWS = all | sub1894 | N
"""

import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL, EPOCHS, LR, TARGET, SEED, ROWS = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
DATA = "/home/maxime/Projects/primeintellect/data/banking77-jev"
OUT = "/home/maxime/Projects/primeintellect/outputs/soft-vs-hard"
BATCH, WARMUP, MAXLEN, VAL_ROWS, SPLIT_SEED = 32, 0.05, 128, 500, 0
DEVICE = "cuda:0"
T0 = time.time()

q = json.load(open(f"{DATA}/questions.json"))
LABELS = q["labels"]
rows = [json.loads(line) for line in open(f"{DATA}/train.jsonl")]
test = [json.loads(line) for line in open(f"{DATA}/test.jsonl")]
subset = set(q["deepseek_subset"])
pool = [r for r in rows if r["text"] not in subset]
random.Random(SPLIT_SEED).shuffle(pool)               # the split never changes with SEED
val = pool[:VAL_ROWS]
val_texts = {r["text"] for r in val}
train = [r for r in rows if r["text"] in subset] if ROWS == "sub1894" else [r for r in rows if r["text"] not in val_texts]
SYNTH = {}
if TARGET in ("synth", "human_synth"):               # our distillable pipeline's labels (no teacher we may not train on)
    SYNTH = {json.loads(l)["text"]: json.loads(l)["probs"] for l in open(
        "/home/maxime/Projects/primeintellect/data/pipeline/b77_2000.jsonl")}
    train = [r for r in train if r["text"] in SYNTH]
if ROWS.isdigit():                                   # learning curve: random N rows, more passes when small
    random.Random(SEED).shuffle(train)
    train = train[:int(ROWS)]
    EPOCHS = min(60, max(EPOCHS, round(EPOCHS * 9493 / len(train))))


def jev(rs):
    p = torch.tensor([r["probs"] for r in rs], dtype=torch.float32)
    return p / p.sum(1, keepdim=True)


train_p = jev(train)
if TARGET == "hard":
    train_p = F.one_hot(train_p.argmax(1), len(LABELS)).float()
elif TARGET == "synth":
    train_p = torch.tensor([SYNTH[r["text"]] for r in train], dtype=torch.float32)
    train_p = train_p / train_p.sum(1, keepdim=True)
elif TARGET in ("human", "human_synth"):                        # no teacher: the dataset's own labels
    train_p = F.one_hot(torch.tensor([LABELS.index(r["label"]) for r in train]), len(LABELS)).float()
val_human = torch.tensor([LABELS.index(r["label"]) for r in val])
test_jev = jev(test)
test_gold = torch.tensor([LABELS.index(r["label"]) for r in test])

torch.manual_seed(SEED)
random.seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
ids = lambda rs: [tok(r["text"], truncation=True, max_length=MAXLEN)["input_ids"] for r in rs]
train_ids, val_ids, test_ids = ids(train), ids(val), ids(test)
ATTN = "sdpa" if "Qwen" in MODEL else "flash_attention_2"
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(MODEL, num_labels=len(LABELS))
if hasattr(cfg, "text_config"):                     # Qwen3.5 nests its text settings
    cfg = cfg.text_config; cfg.num_labels = len(LABELS)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, config=cfg, attn_implementation=ATTN, dtype=torch.float32).to(DEVICE)
assert model.config.num_labels == len(LABELS)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def batches(xs, shuffle, bs=BATCH):
    order = list(range(len(xs)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(xs), bs):
        b = order[i:i + bs]
        yield b, tok.pad({"input_ids": [xs[j] for j in b]}, return_tensors="pt").to(DEVICE)


def logits_of(enc):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**enc).logits.float()


@torch.no_grad()
def predict(xs):
    model.eval()
    out = torch.cat([logits_of(e) for _, e in batches(xs, False, 256)]).cpu()
    model.train()
    return out


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * math.ceil(len(train) / BATCH)
warm = max(1, int(WARMUP * steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
model.train()
t_train = time.time()
for epoch in range(EPOCHS):
    for b, enc in batches(train_ids, True):
        loss = F.kl_div(F.log_softmax(logits_of(enc), 1), train_p[b].to(DEVICE), reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
torch.cuda.synchronize()
train_s = time.time() - t_train

# temperature on the validation rows, against their human labels
vl = predict(val_ids)
T = torch.ones(1, requires_grad=True)
topt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)


def closure():
    topt.zero_grad(); l = F.cross_entropy(vl / T, val_human); l.backward(); return l


topt.step(closure)
temp = float(T.detach())

logits = predict(test_ids)


def ece(probs, gold, bins=15):
    conf, pred = probs.max(1)
    right = (pred == gold).float()
    e = 0.0
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += float(m.float().mean() * (conf[m].mean() - right[m].mean()).abs())
    return e


def metrics(lg):
    p = F.softmax(lg, 1)
    conf, pred = p.max(1)
    oc = conf.argsort(descending=True)
    n = len(pred)
    return {"ece": round(ece(p, test_gold), 4),
            "nll_human": round(float(F.cross_entropy(lg, test_gold)), 4),
            "mean_conf": round(float(conf.mean()), 4),
            "acc_top50_conf": round(float((pred[oc[:n // 2]] == test_gold[oc[:n // 2]]).float().mean()), 4),
            "acc_top80_conf": round(float((pred[oc[:int(n * .8)]] == test_gold[oc[:int(n * .8)]]).float().mean()), 4)}


p = F.softmax(logits, 1)
pred = p.argmax(1)
res = {"model": MODEL, "target": TARGET, "seed": SEED, "rows": ROWS, "train_rows": len(train), "epochs": EPOCHS, "lr": LR,
       "test_n": len(test),
       "acc": round(float((pred == test_gold).float().mean()), 4),
       "top3": round(float((p.topk(3, 1).indices == test_gold[:, None]).any(1).float().mean()), 4),
       "agree_jev": round(float((pred == test_jev.argmax(1)).float().mean()), 4),
       "kl_to_jev": round(float(F.kl_div(F.log_softmax(logits, 1), test_jev, reduction="batchmean")), 4),
       "raw": metrics(logits), "temp": round(temp, 3), "calibrated": metrics(logits / temp),
       "train_s": round(train_s, 1), "total_s": round(time.time() - T0, 1)}
os.makedirs(OUT, exist_ok=True)
name = f"{MODEL.split('/')[-1]}__{ROWS}__{TARGET}__s{SEED}"
json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
print(json.dumps(res))
