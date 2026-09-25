"""Replace the distilled student's 248k-word output layer with a 77-way head.

The student (Qwen3.5-0.8B, distilled on the short layout) reads the same
prompt as in production, ending with "Intent:"; the hidden state of that last
token goes into a linear layer with 77 outputs. One forward pass, no
generation, a probability for every intent.

Trained on the same 1,894 teacher labels as the student (a held-out 15% of
them calibrates the confidence), measured on the same 200 test questions.

    python classifier_head.py train     # -> OUT/head.pt, OUT/results.json
    python classifier_head.py bench     # throughput, batch sizes, bf16
"""

import json
import math
import os
import random
import sys
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = tuple(sorted(set(load_dataset("mteb/banking77", split="test")["label_text"])))   # as banking77_lmfn.program

STUDENT = "/home/maxime/Projects/primeintellect/outputs/banking77-sft-0.8b-short/hf_step_150"
DATA = "/home/maxime/Projects/primeintellect/data/banking77-teacher-short/train.parquet"
OUT = "/home/maxime/Projects/primeintellect/outputs/banking77-head"
SYSTEM = "Classify the bank customer's message by what they need.\nIntent: <intent>"
DEVICE = "cuda:0"
IDX = {label: i for i, label in enumerate(LABELS)}

tok = AutoTokenizer.from_pretrained(STUDENT)
tok.padding_side = "left"          # the last position is the "Intent:" token for every row


def prompt(text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "Intent:"


def load_backbone(dtype=torch.bfloat16):
    lm = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=dtype).to(DEVICE)
    lm.eval()
    body = lm.model                  # everything but lm_head
    return lm, body


@torch.no_grad()
def features(body, texts, batch=128):
    """The last token's hidden state for each prompt (the backbone is frozen)."""
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([prompt(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True).to(DEVICE)
        h = body(**enc).last_hidden_state[:, -1, :]
        out.append(h.float())
    return torch.cat(out)


def teacher_rows():
    df = pq.read_table(DATA).to_pylist()
    texts = [r["prompt"][-1]["content"] for r in df]
    labels = [r["completion"][0]["content"].removeprefix("Intent:").strip() for r in df]
    return texts, labels


def ece(probs, targets, bins=15):
    """Expected calibration error: how far confidence is from accuracy."""
    probs, targets = probs.cpu(), targets.cpu()
    conf, pred = probs.max(1)
    right = (pred == targets).float()
    err = torch.zeros(())
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            err += m.float().mean() * (conf[m].mean() - right[m].mean()).abs()
    return float(err)


def train():
    os.makedirs(OUT, exist_ok=True)
    lm, body = load_backbone()
    texts, labels = teacher_rows()
    order = list(range(len(texts)))
    random.Random(0).shuffle(order)
    cut = int(0.85 * len(order))
    tr, cal = order[:cut], order[cut:]
    y = torch.tensor([IDX[l] for l in labels], device=DEVICE)
    t0 = time.time()
    X = features(body, texts)
    print(f"features for {len(texts)} teacher rows in {time.time() - t0:.1f}s")

    # the head starts from the student's own output rows for each intent's
    # first token: it already "knows" which words follow "Intent:"
    head = torch.nn.Linear(X.shape[1], len(LABELS), device=DEVICE)
    with torch.no_grad():
        first = [tok.encode(" " + l)[0] for l in LABELS]
        head.weight.copy_(lm.lm_head.weight[first].float())
        head.bias.zero_()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    Xtr, ytr = X[tr], y[tr]
    for epoch in range(300):
        perm = torch.randperm(len(tr), device=DEVICE)
        for i in range(0, len(tr), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(head(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cal_logits = head(X[cal])
    # temperature scaling on the held-out teacher rows (calibration)
    T = torch.ones(1, device=DEVICE, requires_grad=True)
    topt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)

    def closure():
        topt.zero_grad()
        l = F.cross_entropy(cal_logits / T, y[cal]); l.backward(); return l
    topt.step(closure)
    temperature = float(T.detach())
    torch.save({"state": head.state_dict(), "temperature": temperature, "labels": LABELS}, f"{OUT}/head.pt")

    # evaluate on the same 200 test questions as every other run
    test = load_dataset("mteb/banking77", split="test").shuffle(seed=0).select(range(200))
    Xt = features(body, test["text"])
    gold = torch.tensor([IDX[l] for l in test["label_text"]], device=DEVICE)
    with torch.no_grad():
        probs = F.softmax(head(Xt) / temperature, dim=1)
        raw = F.softmax(head(Xt), dim=1)
    conf, pred = probs.max(1)
    acc = float((pred == gold).float().mean())
    top3 = float((probs.topk(3, 1).indices == gold[:, None]).any(1).float().mean())
    # "send the unsure rows to the teacher": accuracy of the confident share
    order_c = conf.argsort(descending=True)
    buckets = {}
    for keep in (1.0, 0.9, 0.8, 0.7, 0.5):
        k = order_c[: max(1, int(keep * len(order_c)))]
        buckets[f"top_{int(keep * 100)}pct_confident"] = round(float((pred[k] == gold[k]).float().mean()), 3)

    # agreement with the teacher on the test questions it answered
    teacher = {}
    import glob
    for line in open(glob.glob("/tmp/b77_teacher/**/traces.jsonl", recursive=True)[0]):
        r = json.loads(line)
        t = r["traces"][0]["info"].get("lmfn", {}).get("turns", []) if r["ok"] else []
        if t and "refusal" not in t[0]:
            teacher[r["task"]["data"]["info"]["inputs"]["text"]] = t[0]["outputs"]["answer"]
    agree = [LABELS[p] == teacher[x] for x, p in zip(test["text"], pred.tolist()) if x in teacher]
    results = {"accuracy": round(acc, 3), "top3_accuracy": round(top3, 3),
               "agrees_with_teacher": round(sum(agree) / len(agree), 3), "n_teacher": len(agree),
               "ece_calibrated": round(ece(probs, gold), 3), "ece_raw": round(ece(raw, gold), 3),
               "temperature": round(temperature, 3), "mean_confidence": round(float(conf.mean()), 3),
               **buckets}
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
    print(json.dumps(results, indent=1))


@torch.no_grad()
def bench():
    lm, body = load_backbone()
    ck = torch.load(f"{OUT}/head.pt")
    head = torch.nn.Linear(lm.config.get_text_config().hidden_size, len(LABELS)).to(DEVICE, torch.bfloat16)
    head.load_state_dict(ck["state"])
    texts = load_dataset("mteb/banking77", split="train")["text"]
    n = 20000
    rows = [texts[i % len(texts)] for i in range(n)]
    t0 = time.time()
    enc_all = [tok(prompt(t))["input_ids"] for t in rows]
    tok_rate = n / (time.time() - t0)
    out = {"client_tokenize_rows_per_s": round(tok_rate)}
    # sort by length so each batch pads little (what a batch job does)
    order = sorted(range(n), key=lambda i: len(enc_all[i]))
    for batch in (256, 512, 1024):
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(0, n, batch):
            ids = [enc_all[j] for j in order[i:i + batch]]
            enc = tok.pad({"input_ids": ids}, return_tensors="pt").to(DEVICE)
            h = body(**enc).last_hidden_state[:, -1, :]
            head(h).float().softmax(1).max(1)
        torch.cuda.synchronize()
        out[f"rows_per_s_batch{batch}"] = round(n / (time.time() - t0))
    best = max(v for k, v in out.items() if k.startswith("rows_per_s_batch"))
    out["hours_per_100M_rows_one_gpu"] = round(1e8 / best / 3600, 1)
    out["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/bench.json", "w"), indent=1)


if __name__ == "__main__":
    {"train": train, "bench": bench}[sys.argv[1]]()
