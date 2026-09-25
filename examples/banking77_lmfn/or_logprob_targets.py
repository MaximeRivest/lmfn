"""Can an OpenRouter model's token logprobs serve as a 77-way soft target?

The model answers with one intent name (greedy, no reasoning). Names span
several tokens and providers return top-k alternatives only along the sampled
path, so the class distribution is rebuilt from that path: an alternative at
position i whose text (greedy prefix + alternative) is consistent with exactly
one intent gives that intent P(prefix) * P(alternative); the greedy path's own
mass goes to the answered intent. Mass on alternatives consistent with several
intents is split evenly among them ("shared"); mass consistent with none
(formatting, prose) is "lost". Coverage = assigned / total, reported per row.

    python or_logprob_targets.py MODEL PROVIDER|any TOP_K [N] [off|none]

Every request sets provider.enforce_distillable_text (only models whose authors
allow distillation). The last argument: "off" sends reasoning.enabled=false,
"none" sends no reasoning setting (models without one).

Writes outputs/or-logprobs/<model>__<provider>.jsonl and prints a summary
against the human labels and Jev, on the same 200 test questions.
"""

import asyncio
import json
import math
import os
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
Q = json.load(open("/home/maxime/Projects/primeintellect/data/banking77-jev/questions.json"))
LABELS = Q["labels"]
TEST = json.load(open(os.path.join(HERE, "jev_test200_predictions.json")))
SYSTEM = ("Classify the bank customer's message by what they need. Answer with exactly one intent name "
          "from this list and nothing else:\n" + "\n".join(LABELS))
OUT = "/home/maxime/Projects/primeintellect/outputs/or-logprobs"


def rebuild(tokens):
    """tokens: [{token, top_logprobs:[{token, logprob}]}] along the greedy path -> (dist, stats)."""
    dist = [0.0] * len(LABELS)
    shared = lost = 0.0
    prefix, p_prefix = "", 1.0
    for t in tokens:
        for alt in t["top_logprobs"]:
            if alt["token"] == t["token"]:
                continue                                    # the greedy branch is followed below
            p = p_prefix * math.exp(alt["logprob"])
            s = prefix + alt["token"]
            match = [i for i, l in enumerate(LABELS) if l.startswith(s) or (s.startswith(l) and not s[len(l):].strip("\n ").replace("<", "")[:1].isalnum())]
            if len(match) == 1:
                dist[match[0]] += p
            elif match:
                shared += p
                for i in match:
                    dist[i] += p / len(match)
            else:
                lost += p
        greedy = [a for a in t["top_logprobs"] if a["token"] == t["token"]]
        p_prefix *= math.exp(greedy[0]["logprob"]) if greedy else math.exp(t["logprob"])
        prefix += t["token"]
        done = [i for i, l in enumerate(LABELS) if prefix.strip().startswith(l)]
        live = [i for i, l in enumerate(LABELS) if l.startswith(prefix)]
        if len(live) <= 1:                                  # the path now names one intent (or none)
            i = (live or done or [None])[0]
            if i is None:
                lost += p_prefix
            else:
                dist[i] += p_prefix
            break
    else:
        lost += p_prefix
    return dist, {"shared": shared, "lost": lost, "coverage": sum(dist)}


async def main(model, provider, top_k, n):
    key = os.environ["OPENROUTER_API_KEY"]
    rows = TEST[:n]
    out = [None] * len(rows)
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {key}"}) as c:
        async def one(k):
            body = {"model": model, "max_tokens": 30, "temperature": 0, "logprobs": True, "top_logprobs": top_k,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rows[k]["text"]}],
                    "provider": {"enforce_distillable_text": True, "require_parameters": True,
                                 **({} if provider == "any" else {"order": [provider], "allow_fallbacks": False})}}
            if REASONING == "off":
                body["reasoning"] = {"enabled": False}
            async with sem:
                for attempt in range(5):
                    t0 = time.time()
                    r = await c.post("https://openrouter.ai/api/v1/chat/completions", json=body)
                    if r.status_code == 200 and r.json().get("choices"):
                        j = r.json(); ch = j["choices"][0]
                        lp = (ch.get("logprobs") or {}).get("content") or []
                        dist, st = rebuild(lp)
                        out[k] = {"provider": j.get("provider"), "text": rows[k]["text"], "label": rows[k]["label"], "answer": ch["message"].get("content"),
                                  "dist": dist, **st, "secs": time.time() - t0, "cost": j["usage"].get("cost"),
                                  "reasoning_tokens": (j["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                  "raw": lp}
                        return
                    await asyncio.sleep(2 ** attempt)
                print("failed", k, r.status_code, r.text[:200], file=sys.stderr)
        t0 = time.time()
        await asyncio.gather(*(one(k) for k in range(len(rows))))
        wall = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    done = [r for r in out if r]
    with open(f"{OUT}/{model.replace('/', '_')}__{provider}__distillable.jsonl", "w") as f:
        for r in done:
            f.write(json.dumps(r) + "\n")

    idx = {t["text"]: t for t in TEST}
    jev = lambda r: [idx[r["text"]]["probs"][l] for l in LABELS]
    acc = top3 = agree = 0
    kl = ece_rows = 0.0
    confs = []
    for r in done:
        d = r["dist"]; z = sum(d) or 1.0
        p = [x / z for x in d]                       # renormalized over assigned mass
        order = sorted(range(77), key=lambda i: -p[i])
        gold = LABELS.index(r["label"]); j = jev(r); jz = sum(j)
        acc += order[0] == gold; top3 += gold in order[:3]
        agree += order[0] == max(range(77), key=lambda i: j[i])
        kl += sum((a / jz) * math.log((a / jz) / max(b, 1e-4)) for a, b in zip(j, p) if a > 0)
        confs.append((p[order[0]], order[0] == gold))
    n_ = len(done)
    bins = [[] for _ in range(10)]
    for c_, ok in confs:
        bins[min(9, int(c_ * 10))].append((c_, ok))
    ece = sum(len(b) / n_ * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b)
    spread = sum(1 for c_, _ in confs if c_ < 0.95) / n_
    ranked = sorted(confs, key=lambda x: -x[0])
    keep = {f"acc_top{int(f*100)}": round(sum(o for _, o in ranked[:int(f * n_)]) / int(f * n_), 3) for f in (0.8, 0.5)}
    def kept_at(target):                      # largest share kept with accuracy >= target
        best, right = 0.0, 0
        for i, (_, o) in enumerate(ranked, 1):
            right += o
            if right / i >= target:
                best = i / n_
        return round(best, 3)
    pos = [c for c, o in confs if o]; neg = [c for c, o in confs if not o]
    auroc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))
    print(json.dumps({"model": model, "provider": provider, "top_k": top_k, "answered": f"{n_}/{len(rows)}",
                      "accuracy_vs_human": round(acc / n_, 3), "top3": round(top3 / n_, 3),
                      "agrees_with_jev_pick": round(agree / n_, 3), "kl_jev_to_it": round(kl / n_, 3),
                      "mean_confidence": round(sum(c for c, _ in confs) / n_, 3), "ece": round(ece, 3),
                      "rows_with_real_spread(<0.95)": round(spread, 3),
                      "auroc_conf_vs_correct": round(auroc, 3), **keep,
                      "share_kept_at_90pct_acc": kept_at(0.90), "share_kept_at_95pct_acc": kept_at(0.95),
                      "providers": sorted({r["provider"] for r in done if r.get("provider")}),
                      "mean_coverage": round(sum(r["coverage"] for r in done) / n_, 4),
                      "min_coverage": round(min(r["coverage"] for r in done), 3),
                      "mean_shared_mass": round(sum(r["shared"] for r in done) / n_, 4),
                      "reasoning_tokens_total": sum(r["reasoning_tokens"] or 0 for r in done),
                      "median_secs": round(sorted(r["secs"] for r in done)[n_ // 2], 2), "wall_s": round(wall, 1),
                      "cost_usd_200": round(sum(r["cost"] or 0 for r in done), 4)}, indent=1))


REASONING = sys.argv[5] if len(sys.argv) > 5 else "off"

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 200))
