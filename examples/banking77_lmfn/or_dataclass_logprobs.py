"""Confidence from a code-completion frame: Python enum + dataclass, prefilled answer.

The prompt declares `class Intent(Enum)` with the 77 intents and
`@dataclass class Classification: intent: Intent`. The assistant message is
prefilled with `result = Classification(intent=Intent.` and generation stops
at ")", so the model writes only the member name. The 77-way distribution is
rebuilt from the top-20 logprobs along that name (same method as
or_logprob_targets.py); confidence = its top probability.

"plain" mode is the earlier baseline (instruction + list, free-text answer),
run through the SAME model and provider for a fair comparison.

    python or_dataclass_logprobs.py MODEL PROVIDER dataclass|plain [N]
    python or_dataclass_logprobs.py --probe MODEL [MODEL ...]
"""

import asyncio
import json
import math
import os
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = json.load(open("/home/maxime/Projects/primeintellect/data/banking77-jev/questions.json"))["labels"]
MEMBERS = [l.rstrip("?") for l in LABELS]                  # "reverted_card_payment?" is not an identifier
TEST = json.load(open(os.path.join(HERE, "jev_test200_predictions.json")))
URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = "/home/maxime/Projects/primeintellect/outputs/or-dataclass"

CODE_SYSTEM = ("Classify the bank customer's message by what they need, by completing the Python code.\n\n"
               "from dataclasses import dataclass\nfrom enum import Enum\n\n\nclass Intent(Enum):\n"
               + "".join(f"    {m} = {l!r}\n" for m, l in zip(MEMBERS, LABELS))
               + "\n\n@dataclass\nclass Classification:\n    intent: Intent\n")
PREFILL = "result = Classification(intent=Intent."
PLAIN_SYSTEM = ("Classify the bank customer's message by what they need. Answer with exactly one intent name "
                "from this list and nothing else:\n" + "\n".join(LABELS))


def request(model, provider, mode, text):
    if mode == "dataclass":
        msgs = [{"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": f"message = {text!r}"},
                {"role": "assistant", "content": PREFILL}]
        extra = {"stop": [")"]}
    else:
        msgs = [{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": text}]
        extra = {}
    return {"model": model, "messages": msgs, "max_tokens": 30, "temperature": 0, "logprobs": True,
            "top_logprobs": 20, "reasoning": {"enabled": False}, **extra,
            "provider": {"enforce_distillable_text": True, "require_parameters": True,
                         "order": [provider], "allow_fallbacks": False}}


def rebuild(tokens, names):
    """77-way distribution from top-k alternatives along the generated path (see or_logprob_targets)."""
    dist = [0.0] * len(names)
    prefix, p_prefix, lost = "", 1.0, 0.0
    for t in tokens:
        for alt in t["top_logprobs"]:
            if alt["token"] == t["token"]:
                continue
            p = p_prefix * math.exp(alt["logprob"])
            s = (prefix + alt["token"]).lstrip()
            match = [i for i, n in enumerate(names) if n.startswith(s) or (s.startswith(n) and not s[len(n):][:1].isalnum() and s[len(n):][:1] != "_")]
            if match:
                for i in match:
                    dist[i] += p / len(match)
            else:
                lost += p
        g = [a for a in t["top_logprobs"] if a["token"] == t["token"]]
        p_prefix *= math.exp(g[0]["logprob"] if g else t["logprob"])
        prefix += t["token"]
        live = [i for i, n in enumerate(names) if n.startswith(prefix.strip())]
        if len(live) <= 1:
            done = [i for i, n in enumerate(names) if prefix.strip().startswith(n)]
            i = (live or done or [None])[0]
            if i is None:
                lost += p_prefix
            else:
                dist[i] += p_prefix
            break
    else:
        exact = [i for i, n in enumerate(names) if prefix.strip() == n]
        if exact:
            dist[exact[0]] += p_prefix
        else:
            lost += p_prefix
    return dist, lost


def sane(lp):
    """Some routes return impossible data: an alternative that is not the sampled token at ~p=1."""
    return not any(math.exp(a["logprob"]) > 0.9 for t in lp for a in t["top_logprobs"] if a["token"] != t["token"])


async def probe(models):
    key = os.environ["OPENROUTER_API_KEY"]
    async with httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {key}"}) as c:
        for m in models:
            eps = (await c.get(f"https://openrouter.ai/api/v1/models/{m}/endpoints")).json()["data"]["endpoints"]
            provs = sorted({e["provider_name"] for e in eps})

            async def one(p):
                try:
                    r = await c.post(URL, json=request(m, p, "dataclass", TEST[0]["text"]))
                    j = r.json()
                    if r.status_code != 200 or not j.get("choices"):
                        return p, f"refused ({(j.get('error') or {}).get('message', '')[:50]})"
                    ch = j["choices"][0]
                    content = (ch["message"].get("content") or "").strip()
                    lp = (ch.get("logprobs") or {}).get("content") or []
                    ok_prefill = content in MEMBERS or any(content.startswith(n) for n in MEMBERS)
                    return p, (f"{'PREFILL-OK' if ok_prefill else 'prefill-no'} logprobs={'yes' if lp else 'no'} "
                               f"{'sane' if lp and sane(lp) else ('BROKEN' if lp else '')} content={content[:40]!r}")
                except Exception as e:
                    return p, f"error {repr(e)[:60]}"
            res = await asyncio.gather(*(one(p) for p in provs))
            print(m)
            for p, s in res:
                print(f"   {p:18} {s}")


async def run(model, provider, mode, n):
    key = os.environ["OPENROUTER_API_KEY"]
    names = MEMBERS if mode == "dataclass" else LABELS
    rows = TEST[:n]
    out = [None] * len(rows)
    sem = asyncio.Semaphore(12)
    async with httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {key}"}) as c:
        async def one(k):
            async with sem:
                for attempt in range(5):
                    try:
                        r = await c.post(URL, json=request(model, provider, mode, rows[k]["text"]))
                        j = r.json()
                        if r.status_code == 200 and j.get("choices"):
                            ch = j["choices"][0]
                            lp = (ch.get("logprobs") or {}).get("content") or []
                            dist, lost = rebuild(lp, names)
                            out[k] = {"text": rows[k]["text"], "label": rows[k]["label"],
                                      "content": ch["message"].get("content"), "dist": dist, "lost": lost,
                                      "sane": sane(lp), "cost": (j.get("usage") or {}).get("cost") or 0, "raw": lp}
                            return
                    except Exception:
                        pass
                    await asyncio.sleep(2 ** attempt)
        await asyncio.gather(*(one(k) for k in range(len(rows))))
    done = [r for r in out if r]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{model.replace('/', '_')}__{provider}__{mode}.jsonl", "w") as f:
        for r in done:
            f.write(json.dumps(r) + "\n")
    confs = []
    for r in done:
        d = r["dist"]; z = sum(d)
        if z == 0:
            confs.append((0.0, False)); continue
        i = max(range(77), key=lambda k: d[k])
        confs.append((d[i] / z, LABELS[i] == r["label"]))
    nn = len(confs)
    ranked = sorted(confs, key=lambda x: -x[0])

    def kept_at(t):
        best = right = 0
        for i, (_, o) in enumerate(ranked, 1):
            right += o
            if right / i >= t:
                best = i / nn
        return round(best, 3)
    pos = [c for c, o in confs if o]; neg = [c for c, o in confs if not o]
    auroc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))
    bins = [[] for _ in range(10)]
    for c_, o in confs:
        bins[min(9, int(c_ * 10))].append((c_, o))
    ece = sum(len(b) / nn * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b)
    res = {"model": model, "provider": provider, "mode": mode, "answered": nn,
           "accuracy": round(sum(o for _, o in confs) / nn, 3),
           "auroc": round(auroc, 3), "ece": round(ece, 3), "mean_conf": round(sum(c for c, _ in confs) / nn, 3),
           "acc_top80": round(sum(o for _, o in ranked[:int(.8 * nn)]) / int(.8 * nn), 3),
           "acc_top50": round(sum(o for _, o in ranked[:nn // 2]) / (nn // 2), 3),
           "keep_at_90": kept_at(0.90), "keep_at_95": kept_at(0.95),
           "mean_lost_mass": round(sum(r["lost"] for r in done) / nn, 4),
           "insane_rows": sum(not r["sane"] for r in done),
           "off_format_rows": sum(1 for r in done if sum(r["dist"]) == 0),
           "cost_per_1000": round(sum(r["cost"] for r in done) / nn * 1000, 3)}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--probe":
        asyncio.run(probe(sys.argv[2:]))
    else:
        asyncio.run(run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 200))
