---
title: From an lmfn function to RL traces — a walkthrough
rat:
  project: ..
  python:
    requires: ">=3.11"
    dependencies:
      - "-e ."
      - "-e ../lmcc/python"
      - "-e examples/alphabet_sort_lmfn"
      - "lm15==1.0.0rc1"
      - "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers@1863cfbdc75aa3ce708f534cf633e69a5f08fdd6"
---

# From an lmfn function to RL traces

A review script for the ergonomics: what a task author writes, what each
layer does with it, and what a trainer receives. It runs offline (a fake
model server stands in for the model); the last cell is the live version.

Read it top to bottom; every cell prints what it produced.

## 1. The task is one function

```python
import inspect
from alphabet_sort_lmfn import program

print(inspect.getsource(program.Entry))
print(inspect.getsource(program.sort_names))
```

This is the whole task: the inputs of one turn, the answer as typed values.
No prompt, no parser, no tags.

## 2. Adapters are ways of saying it

The same function, bound to three layouts. Each one writes the prompt and
reads the reply back into the same `list[Entry]`.

```python
import lmfn
from alphabet_sort_lmfn.program import ADAPTERS, Entry, sort_names

lmfn.configure(model="gpt-4.1-mini")      # only used to name the model; nothing is sent

turn1 = {"names": ["BoSmith", "AnnJones"], "by": "LAST", "first_turn": True}
turn2 = {"names": ["CyAdams"], "by": "LAST", "first_turn": False}
answer1 = [Entry("AnnJones", False), Entry("BoSmith", False)]

for name, adapter in ADAPTERS.items():
    plan = sort_names.with_adapter(adapter).plan()
    past = plan.example(turn1, {"answer": answer1})
    request = plan.render(plan.turn(turn2), turns=[past])
    print(f"===== {name}")
    if request.system:
        print("[system]", request.system)
    for m in request.messages:
        print(f"[{m['role']}]", m["parts"][0]["text"])
    print()
```

Things to judge here: how much of each prompt is the adapter's choice
(wording, tags, JSON) and how little the task author had to write.

## 3. Reading a reply: values, repairs, refusals

The reward never parses text. It gets values, plus a report of what the
reader had to forgive.

```python
plan = sort_names.with_adapter(ADAPTERS["original"]).plan()

for reply in [
    "<combined_alphabetical_sorted>\nCyAdams // new name!\nAnnJones\nBoSmith\n</combined_alphabetical_sorted>",
    "<Combined_Alphabetical_Sorted>\nCyAdams // new name!\nAnnJones\nBoSmith\n</combined_alphabetical_sorted>",
    "Sure! CyAdams, AnnJones, BoSmith.",
]:
    try:
        r = plan.read(reply)
        print("values :", r.values["answer"])
        print("repairs:", r.repairs or "none")
    except Exception as err:
        print("refused:", err)
    print()
```

## 4. A multi-turn episode is a Session of Turns

What the harness does in each rollout, here with a scripted stand-in for
the model so you can see the record.

```python
import lm15, lmcc

class Scripted:
    """Answers each request from a list, like a model would."""
    def __init__(self, *replies): self.replies = list(replies)
    def resolve(self, model): return type("R", (), {"provider": "scripted", "model": model})()
    def complete(self, request):
        return lm15.Response(id="r", model="m", finish_reason="stop",
                             message=lm15.Message.assistant([lm15.TextPart(self.replies.pop(0))]),
                             usage=lm15.Usage(input_tokens=1, output_tokens=1, total_tokens=2))

adapter = lmcc.adapter(messages=ADAPTERS["original"].template, formats=ADAPTERS["original"].formats,
                       replay="verbatim")          # what the harness sets: replay the model's own words
fn = sort_names.with_adapter(adapter).using(router=Scripted(
    "<Combined_Alphabetical_Sorted>\nAnnJones\nBoSmith\n</combined_alphabetical_sorted>",   # misspelled tag
    "<combined_alphabetical_sorted>\nCyAdams // new name!\nAnnJones\nBoSmith\n</combined_alphabetical_sorted>"))

episode = lmfn.Session(fn)
first = episode.call(**turn1)
second = episode.call(**turn2)
print("turn 1:", first.value, "| repairs:", [r["saw"] for r in first.repairs])
print("turn 2:", second.value)
print()
print("what the model saw on turn 2 (its own misspelling replayed as written):")
for m in fn.render(**turn2, turns=[first.turn]).messages:
    print(f"  [{m.role}]", m.parts[0].text.split("\n\nUse exactly")[0].replace("\n", " | "))
```

The turn-2 request extends the turn-1 request exactly: that is what keeps
the episode one training sample.

## 5. The verifiers side: taskset, reward, scripted user

What a task author writes for verifiers, around the function.

```python
from alphabet_sort_lmfn import taskset
print(inspect.getsource(taskset.AlphabetSortTask.alphabet_sort))
print(inspect.getsource(taskset.AlphabetSortEnv))
```

The reward reads `turns(trace)` (values the adapter read), the scripted user
sends `user_turn(**inputs)`. The episode generator is the original's code
minus the prompt strings.

## 6. Run it through vf-eval (offline, fake model)

The fake server sorts correctly, misspells a tag now and then, and answers
nonsense one time in six, so every path shows up.

```python
import os, socket, subprocess, sys, time, json, tempfile
from pathlib import Path

root = Path.cwd() if (Path.cwd() / "tests").exists() else Path.cwd().parent
port = 18799
server = subprocess.Popen([sys.executable, str(root / "tests" / "fake_openai.py"), str(port)])
time.sleep(1)
out = Path(tempfile.mkdtemp())
run = subprocess.run(
    [str(Path(sys.executable).parent / "vf-eval"), "alphabet-sort-lmfn",
     "--env.agent.harness.id", "lmfn-verifiers",
     "--env.agent.harness.program", "alphabet_sort_lmfn.program:sort_names",
     "--env.agent.harness.adapter", "original",
     "--model", "fake", "--client.base-url", f"http://127.0.0.1:{port}/v1",
     "--client.api-key-var", "FAKE_KEY", "--num-tasks", "12",
     "--no-push", "--no-rich", "--output-dir", str(out)],
    env={**os.environ, "FAKE_KEY": "x"}, capture_output=True, text=True)
server.terminate()
print("vf-eval exit code:", run.returncode)
traces_file = next(out.rglob("traces.jsonl"))
print("traces:", traces_file)
```

## 7. What a trainer receives

Each line of `traces.jsonl` is verifiers' own record. The lmfn harness adds
one thing: the values it read, under `info["lmfn"]`.

```python
from verifiers.v1.trace import Trace

rows = [json.loads(line) for line in traces_file.read_text().splitlines()]
rows.sort(key=lambda r: -r["task"]["data"]["info"]["num_turns"])     # multi-turn episodes first
for row in rows:
    t = row["traces"][0]
    trace = Trace.model_validate(t)
    turns = t["info"]["lmfn"]["turns"]
    print(f"episode {row['task']['data']['idx']}: {len(turns)} turn(s), "
          f"{len(trace.branches)} training path(s), reward {t['rewards']['alphabet_sort']['score']:.2f}")
    for k, x in enumerate(turns):
        state = "UNREADABLE (" + x["refusal"]["code"] + ")" if "refusal" in x else \
                f"{len(x['outputs']['answer'])} names" + (" (repaired)" if x["repairs"] else "")
        print(f"   turn {k + 1}: {state}")
```

```python
t = rows[0]["traces"][0]          # the longest episode
trace = Trace.model_validate(t)
print(f"one training path ({len(trace.branches)} for this episode), as the trainer sees it:\n")
for m in trace.branches[0].messages:
    text = m.content if isinstance(m.content, str) else str(m.content)
    print(f"[{m.role}]", text[:200].replace("\n", " | "))
```

## 8. Live (costs cents; skip if you only review)

Change `RUN_LIVE` to `True` to run 10 episodes on gpt-4.1-mini with each adapter.

```python
RUN_LIVE = False
if RUN_LIVE:
    for line in (Path.home() / "Projects/lm15-dev/.env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
    for name in ADAPTERS:
        live = Path(tempfile.mkdtemp())
        subprocess.run([str(Path(sys.executable).parent / "vf-eval"), "alphabet-sort-lmfn",
                        "--env.agent.harness.id", "lmfn-verifiers",
                        "--env.agent.harness.program", "alphabet_sort_lmfn.program:sort_names",
                        "--env.agent.harness.adapter", name, "--model", "gpt-4.1-mini",
                        "--client.base-url", "https://api.openai.com/v1",
                        "--client.api-key-var", "OPENAI_API_KEY", "--num-tasks", "10",
                        "--no-push", "--no-rich", "--output-dir", str(live)], capture_output=True)
        rs = [json.loads(l)["traces"][0] for l in next(live.rglob("traces.jsonl")).read_text().splitlines()]
        print(f"{name:9} mean reward {sum(r['rewards']['alphabet_sort']['score'] for r in rs) / len(rs):.3f}")
```
