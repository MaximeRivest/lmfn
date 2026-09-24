---
title: "lmfn: functions whose body is a model"
rat:
  project: ..
  python:
    requires: ">=3.10"
    dependencies: ["-e .", "-e ../lmcc/python", "lm15==1.0.0rc1", "polars"]
---

# lmfn: functions whose body is a model

You already know how to describe a task to a program: you write a function. You give it a name, you say what goes in, you say what comes out, and you explain what it does in the docstring. lmfn takes that description and gives it to a language model. The model does the task and you get back an ordinary Python value.

This vignette walks through the whole library with one running example: reading restaurant reviews. Every cell below was run on 2026-09-23 against `gpt-4.1-mini` (plus `claude-haiku-4-5` and `gemini-2.5-flash` in the section about models), and the outputs shown are what it printed. Models don't answer the same way every time, so yours may differ a little.

**Setup.** From the lmfn checkout, `./dev-venv` builds the environment (or `rat ensure docs/vignette.md`). The model provider's key comes from the environment: `OPENAI_API_KEY` for OpenAI models, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` for the others.

```python
import os
from pathlib import Path

# Load provider keys from the local .env file without replacing keys already set.
env_file = Path("~/Projects/lm15-dev/.env").expanduser()
if env_file.exists():
    for line in env_file.read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))

import lmfn

# Use this model for functions that don't specify their own.
lmfn.configure(model="gpt-4.1-mini")
```

```output
<lmfn.core.configure object at 0x7559f3ffd8e0>
```

`configure` sets the model every function uses unless told otherwise. Any model name lm15 knows works here.

## 1. Your first function

```python
@lmfn.ai
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""

summarize("We waited forty minutes for a table, but the ramen was the best "
          "I've had outside Tokyo and the staff apologized twice.")
```

```output
'Despite a forty-minute wait for a table, the ramen was the best the reviewer had outside Tokyo and the staff were apologetic.'
```

That is the whole program. There is no body and no prompt string:

| you write | the model gets |
|---|---|
| the docstring | the instruction |
| the parameters (`text: str`) | the inputs, one labelled section each |
| the return type (`-> str`) | the form of the answer it must give |

What you get back is a plain `str`. Calling `summarize` looks and feels like calling any other function, which is the point: code that uses it does not need to know a model is involved.

## 2. Types are the contract

The return type is not decoration. It decides what the model is asked for and what you get back. An `Enum` restricts the answer to its members:

```python
from enum import Enum

class Sentiment(Enum):
    positive = "positive"
    negative = "negative"
    mixed = "mixed"

@lmfn.ai
def classify(text: str) -> Sentiment:
    """Classify the sentiment of the restaurant review."""

classify("Great tacos, loud music. I'll be back.")
```

```output
<Sentiment.positive: 'positive'>
```

The result is a real `Sentiment` member, so `if s is Sentiment.mixed:` just works. A reply that isn't one of the members is an error, not a string that slips through.

A dataclass asks for several things at once. Each field becomes its own output, and comments in your code are for you: put hints the model needs in the docstring.

```python
from dataclasses import dataclass

@dataclass
class Review:
    sentiment: Sentiment
    stars: int
    would_return: bool
    dish: str

@lmfn.ai
def read_review(text: str) -> Review:
    """Read the restaurant review. Stars are 1 to 5. dish: the dish the
    reviewer talks about most, or 'none'."""

read_review("The carbonara was perfect but the service was painfully slow. "
            "Probably worth another try.")
```

```output
Review(sentiment=<Sentiment.mixed: 'mixed'>, stars=3, would_return=True, dish='carbonara')
```

The types that work today: `str`, `int`, `float`, `bool`, `Enum`, `Literal[...]`, `Optional[...]` of those, and dataclasses made of them.

Types that don't work yet are refused **before any call is made**, so a mistake never costs money. A list, for example:

```python
import lmcc

@lmfn.ai
def dishes(text: str) -> list[str]:
    """List every dish the review names."""

try:
    dishes.render("Tacos, then churros.")     # render never calls the model
except lmcc.Refusal as err:
    print(err.code)
```

```output
no-format
```

(Lists, dicts and dataframes are the biggest gap today; see the end of this vignette.)

## 3. Letting the model think first

Some questions go better when the model works through them before answering. `reasoning=True` asks for that without changing what you get back:

```python
@lmfn.ai(reasoning=True)
def bill(problem: str) -> float:
    """Work out the amount asked for, in dollars."""

bill("Four friends share a 96 dollar dinner and add a 15% tip. "
     "How much does each one pay?")
```

```output
27.6
```

The value is still a `float`. The reasoning went somewhere you can look at (next section). How it travels depends on the model: models with built-in thinking use it, others write it between `<think>` tags. The same code works on both kinds.

## 4. Getting everything, not just the value

`f(...)` gives you the value. `f.call(...)` gives you everything the call produced:

```python
res = bill.call("Four friends share a 96 dollar dinner and add a 15% tip. "
                "How much does each one pay?")
print("value:    ", res.value)
print("reasoning:", res.values["reasoning"])
print("tokens:   ", res.usage["input_tokens"], "in,", res.usage["output_tokens"], "out")
print("repairs:  ", res.repairs)
```

```output
value:     27.6
reasoning: 15% of 96 = 0.15 × 96 = 14.4
Total amount = 96 + 14.4 = 110.4
Each pays = 110.4 ÷ 4 = 27.6
tokens:    82 in, 139 out
repairs:   [{'repair': 'ignored', 'saw': 'First, we calculate the tip amount by taking 15% of 96 dollars. \n\nNext, add the tip to the original bill to find the total amount paid. \n\nSince four friends share this total equally, we divide the total amount by 4. \n\nTherefore, each friend pays 27.6 dollars.'}]
```

- `value`: what `bill(...)` would have returned.
- `values`: every output by name, including hidden ones like `reasoning`.
- `usage`: tokens, added up over every model call this turn made.
- `repairs`: what was fixed or set aside while reading the reply (a missing closing tag, say). Empty when the reply was clean. In the run above, the model also wrote its working outside the tags it was asked for; lmfn ignored that text and recorded that it did.
- `turn`: the full record of the call (inputs, every model step and tool step, outputs). Sections 8 and 11 use it.
- `response`: the provider's last raw response, if you need something provider-specific.

## 5. Seeing the call before paying for it

`render` builds the exact request that would be sent, without sending it. `explain` says how each field will be written and read.

```python
req = read_review.render("The carbonara was perfect but the service was slow.")
print(req.system)
print("---")
print(req.messages[-1].parts[0].text)
```

```output
Read the restaurant review. Stars are 1 to 5. dish: the dish the
reviewer talks about most, or 'none'.

Reply in exactly this form:
<sentiment>
one of: positive, negative, mixed
</sentiment>
<stars>
(integer)
</stars>
<would_return>
(boolean)
</would_return>
<dish>
...
</dish>

---
<text>
The carbonara was perfect but the service was slow.
</text>
```

```python
print(read_review.explain())
```

```output
adapter: lmfn_default
reader: derived
input  text                 kernel-scalar (kernel)
output sentiment            kernel-scalar (kernel)
output stars                kernel-scalar (kernel)
output would_return         kernel-scalar (kernel)
output dish                 kernel-scalar (kernel)
```

Use these when an answer surprises you: most surprises are visible in the prompt.

## 6. Teaching by example

When the instruction alone leaves room for doubt, show a few examples. Here, a review that praises the food but will not come back should count as `mixed`, not `positive`:

```python
@lmfn.ai(examples=[
    ("Loved the food, but I'm never sitting in that draft again.", Sentiment.mixed),
    ("Flawless from start to finish.", Sentiment.positive),
])
def classify_strict(text: str) -> Sentiment:
    """Classify the sentiment of the restaurant review."""

classify_strict("Best curry in town. Shame about the forty-minute wait, I won't queue like that again.")
```

```output
<Sentiment.mixed: 'mixed'>
```

Each example is written into the prompt as an earlier question and its answer, in exactly the same layout as the real call. A tuple is `(input, output)` for a function with one input; for more inputs, use `{"inputs": {...}, "outputs": {...}}`.

## 7. Tools: letting the model look things up

A tool is a plain Python function with type hints and a docstring. Give the model tools and lmfn runs a loop: call the model; if it asks for a tool, run the tool, show it the result, and call it again; stop when it answers.

Here the model gets SQL access to a small table of reviews:

```python
import polars as pl

reviews = pl.DataFrame({
    "cuisine": ["mexican", "mexican", "italian", "italian", "thai", "thai", "thai", "italian"],
    "stars":   [4, 1, 5, 3, 3, 2, 5, 1],
    "review":  ["Great tacos, loud music.", "Cold burrito, rude staff.",
                "Best carbonara in town.", "Pizza was fine, nothing special.",
                "Too spicy for me.", "Pad thai was soggy.",
                "Wonderful curry, friendly owner!", "Overpriced and slow."],
})

def query(sql: str) -> str:
    """Run a SQL query on the table `reviews` (columns: cuisine, stars, review)."""
    return str(pl.SQLContext(reviews=reviews).execute(sql).collect())

@lmfn.ai(tools=[query], max_steps=6, retries=1)
def analyst(question: str) -> str:
    """Answer the question about the restaurant reviews. Use the query tool to look at the data."""

res = analyst.call("Which cuisine has the lowest average stars, and what do people complain about?")
print(res.value)
print()
for step in res.turn.steps:
    if step.kind == "model" and step.outputs.get("calls"):
        for c in step.outputs["calls"]:
            print("model asked:", c.input["sql"])
```

```output
The Mexican cuisine has the lowest average stars, with an average rating of 2.5. People complain about cold food and rude staff.

model asked: SELECT cuisine, AVG(stars) as avg_stars FROM reviews GROUP BY cuisine ORDER BY avg_stars ASC LIMIT 1;
model asked: SELECT review FROM reviews WHERE cuisine = 'mexican' AND stars < 3;
```

What lmfn promises about the loop:

- **It is bounded.** After `max_steps` model calls (8 by default) without an answer, it raises `lmfn.StepLimit`, which carries the turn so far.
- **Tool errors don't crash your program.** If a tool raises, the model sees `"error: ..."` and can try again. Pass `tool_errors="raise"` to get the exception instead.
- **The prompt style doesn't change** because tools were added. Models with built-in tool calling use it; others get a text form.
- **Everything is on the record.** `res.turn.steps` lists every model call and every tool result.

Why `retries=1`: after a few tool calls, the instruction about the reply form is far back in the conversation, and small models sometimes forget it. Without the retry, `gpt-4.1-mini` failed 2 runs out of 3 here: its final answer was right but wrapped in a made-up tag (`<mexican ...>` instead of `<answer>`). With `retries=1` it passed 5 runs out of 5 (section 9 explains retries).

## 8. Choosing the model

Settings can live in four places. The most specific one wins:

```python
lmfn.configure(model="gpt-4.1-mini")        # 1. for the whole program

with lmfn.configure(model="claude-haiku-4-5"):   # 2. for a block
    ...

@lmfn.ai(model="gemini-2.5-flash")           # 3. for one function
def f(x: str) -> str: ...

big_f = f.using(model="gpt-4.1")             # 4. a copy for one use
```

Settings are the model plus lm15's options (`temperature`, `max_tokens`, ...); a misspelled one is an error. `using` returns a new function and leaves the original alone, which is why no per-call keyword is needed (it could clash with an input of the same name).

The same function on three providers:

```python
text = "Best curry in town. Shame about the forty-minute wait, I won't queue like that again."
for model in ["gpt-4.1-mini", "claude-haiku-4-5", "gemini-2.5-flash"]:
    print(f"{model:18}", read_review.using(model=model, max_tokens=2000)(text))
```

```output
gpt-4.1-mini       Review(sentiment=<Sentiment.mixed: 'mixed'>, stars=3, would_return=False, dish='curry')
claude-haiku-4-5   Review(sentiment=<Sentiment.mixed: 'mixed'>, stars=3, would_return=False, dish='curry')
gemini-2.5-flash   Review(sentiment=<Sentiment.mixed: 'mixed'>, stars=4, would_return=False, dish='curry')
```

`max_tokens=2000` is there for Gemini 2.5, which thinks before it answers by default, and that thinking counts against the token budget.

## 9. When the reply can't be read

Sometimes a reply comes back in a form that can't be read: a section is missing, or it was cut off. lmfn raises `lmcc.Refusal`, with a code that says what went wrong. Here the token budget is far too small:

```python
try:
    read_review.using(max_tokens=16)("The carbonara was perfect.")
except lmcc.Refusal as err:
    print(err.code)
    print(err.hint)
```

```output
parse-truncated
the provider cut the reply at its length limit before field 'would_return'; raise max_tokens or ask for less
```

Retrying costs money, so it is off unless you ask. With `@lmfn.ai(retries=1)`, lmfn tries once more: for a missing section it tells the model what was wrong, and for a cut-off reply it doubles `max_tokens`. Both attempts are in `res.turn`.

## 10. Streaming

`stream` gives you the reply as it arrives, field by field:

```python
s = summarize.stream("We waited forty minutes for a table, but the ramen was the best "
                     "I've had outside Tokyo and the staff apologized twice.")
for event in s:
    if event["kind"] == "field_delta":
        print(repr(event["text"]), end=" ")
print()
print(s.result.value)
```

```output
'Despite' ' waiting' ' forty' ' minutes' ' for' ' a' ' table' ',' ' the' ' ramen' ' was' ' the' ' best' ' outside' ' Tokyo' ' and' ' the' ' staff' ' apologized' ' twice' '.' 
Despite waiting forty minutes for a table, the ramen was the best outside Tokyo and the staff apologized twice.
```

After the loop, `s.result` is the same `CallResult` that `call` returns. Streaming does not work with tools yet.

## 11. Conversations

Every call returns its turn. Pass earlier turns back and the model sees them as the conversation so far:

```python
@lmfn.ai
def host(message: str) -> str:
    """You are the host of a small restaurant. Reply in one short sentence."""

first = host.call("Hi, a table for four at 7pm please. The name is Ana.")
print(first.value)
second = host.call("Sorry, what name did I give?", turns=[first.turn])
print(second.value)
```

```output
Thank you, Ana; your table for four at 7 PM is confirmed.
You gave the name Ana for the reservation.
```

Keeping that list by hand gets tedious, so `lmfn.Session` keeps it for you:

```python
chat = lmfn.Session(host)
chat("Hi, a table for four at 7pm please. The name is Ana.")
chat("Actually, make it five people.")
print(chat("Can you confirm my booking?"))
print(len(chat), "turns so far")
```

```output
Ana, your reservation for five people at 7pm is confirmed.
3 turns so far
```

A session is a list of turns plus a function, and everything else edits that list:

| | |
|---|---|
| `chat.undo()` | drop the last turn (retry a question differently) |
| `chat.fork()` | an independent copy (explore two directions) |
| `chat.add(inputs, outputs)` | record a turn by hand (a correction) |
| `chat.save(path)`, `Session.load(path, fn)` | keep it on disk |
| `window=20` | send only the last 20 turns |
| `forget=["context"]` | leave that input out of *past* turns |
| `chat.score(metric)`, `chat.examples(min_score=...)` | grade turns, keep the good ones as training data |

`forget` is for bulky inputs like retrieved documents: the current call still gets them, later calls don't. That has a real cost. A fact that was only in a forgotten document is gone unless the model's answer repeated it. In a live test, one model answered "Marie Tremblay founded the bakery" without the year, so when asked the year on the next turn it guessed wrong (`tests/live_session.py`).

## 12. What lmfn does not do (yet)

It is honest to end with the edges:

- **Only simple types.** No `list`, `dict` or dataframe as an input or output yet (section 2). To send a table, turn it into text first.
- **`lmcc.One[...]` fails**, although the README shows it.
- **Tool loops need `retries=1` in practice** with small models (section 7). It could become the default when tools are given.
- **One call at a time.** Running a function over 10,000 rows means 10,000 calls one after another (about 1.8 s each on `gpt-4.1-mini`), with no parallelism, no cache and no cost estimate.
- **No async** (`acall`), and **no streaming with tools**.
- **Model abilities come from a small table** (`lmfn/models.py`) until lm15 reports them. Override with `@lmfn.ai(capabilities={...})`.

## Cheat sheet

| you want | write |
|---|---|
| a function | `@lmfn.ai` over a typed function with a docstring |
| the default model | `lmfn.configure(model=...)` |
| a model for a block | `with lmfn.configure(model=...):` |
| a model for one function | `@lmfn.ai(model=...)` |
| a one-off variant | `f.using(model=..., temperature=...)` |
| a fixed set of answers | return an `Enum` or `Literal` |
| several answers | return a dataclass |
| thinking first | `@lmfn.ai(reasoning=True)` |
| everything about a call | `f.call(...)` |
| the prompt, for free | `f.render(...)`, `f.explain()` |
| examples | `@lmfn.ai(examples=[(input, output), ...])` |
| tools | `@lmfn.ai(tools=[fn, ...], max_steps=...)` |
| retry unreadable replies | `@lmfn.ai(retries=1)` |
| streaming | `for e in f.stream(...)`, then `.result` |
| a conversation | `f.call(..., turns=[...])` or `lmfn.Session(f)` |
| your own prompt layout | `@lmfn.ai(adapter=lmcc.adapter(...))` |
