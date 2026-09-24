# lmfn

```bash
pip install lmfn                 # functions and sessions
pip install "lmfn[verifiers]"    # + the verifiers harness, for RL and distillation
```

**Status: alpha.** Every example below runs, except where a section says
otherwise. Offline tests: `./check`. Live: `tests/live.py` passed on
gpt-4.1-mini, claude-haiku-4-5, gemini-2.5-flash and qwen-2.5-72b. Training:
`docs/rl-walkthrough.md` (a runnable walkthrough) and the examples
`examples/alphabet_sort_lmfn` (multi-turn RL) and `examples/banking77_lmfn`
(distilling a thinking teacher into a small model). Development setup:
`./dev-venv` (lmcc editable from `../lmcc`).

**The function is the prompt; calling it calls the model.** lmfn is the
thin layer that funnydspy, functai and elemai each prototyped: a typed
Python function whose body is a model call. It stands on two libraries
and adds only what they deliberately leave out:

| layer | answers | library |
|---|---|---|
| wire | what bytes go to which provider | **lm15** (router, providers, tools from functions) |
| spelling | how each value is written into the prompt and read back | **lmcc** (signature, adapter, bind, render, read, Turn) |
| **control flow** | **make the call; if the model asks for tools, run them and call again; return values** | **lmfn** |
| state | what a later call remembers | a session library, later, on top of lmfn |

What lmfn owns and nothing else: *which model* (a default, overridable),
*the loop* (one call, or the tool loop), and *what you get back* (the
value, or everything). No prompt syntax, no parser, no provider code.

Working name `lmfn`.

---

## 1. A function that calls a model

```python
import lmfn

lmfn.configure(model="gpt-4.1-mini")          # any lm15 model string

@lmfn.ai
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""

summarize("LMCC is the calling convention for language models ...")
# 'LMCC defines how values are passed to and read back from language models.'
```

The parameters are the inputs, the return type is the output, the
docstring is the instruction. No body, no sentinel: `@lmfn.ai` is
`@lmcc.fn` plus a model. The call returns a plain `str`.

## 2. Types are the contract

```python
from dataclasses import dataclass
from enum import Enum

class Sentiment(Enum):
    positive = "positive"
    negative = "negative"
    mixed = "mixed"

@dataclass
class Review:
    sentiment: Sentiment
    stars: int            # 1 to 5
    would_return: bool

@lmfn.ai
def read_review(text: str) -> Review:
    """Read the restaurant review."""

r = read_review("Great tacos, loud music. I'll be back.")
r.sentiment, r.stars, r.would_return        # (Sentiment.mixed, 4, True)
```

A dataclass return is several outputs; `lmcc.One[Review]` makes it one
structured output instead. Everything lmcc checks, it checks here, and
before any call: a type with no format refuses at the first call's bind,
not at the provider.

## 3. Thinking first, without changing what you get back

```python
@lmfn.ai(reasoning=True)
def solve(problem: str) -> int:
    """Solve the word problem."""

solve("Ana buys 7 pens at 3 each and pays with 50. Change?")    # 29
```

`reasoning=True` adds a hidden `reasoning` output. The value you get back
is still the `int`. How the reasoning travels is chosen per model, by
lmcc's `choose`: the provider's native thinking where the model has it,
`<think>` tags otherwise. Same program on both.

## 4. Everything, when you want it

```python
res = solve.call("Ana buys 7 pens ...")
res.value          # 29
res.values         # {'answer': 29, 'reasoning': '7 × 3 = 21, 50 − 21 = 29'}
res.repairs        # what the reader repaired (lmcc §4a), [] when clean
res.turn           # the lmcc.Turn: inputs, every model step and tool step, outputs
res.response       # the last lm15 Response (usage, finish reason, provider data)
res.usage          # summed over every model call of this turn
```

`f(...)` is `f.call(...).value`. One method, not a magic keyword: a
keyword like functai's `all=True` collides with an input named `all`.

## 5. See the call before paying for it

```python
req = solve.render("Ana buys 7 pens ...")   # the exact lm15 Request; no network
print(req.system)
solve.explain()                              # adapter, transports chosen, formats, stops
```

## 6. Tools: the loop, stated

```python
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"Sunny and 22°C in {city}."

@lmfn.ai(tools=[get_weather], max_steps=5)
def assistant(question: str) -> str:
    """Answer the question. Use a tool when you need facts you do not have."""

assistant("What's the weather in Montreal?")     # 'It is sunny and 22°C in Montreal.'
```

Tools are plain functions (lm15's `derive_tool` makes their schema).
With tools, `call` runs the loop: render, call the model, and while the
reply asks for tools, run them, record each result as a step of the same
turn, call again. `max_steps` bounds it (default 8); reaching it raises
`lmfn.StepLimit` with the turn so far. A tool that raises becomes a
result the model sees (`"error: ..."`), unless `tool_errors="raise"`.

What does **not** happen: the prompt style never changes because tools
were added (functai silently switched to ReAct). Native tool calls where
the model has them, fenced text calls otherwise, chosen by lmcc like
reasoning. `res.turn.steps` shows every call and result.

## 7. Examples

```python
@lmfn.ai(examples=[("I love it", Sentiment.positive), ("Awful.", Sentiment.negative)])
def classify(text: str) -> Sentiment:
    """Classify the sentiment."""
```

Each example becomes an lmcc example turn, written through the adapter
like everything else. A tuple is `(inputs, output)` for a one-input
function; a dict `{"inputs": {...}, "outputs": {...}}` in general.

## 8. Which model, and where settings live

```python
lmfn.configure(model="gpt-4.1-mini", temperature=0.2)     # process default

with lmfn.configure(model="claude-haiku-4-5"):             # this block
    solve("...")

@lmfn.ai(model="gemini-2.5-flash", max_tokens=2000)        # this function
def draft(topic: str) -> str: ...

draft.using(model="gpt-4.1")("robots")                     # this call
```

Innermost wins: `using` > decorator > `with` block > `configure`.
Settings are lm15 `Config` fields; unknown ones refuse. `using` returns a
new function and never touches the original, so no per-call keyword can
collide with an input name.

**What the model can do** (lmcc capabilities: `instruct`,
`native_function_calling`, `native_reasoning`, `stop_sequences`) comes
from the lm15 provider the model routes to and a small table of model
names (`lmfn/models.py`), because lm15's model information does not say
yet. OpenAI, Anthropic, Gemini and xAI get native tool calls; other hosts
get text tool calls. Override when you know better:
`@lmfn.ai(capabilities={"native_reasoning": False})`.

**Thinking and the token budget.** Some models think before answering
by default (Gemini 2.5). That thinking counts against `max_tokens`. lmfn
leaves the model's default alone (some models cannot turn thinking off);
if a reply is cut off, the error says how many tokens went to thinking.
With native reasoning, some providers keep the thinking text private, so
`res.values["reasoning"]` can be empty while the answer is right.

## 9. Your own layout

```python
xml = lmcc.adapter(messages=[
    lmcc.system("{instruction}\n{% for f in outputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"),
    lmcc.turns(), lmcc.user("{text}")])

@lmfn.ai(adapter=xml)
def summarize(text: str) -> str: ...
```

The default adapter is one lmfn ships, printed by `explain()`. Any lmcc
adapter works, including one loaded from JSON.

## 10. When the reply is wrong

A reply lmcc cannot read raises its `lmcc.Refusal` (`parse-missing-fields`,
`parse-truncated`, ...), with `.partial`. Retrying is a decision about
money, so it is opt-in and visible:

```python
@lmfn.ai(retries=1)
def extract(text: str) -> Review: ...
```

On a parse refusal, lmfn sends the reply back once with the refusal's
hint ("reply is missing pattern section(s): 'stars'"), as a follow-up in
the same turn. `res.turn` shows both attempts. `parse-truncated` retries
with double `max_tokens` instead.

## 11. Streaming

```python
s = solve.stream("...")
for event in s:                        # lmcc stream events, field by field
    ...
s.result                               # the same CallResult, after the loop
```

Not built yet: streaming a tool loop, and async (`acall`).

## 12. Sessions

```python
chat = lmfn.Session(rag, forget=["context"], window=20)
chat("Who founded the bakery?", context=docs_1)
chat("And in what year?", context=docs_2)   # docs_1 is not sent again
chat.undo(); chat.fork(); chat.save("s.json"); lmfn.Session.load("s.json", rag)
chat.score(metric); chat.examples(min_score=0.5)   # training data
```

A session is a list of lmcc turns and a function. `forget` leaves inputs
out of *past* turns (the current call still gets them; the record keeps
them). **The cost of forgetting, seen live:** a fact that was only in a
forgotten document is gone unless the model's answer repeated it. On
2026-09-23, gpt-4.1-mini answered "Marie Tremblay founded the bakery"
without the year, so on the next turn it guessed a wrong year; Claude
had written the year in its answer and kept it (`tests/live_session.py`).

## 13. Training: the verifiers harness

`pip install "lmfn[verifiers]"` adds the harness `lmfn-verifiers`: any
`@lmfn.ai` function runs as a verifiers agent, with any lmcc adapter, and
verifiers records the traces trainers use (prime-rl RL, SFT distillation).

```bash
vf-eval alphabet-sort-lmfn \
    --env.agent.harness.id lmfn-verifiers \
    --env.agent.harness.program alphabet_sort_lmfn.program:sort_names \
    --env.agent.harness.adapter original
```

Past replies are replayed verbatim, so a multi-turn episode stays one
training sample; `lmfn_verifiers.AnswerOnlyTask` drops a teacher's
reasoning before the trace is recorded, so a small student learns the
answer only. See `docs/rl-walkthrough.md`.

## 14. Turns connect to the next layer

```python
first = assistant.call("My name is Ana.")
second = assistant.call("What is my name?", turns=[first.turn])
```

A past turn is lmcc's record, re-spelled through the current adapter.
Keeping, trimming, forking and saving a list of turns is the session
library's job, built on exactly this.

---

## Decisions taken (2026-09-23)

1. **Name.** `lmfn` is a working name, not final.
2. **No hidden steps in the body** (functai/elemai's `thinking: str = _ai["..."]`):
   it needs the body parsed as code and makes the body mean two things.
   `reasoning=True` and `lmcc.Purpose` in a dataclass cover it.
3. **Retries** are opt-in (`retries=0` by default): retrying spends money.
4. **Capabilities** come from a small table in lmfn until lm15 carries them.
