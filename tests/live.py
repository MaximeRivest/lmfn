"""lmfn against real models (costs cents). Not run by ./check.

    set -a; source ~/Projects/lm15-dev/.env; set +a
    .venv/bin/python tests/live.py
"""

import dataclasses
import enum
import sys

import lmcc
import lmfn


class Sentiment(enum.Enum):
    positive = "positive"
    negative = "negative"
    mixed = "mixed"


@dataclasses.dataclass
class Review:
    sentiment: Sentiment
    stars: int
    would_return: bool


@lmfn.ai(max_tokens=2000)
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""


@lmfn.ai(max_tokens=2000)
def read_review(text: str) -> Review:
    """Read the restaurant review. Stars are 1 to 5."""


@lmfn.ai(reasoning=True, max_tokens=2000)
def solve(problem: str) -> int:
    """Solve the word problem."""


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"Sunny and 22C in {city}."


@lmfn.ai(tools=[get_weather], max_tokens=2000)
def assistant(question: str) -> str:
    """Answer the question. Use a tool when you need facts you do not have."""


@lmfn.ai(max_tokens=2000)
def chat(message: str) -> str:
    """You are a friendly assistant. Reply in one short sentence."""


MODELS = ["gpt-4.1-mini", "claude-haiku-4-5", "gemini-2.5-flash", "openrouter:qwen/qwen-2.5-72b-instruct"]


def run(model):
    out = {}
    with lmfn.configure(model=model):
        out["summarize"] = summarize("LMCC is the calling convention for language models: it lays out "
                                     "each call and reads each reply as typed values.")
        r = read_review("Great tacos, far too loud. I'll be back.")
        assert isinstance(r.sentiment, Sentiment) and 1 <= r.stars <= 5 and r.would_return is True, r
        out["review"] = r
        res = solve.call("Ana buys 7 pens at 3 each and pays with a 50. How much change does she get?")
        assert res.value == 29, res.values
        out["solve"] = (res.value, res.values["reasoning"][:50])
        res = assistant.call("What is the weather in Montreal right now?")
        assert "22" in res.value, res.value
        out["tools"] = ([s.kind for s in res.turn.steps], res.value[:50])
        first = chat.call("Hi, my name is Ana.")
        second = chat.call("What is my name?", turns=[first.turn])
        assert "Ana" in second.value, second.value
        out["turns"] = second.value[:50]
        events = list(s := summarize.stream("Streaming reads the reply field by field as it arrives."))
        assert s.result.value and sum(e["kind"] == "field_delta" for e in events) >= 1
        out["stream"] = f"{sum(e['kind'] == 'field_delta' for e in events)} deltas"
    return out


def main():
    failed = 0
    for model in sys.argv[1:] or MODELS:
        try:
            out = run(model)
            print(f"OK   {model}")
            for k, v in out.items():
                print(f"       {k:10} {v}")
        except (AssertionError, lmcc.Refusal, lmfn.StepLimit, Exception) as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {model}: {type(exc).__name__}: {str(exc)[:300]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
