"""Sessions: a function that remembers (dspy_session's ideas, on lmcc turns).

A session is a list of lmcc turns and a function. Each call sends the
kept turns as the past, then records the new turn. Everything else is an
operation on that list: forget inputs in past turns, keep a window,
undo, fork, add by hand, save, load, score, and turn the list into
training examples. Nothing here touches prompts or providers: the
function's adapter writes past turns, lm15 sends them.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import lmcc

from .core import CallResult, Function


@dataclasses.dataclass
class Example:
    """One training example from a session: the turns the model saw before,
    and the turn it produced."""
    past: list
    turn: lmcc.Turn

    @property
    def inputs(self) -> dict:
        return dict(self.turn.inputs)

    @property
    def outputs(self) -> dict:
        return dict(self.turn.outputs or {})


class Session:
    """``chat = lmfn.Session(fn, forget=["context"], window=20)``

    - ``forget``: inputs left out of past turns (bulky retrieved context);
      the current call still gets them. The adapter must read a turn
      without them: lmfn's default adapter does (its inputs loop writes
      what a turn has); a custom template uses a guard,
      ``{% if context %}…{% endif %}`` (lmcc D-45).
    - ``window``: send at most this many past turns (the most recent).
    - ``turns``: start from recorded turns (e.g. loaded from disk).
    """

    def __init__(self, fn: Function, *, forget: Iterable[str] = (), window: int | None = None,
                 turns: Iterable[lmcc.Turn] = ()):
        if not isinstance(fn, Function):
            raise TypeError("Session wraps an @lmfn.ai function")
        unknown = set(forget) - {f.name for f in fn.signature.fields if f.direction == "input"}
        if unknown:
            raise TypeError(f"forget: {sorted(unknown)} are not inputs of {fn.__name__}")
        self.fn = fn
        self.forget = tuple(forget)
        self.window = window
        self.turns: list[lmcc.Turn] = list(turns)

    # -- calling

    def past(self) -> list[lmcc.Turn]:
        """The turns the next call sends: windowed, forgotten inputs removed."""
        kept = self.turns[-self.window:] if self.window else list(self.turns)
        if not self.forget:
            return kept
        return [dataclasses.replace(t, inputs={k: v for k, v in t.inputs.items() if k not in self.forget})
                for t in kept]

    def call(self, *args, **kwargs) -> CallResult:
        res = self.fn.call(*args, turns=self.past(), **kwargs)
        self.turns.append(res.turn)
        return res

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs).value

    def render(self, *args, **kwargs):
        """The request the next call would send. No network."""
        return self.fn.render(*args, turns=self.past(), **kwargs)

    # -- editing

    def add(self, inputs: dict, outputs: dict) -> lmcc.Turn:
        """Record a turn by hand (a gold answer, a correction)."""
        turn = self.fn.plan().example(inputs, outputs)
        self.turns.append(turn)
        return turn

    def undo(self, n: int = 1) -> list[lmcc.Turn]:
        """Remove the last ``n`` turns; returns them, oldest first."""
        n = max(0, min(n, len(self.turns)))
        removed = self.turns[len(self.turns) - n:]
        del self.turns[len(self.turns) - n:]
        return removed

    def reset(self) -> None:
        self.turns.clear()

    def fork(self) -> "Session":
        """An independent copy: calls on either do not change the other."""
        return Session(self.fn, forget=self.forget, window=self.window, turns=copy.deepcopy(self.turns))

    def using(self, fn: Function) -> "Session":
        """The same conversation continued with another function (an improved
        prompt, another model). Its signature must be the same."""
        if fn.plan().fingerprint != self.fn.plan().fingerprint:
            raise ValueError("the new function has a different signature; its turns would not fit")
        return Session(fn, forget=self.forget, window=self.window, turns=list(self.turns))

    def __len__(self) -> int:
        return len(self.turns)

    # -- scoring and training data

    def score(self, metric: Callable[[lmcc.Turn], float]) -> list[float]:
        """Score every turn with ``metric(turn) -> float``; stored on the turn."""
        scores = [float(metric(t)) for t in self.turns]
        self.turns = [dataclasses.replace(t, score=s) for t, s in zip(self.turns, scores)]
        return scores

    def examples(self, *, min_score: float | None = None, stop_at_bad: bool = False) -> list[Example]:
        """Each turn with the past it was answered after, as a training
        example. ``min_score`` keeps scored turns at or above it;
        ``stop_at_bad`` drops everything from the first turn below it
        (later turns may rest on a bad answer)."""
        out = []
        for i, turn in enumerate(self.turns):
            bad = min_score is not None and (turn.score is None or turn.score < min_score)
            if bad and stop_at_bad:
                break
            if bad:
                continue
            past = self.turns[max(0, i - self.window) if self.window else 0:i]
            if self.forget:
                past = [dataclasses.replace(t, inputs={k: v for k, v in t.inputs.items()
                                                       if k not in self.forget}) for t in past]
            out.append(Example(past, turn))
        return out

    # -- saving

    def to_dict(self) -> dict:
        return {"version": 1, "function": self.fn.__name__, "signature": self.fn.plan().fingerprint,
                "forget": list(self.forget), "window": self.window,
                "turns": [t.to_dict() for t in self.turns]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: str | Path, fn: Function) -> "Session":
        """A saved session, continued with ``fn`` (same signature)."""
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1:
            raise ValueError(f"unknown session file version {data.get('version')!r}")
        plan = fn.plan()
        if data["signature"] != plan.fingerprint:
            raise ValueError(f"{path}: saved for another signature than {fn.__name__}'s")
        return cls(fn, forget=data["forget"], window=data["window"],
                   turns=[plan.load_turn(t) for t in data["turns"]])
