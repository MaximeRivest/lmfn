"""alphabet-sort, as one lmfn function: the verifiers taskset.

The same episodes as verifiers' `alphabet-sort` (same dataset, seed and
random draws, so the same names, splits and orders), but each turn is the
function's inputs, not a prompt: the scripted user sends them with
`user_turn`, the adapter chosen on the harness writes the prompt, and the
reward scores the values the adapter read back.

    vf-eval alphabet-sort-lmfn \
        --env.agent.harness.id lmfn-verifiers \
        --env.agent.harness.program alphabet_sort_lmfn.program:sort_names \
        --env.agent.harness.adapter original      # or tags, json
"""

from __future__ import annotations

import difflib
import random
from collections.abc import Iterator
from typing import Literal

from datasets import load_dataset

import verifiers.v1 as vf
from lmfn_verifiers import turns, user_turn

from .program import NEW, sort_names

DATASET = "kalomaze/alphabetic-arxiv-authors-it1"
SEED = 1337420


class AlphabetSortTaskConfig(vf.TaskConfig):
    similarity_power: int = 4
    power_per_turn: bool = True


class AlphabetSortConfig(vf.TasksetConfig):
    min_turns: int = 1
    max_turns: int = 3
    min_names_per_turn: int = 1
    max_names_per_turn: int = 5
    split: Literal["train"] = "train"
    task: AlphabetSortTaskConfig = AlphabetSortTaskConfig()


class AlphabetSortTaskData(vf.TaskData):
    info: dict
    """`turn_inputs` (the function's inputs, one dict per turn), `ground_truths`
    (per turn, `[{name, new}]`) and `num_turns`."""


def _line(name: str, new: bool) -> str:
    return (f"{name} {NEW}" if new else name).strip().lower()


class AlphabetSortTask(vf.Task[AlphabetSortTaskData, vf.State, AlphabetSortTaskConfig]):
    @vf.reward(weight=1.0)
    async def alphabet_sort(self, trace: vf.Trace) -> float:
        """The original's metric (power-scaled sequence similarity per turn),
        computed on the values the adapter read instead of on regex matches.
        An unreadable turn scores 0, as a reply without the tag did."""
        truths = self.data.info["ground_truths"]
        num_turns = self.data.info["num_turns"]
        recorded = turns(trace, sort_names)
        power = self.config.similarity_power
        scores = []
        for t in range(num_turns):
            expected = "\n".join(_line(e["name"], e["new"]) for e in truths[t])
            got = recorded[t]["outputs"].get("answer") if t < len(recorded) else None
            if not got:
                scores.append(0.0)
                continue
            pred = "\n".join(_line(e.name, e.new) for e in got)
            sim = difflib.SequenceMatcher(None, pred, expected).ratio()
            scores.append(sim ** power if self.config.power_per_turn else sim)
        avg = sum(scores) / num_turns if num_turns else 0.0
        return avg if self.config.power_per_turn else avg ** power

    @vf.metric
    async def format_ok(self, trace: vf.Trace) -> float:
        """Share of turns read with no repair and no refusal."""
        recorded = turns(trace)
        if not recorded:
            return 0.0
        clean = [not t.get("refusal") and not [r for r in t["repairs"] if r["repair"] in ("marker", "value", "unclosed")]
                 for t in recorded]
        return sum(clean) / len(recorded)


class AlphabetSortEnv(vf.SingleAgentEnv):
    """The scripted user: one turn per episode step, sending the inputs."""

    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            for inputs in task.data.info["turn_inputs"]:
                if (await interaction.turn(user_turn(**inputs))).terminated:
                    break


class AlphabetSortTaskset(vf.Taskset[AlphabetSortTask, AlphabetSortConfig]):
    INFINITE = True

    def load(self) -> Iterator[AlphabetSortTask]:
        c = self.config
        assert 1 <= c.min_turns <= c.max_turns
        assert 1 <= c.min_names_per_turn <= c.max_names_per_turn
        rng = random.Random(SEED)
        entries = load_dataset(DATASET, split=c.split)
        idx = 0
        while True:
            pass_start = idx
            for entry in entries:
                names = list(dict.fromkeys(n.replace(" ", "") for n in entry["names"]))
                counts = [rng.randint(c.min_names_per_turn, c.max_names_per_turn)
                          for _ in range(rng.randint(c.min_turns, c.max_turns))]
                if len(names) < sum(counts):
                    continue
                by_first = rng.choice([True, False])

                def sort_key(s: str, by_first: bool = by_first) -> str:
                    cut = next((i for i in range(1, len(s)) if s[i].isupper()), len(s))
                    return s[:cut] if by_first else s[cut:]

                turns_, cumulative, truths, i = [], [], [], 0
                for count in counts:
                    turn = names[i: i + count]
                    i += count
                    turns_.append(turn)
                    cumulative += turn
                    ranked = sorted(cumulative, key=sort_key)
                    truths.append([{"name": x, "new": len(turns_) > 1 and x in turn} for x in ranked])

                # the same random draws as the original, in the same order, so the
                # episodes match; the draws that only shaped prompt text are unused
                first = turns_[0][:]
                rng.shuffle(first)
                rng.randint(c.min_names_per_turn, c.max_names_per_turn)        # example length
                shown_names = [first]
                for t in range(1, len(turns_)):
                    shuffled = turns_[t][:]
                    rng.shuffle(shuffled)
                    shown = rng.randint(c.min_names_per_turn, sum(len(x) for x in turns_[: t + 1]))
                    rng.randint(0, shown - 1)                                  # example marks
                    shown_names.append(shuffled)

                by = "FIRST" if by_first else "LAST"
                yield AlphabetSortTask(
                    AlphabetSortTaskData(
                        idx=idx, prompt=None,
                        info={"turn_inputs": [{"names": n, "by": by} for n in shown_names],
                              "ground_truths": truths, "num_turns": len(turns_)},
                    ),
                    c.task,
                )
                idx += 1
            if idx == pass_start:
                raise ValueError("no source name list is long enough for the configured turns")
