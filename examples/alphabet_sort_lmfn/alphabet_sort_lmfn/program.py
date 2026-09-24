"""The task, as one function, and the ways of spelling it (lmcc adapters).

The function is what the task *is*: given this turn's new names and the
sort key, return the whole sorted list so far, marking the names that are
new this turn. The adapters are how it is *said* to a model; each one is
bound to the same function and read back into the same values, so the
reward never sees text.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Literal

import lmcc
from lmcc import core
from lmcc.turn import lift

import lmfn

NEW = "// new name!"


@dataclasses.dataclass
class Entry:
    name: str
    new: bool


@lmfn.ai
def sort_names(names: list[str], by: Literal["FIRST", "LAST"], first_turn: bool) -> list[Entry]:
    """Keep an alphabetically sorted list of every name given so far, sorted by
    first or last name. After the first list, mark the names that are new."""


# ------------------------------------------------------------------ formats
#
# How the pieces are spelled. Inputs are a comma list; the answer is one of
# three spellings. Each reads back what it writes.

comma_list = lmcc.make_format(write=lambda names: ", ".join(names), direction="in",
                              describe=lambda: "names separated by commas")


def _write_lines(entries) -> str:
    return "\n".join(e.name + (f" {NEW}" if e.new else "") for e in entries)


def _read_lines(capture, field):
    out = []
    for line in capture.text.split("\n"):
        line = core.strip(line)
        if not line:
            continue
        new = line.lower().endswith(NEW)
        out.append({"name": core.strip(line[: -len(NEW)]) if new else line, "new": new})
    return lift(field.annotation, out)


# The example shows plain names, as the original's first-turn example does:
# showing the mark in every prompt made gpt-4.1-mini mark every name new on the
# first turn, and copy the example into the tag (live run, 2026-09-23).
lines = lmcc.make_format(write=_write_lines, read=_read_lines, describe=lambda: "Name1\nName2\n...")


def _write_json(entries) -> str:
    return json.dumps([{"name": e.name, "new": e.new} for e in entries], ensure_ascii=False)


def _read_json(capture, field):
    text = core.strip(capture.text)
    if text.startswith("```"):
        text = core.strip(text.strip("`").removeprefix("json"))
    return lift(field.annotation, json.loads(text))


as_json = lmcc.make_format(write=_write_json, read=_read_json,
                           describe=lambda: '[{"name": "...", "new": false}, ...]')

INPUT_FORMATS = {"list[str]": comma_list}


# ------------------------------------------------------------------ adapters

ADAPTERS = {
    # The original environment's wording, first turn and follow-ups, as user
    # messages only. Differences from the original, stated: one tag on every
    # turn (the original uses `alphabetical_sorted` on the first; a reply
    # layout that changes with the turn is what lmcc forbids, §4), the format
    # example on every turn, and no randomized example length.
    "original": lmcc.adapter(name="alphabet_original", messages=[
        lmcc.turns(),
        lmcc.user("{% if first_turn %}Sort these names in alphabetical order by {by} name: {names}"
                  "{% else %}Now sort ALL of these names alphabetically by {by} name: {names}\n\n"
                  "These are in addition to the prior list. Mark any NEW names (that weren't "
                  f"in the prior list) with `{NEW}` at the end.{{% endif %}}\n\n"
                  "Use exactly this format:\n"
                  "{% for f in outputs %}<combined_alphabetical_sorted>\n{f.value}\n"
                  "</combined_alphabetical_sorted>{% endfor %}"),
    ], formats={**INPUT_FORMATS, "list[Entry]": lines}),

    # lmfn's house style: the instruction in the system message, tags per field.
    "tags": lmcc.adapter(name="alphabet_tags", messages=[
        lmcc.system("{instruction}\n\nOne name per line; end a new name's line with "
                    f"`{NEW}`.\n\nReply in exactly this form:\n"
                    "{% for f in outputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"),
        lmcc.turns(),
        lmcc.user("Sort by: {by}\n{% if first_turn %}Names{% else %}New names{% endif %}: {names}"),
    ], formats={**INPUT_FORMATS, "list[Entry]": lines}),

    # The answer as JSON after a label.
    "json": lmcc.adapter(name="alphabet_json", messages=[
        lmcc.system("{instruction}\n\nReply with one line:\nANSWER: {answer}"),
        lmcc.turns(),
        lmcc.user("Sort by {by} name. {% if first_turn %}Names{% else %}New names{% endif %}: {names}"),
    ], formats={**INPUT_FORMATS, "list[Entry]": as_json}),
}
