"""The function is the prompt; calling it calls the model.

lmfn owns three things and nothing else: which model (settings), the loop
(one call, the tool loop, opt-in retries), and what you get back. lmcc
lays out each call and reads each reply; lm15 talks to the provider.
"""

from __future__ import annotations

import contextvars
import dataclasses
import inspect
import json
import typing
from collections.abc import Callable, Iterable

import lm15
import lmcc
import lmcc_lm15
import lmcc_std
from lmcc import core as lmcc_core
from lmcc_std.tools import Tool, ToolCall

from . import models

CONFIG_FIELDS = frozenset(lm15.Config.__dataclass_fields__)
OWN_SETTINGS = frozenset({"model", "capabilities"})

_REGISTRY = lmcc.Registry()
lmcc_std.install(_REGISTRY)

# ---------------------------------------------------------------- settings

_settings: contextvars.ContextVar[dict] = contextvars.ContextVar("lmfn_settings", default={})
_router: lm15.LMRouter | None = None


def _check_settings(kw: dict, where: str) -> dict:
    unknown = set(kw) - CONFIG_FIELDS - OWN_SETTINGS
    if unknown:
        raise TypeError(f"{where}: unknown setting(s) {sorted(unknown)}; settings are 'model', "
                        f"'capabilities' and the lm15 Config fields {sorted(CONFIG_FIELDS)}")
    return dict(kw)


class configure:
    """Set defaults for every function: ``lmfn.configure(model="gpt-4.1-mini")``.
    Used as ``with lmfn.configure(...):`` the change lasts for the block."""

    def __init__(self, **settings):
        new = {**_settings.get(), **_check_settings(settings, "configure")}
        self._token = _settings.set(new)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _settings.reset(self._token)
        return False


def router() -> lm15.LMRouter:
    """The lm15 router lmfn sends through (provider keys from the environment)."""
    global _router
    if _router is None:
        _router = lm15.LMRouter()
    return _router


def use_router(r) -> None:
    """Send through another lm15 router (or anything with ``complete``,
    ``stream`` and ``resolve``), e.g. a fake one in tests."""
    global _router
    _router = r


# ------------------------------------------------------------------ errors


class StepLimit(RuntimeError):
    """The tool loop reached ``max_steps`` without an answer. ``.turn`` is
    the turn so far."""

    def __init__(self, message: str, turn: lmcc.Turn):
        super().__init__(message)
        self.turn = turn


# ------------------------------------------------------------------ result


@dataclasses.dataclass
class CallResult:
    """Everything one call produced. ``value`` is what ``f(...)`` returns."""
    value: object
    values: dict
    repairs: list
    turn: lmcc.Turn
    response: lm15.Response
    responses: list
    attempts: int = 1

    @property
    def usage(self) -> dict:
        """Token counts summed over every model call of this turn."""
        total: dict = {}
        for r in self.responses:
            for k, v in dataclasses.asdict(r.usage).items():
                if isinstance(v, int):
                    total[k] = total.get(k, 0) + v
        return total


# ------------------------------------------------------------------ adapter


def default_adapter() -> lmcc.Adapter:
    """The layout lmfn uses unless given one: tagged sections, turns before
    the input, reasoning and tools chosen by what the model can do."""
    return lmcc.adapter(
        name="lmfn_default",
        messages=[
            lmcc.system("{instruction}\n\nReply in exactly this form:\n"
                        "{% for f in outputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"),
            lmcc.turns(),
            lmcc.user("{% for f in inputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"),
        ],
        transports={
            "reasoning": lmcc.Transport(choose=[
                {"when": {"capability": "native_reasoning"},
                 "use": _REGISTRY.transport("native_reasoning", {"effort": "low"})},
                {"else": _REGISTRY.transport("reasoning_tags", {})}]),
            "tools": lmcc.Transport(choose=[
                {"when": {"capability": "native_function_calling"},
                 "use": _REGISTRY.transport("native_tools", {})},
                {"else": _REGISTRY.transport("fenced_tools", {})}]),
        })


# ------------------------------------------------------------------ signature


def _signature(func, *, reasoning: bool, tools: bool) -> tuple[lmcc.SignatureCore, str | None, type | None]:
    """Lower the function with lmcc, then: a plain return becomes the output
    ``answer``; ``reasoning`` and tools add hidden fields. Returns the
    signature, the name of the single output (None for a dataclass), and
    the dataclass to build (or None)."""
    base = lmcc.fn(func, registry=_REGISTRY).signature
    hints = typing.get_type_hints(func, include_extras=True)
    ret = hints.get("return")
    plain = typing.get_origin(ret) is typing.Annotated or not (
        dataclasses.is_dataclass(ret) and isinstance(ret, type))
    fields = list(base.fields)
    single = None
    if plain:
        name = "answer" if "answer" not in {f.name for f in fields if f.direction == "input"} else "result"
        fields = [dataclasses.replace(f, name=name) if f.direction == "output" else f for f in fields]
        single = name
    extra_in, extra_out = [], []
    if tools:
        extra_in.append(lmcc_core.Field("tools", "input",
                                        lmcc_core.annotation_to_shape(list[Tool], _REGISTRY, field_name="tools"),
                                        type=lmcc_core.typename(list[Tool]), purpose="tools",
                                        annotation=list[Tool]))
        extra_out.append(lmcc_core.Field("calls", "output",
                                         lmcc_core.annotation_to_shape(list[ToolCall], _REGISTRY, field_name="calls"),
                                         type=lmcc_core.typename(list[ToolCall]), purpose="tools.calls",
                                         annotation=list[ToolCall]))
    if reasoning:
        extra_out.insert(0, lmcc_core.Field("reasoning", "output", {"type": "string"}, type="str",
                                            purpose="reasoning", annotation=str))
    inputs = [f for f in fields if f.direction == "input"] + extra_in
    outputs = extra_out + [f for f in fields if f.direction == "output"]
    sig = lmcc_core._validated(lmcc_core.SignatureCore(base.instructions, inputs + outputs))
    return sig, single, (None if plain else ret)


def _tool_spec(fn: Callable) -> Tool:
    t = lm15.derive_tool(fn).tool
    return Tool(t.name, t.description, t.parameters)


# ------------------------------------------------------------------ function


class Function:
    """A typed function whose body is a model call. Build with ``@lmfn.ai``."""

    def __init__(self, func, *, adapter=None, reasoning=False, tools=(), max_steps=8,
                 examples=(), retries=0, tool_errors="report", settings=None):
        if tool_errors not in ("report", "raise"):
            raise TypeError("tool_errors is 'report' or 'raise'")
        self.settings = _check_settings(settings or {}, f"@ai on {func.__name__}")
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func
        self.tools = {t.__name__: t for t in tools}
        self.tool_specs = [_tool_spec(t) for t in tools]
        self.signature, self._single, self._dataclass = _signature(
            func, reasoning=reasoning, tools=bool(tools))
        self.adapter = adapter or default_adapter()
        self.max_steps = max_steps
        self.retries = retries
        self.tool_errors = tool_errors
        self.examples = list(examples)
        self._params = inspect.signature(func)
        self._plans: dict = {}

    # -- configuration

    def using(self, **settings) -> "Function":
        """A copy of this function with other settings; the original is untouched."""
        clone = object.__new__(Function)
        clone.__dict__ = {**self.__dict__, "_plans": {},
                          "settings": {**self.settings, **_check_settings(settings, "using")}}
        return clone

    def _resolved(self) -> dict:
        merged = {**_settings.get(), **self.settings}
        if "model" not in merged:
            raise RuntimeError("no model: call lmfn.configure(model=...) or pass model= to @ai")
        return merged

    def plan(self) -> lmcc.Plan:
        """The lmcc plan for the current model: bound once per model and capabilities."""
        s = self._resolved()
        provider = router().resolve(s["model"]).provider
        caps = {**models.capabilities(provider, router().resolve(s["model"]).model),
                **s.get("capabilities", {})}
        key = (s["model"], json.dumps(caps, sort_keys=True))
        if key not in self._plans:
            self._plans[key] = self.adapter.bind(self.signature, caps, registry=_REGISTRY)
        return self._plans[key]

    # -- inputs and turns

    def _inputs(self, args, kwargs) -> dict:
        bound = self._params.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        if self.tool_specs:
            inputs["tools"] = self.tool_specs
        return inputs

    def _example_turns(self, plan) -> list:
        out = []
        outputs_of = lambda v: (dataclasses.asdict(v) if self._dataclass else {self._single: v})  # noqa: E731
        for ex in self.examples:
            if isinstance(ex, lmcc.Turn):
                out.append(ex)
            elif isinstance(ex, dict):
                out.append(plan.example(ex["inputs"], ex["outputs"]))
            else:
                inp, val = ex
                names = [p for p in self._params.parameters]
                inputs = inp if isinstance(inp, dict) else {names[0]: inp}
                out.append(plan.example(inputs, outputs_of(val)))
        return out

    def _config(self, overrides: dict | None = None) -> lm15.Config | None:
        s = {**self._resolved(), **(overrides or {})}
        fields = {k: v for k, v in s.items() if k in CONFIG_FIELDS}
        return lm15.Config(**fields) if fields else None

    def render(self, *args, turns=(), **kwargs) -> lm15.Request:
        """The exact lm15 request the first model call would send. No network."""
        plan = self.plan()
        current = plan.turn(self._inputs(args, kwargs))
        rendered = plan.render(current, turns=self._example_turns(plan) + list(turns))
        return lmcc_lm15.request(rendered, model=self._resolved()["model"], config=self._config())

    def explain(self) -> str:
        return self.plan().explain()

    # -- the call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs).value

    def call(self, *args, turns=(), **kwargs) -> CallResult:
        """Call the model (and run tools until it answers). Returns everything."""
        plan = self.plan()
        model = self._resolved()["model"]
        past = self._example_turns(plan) + list(turns)
        turn = plan.turn(self._inputs(args, kwargs))
        responses: list = []
        attempts = 0
        for _ in range(self.max_steps):
            rendered = plan.render(turn, turns=past)
            response, reading, attempts = self._complete(plan, rendered, model, responses, attempts)
            turn = turn.with_step(lmcc.ModelStep(reading.values, lmcc_core_message(response),
                                                 lmcc.turn.sha256(rendered.request()), plan.calls_field))
            calls = reading.values.get("calls") or []
            if not calls:
                turn = turn.finish()
                return self._result(turn, reading, response, responses, attempts)
            for call in calls:
                turn = turn.tool(call.id, self._run_tool(call))
        raise StepLimit(f"{self.__name__}: no answer after {self.max_steps} model steps", turn)

    def _complete(self, plan, rendered, model, responses, attempts):
        """One model call; on a parse refusal, up to ``retries`` follow-ups."""
        request = lmcc_lm15.request(rendered, model=model, config=self._config())
        overrides: dict = {}
        for attempt in range(self.retries + 1):
            response = router().complete(request)
            responses.append(response)
            attempts += 1
            try:
                return response, lmcc_lm15.read(plan, response), attempts
            except lmcc.Refusal as err:
                thought = getattr(response.usage, "reasoning_tokens", None) or 0
                if err.code == "parse-truncated" and thought:
                    err = lmcc.Refusal(err.code, f"{err.hint} (the model spent {thought} of its tokens "
                                                 f"thinking first; raise max_tokens)",
                                       fix=err.fix, partial=err.partial)
                if attempt == self.retries or not err.code.startswith("parse-"):
                    raise err
                if err.code == "parse-truncated":
                    current = (self._config(overrides) or lm15.Config()).max_tokens or 1024
                    overrides["max_tokens"] = current * 2
                    request = lmcc_lm15.request(rendered, model=model, config=self._config(overrides))
                else:
                    request = dataclasses.replace(request, messages=request.messages + (
                        response.message,
                        lm15.Message.user(f"Your reply could not be read: {err.hint}. Reply again, "
                                          f"in exactly the form the instructions give."),))
        raise AssertionError("unreachable")

    def _run_tool(self, call: ToolCall):
        fn = self.tools.get(call.name)
        if fn is None:
            if self.tool_errors == "raise":
                raise KeyError(f"the model called unknown tool {call.name!r}")
            return f"error: there is no tool named {call.name!r}"
        try:
            out = fn(**(call.input or {}))
        except Exception as exc:  # noqa: BLE001 — reported to the model unless asked to raise
            if self.tool_errors == "raise":
                raise
            return f"error: {type(exc).__name__}: {exc}"
        return out if isinstance(out, str) else json.dumps(out, default=str, ensure_ascii=False)

    def _result(self, turn, reading, response, responses, attempts) -> CallResult:
        values = {k: v for k, v in turn.outputs.items() if k != "calls"}
        if self._dataclass is not None:
            names = {f.name for f in dataclasses.fields(self._dataclass)}
            value = self._dataclass(**{k: v for k, v in values.items() if k in names})
        else:
            value = values[self._single]
        return CallResult(value, values, reading.repairs, turn, response, responses, attempts)

    # -- streaming

    def stream(self, *args, turns=(), **kwargs) -> "Streamed":
        """Stream the reply field by field (no tools). Iterate for lmcc stream
        events; ``.result`` is the CallResult at the end."""
        if self.tools:
            raise TypeError(f"{self.__name__}: streaming a tool loop is not supported yet; use call()")
        return Streamed(self, self._inputs(args, kwargs), list(turns))


class Streamed:
    def __init__(self, fn: Function, inputs: dict, turns: list):
        self.fn, self.inputs, self.turns = fn, inputs, turns
        self.result: CallResult | None = None

    def __iter__(self):
        fn = self.fn
        plan = fn.plan()
        turn = plan.turn(self.inputs)
        rendered = plan.render(turn, turns=fn._example_turns(plan) + self.turns)
        request = lmcc_lm15.request(rendered, model=fn._resolved()["model"], config=fn._config())
        s = plan.stream()
        parts: list = []
        end = None
        for event in router().stream(request):
            if isinstance(event, lm15.StreamDeltaEvent):
                delta = lm15.serde.delta_to_dict(event.delta)
                parts.append(delta)
                yield from s.feed(delta)
            elif isinstance(event, lm15.StreamEndEvent):
                end = event
        done = s.finish(end.finish_reason if end else None)
        yield from done.events
        message = {"role": "assistant", "parts": _coalesce(parts)}
        turn = turn.with_step(lmcc.ModelStep(done.values, message, lmcc.turn.sha256(rendered.request()),
                                             plan.calls_field)).finish()
        self.result = fn._result(turn, lmcc.Reading(done.values, done.repairs), None, [], 1)


def _coalesce(deltas: list) -> list:
    out: list = []
    for d in deltas:
        if out and out[-1].get("type") == d.get("type") and isinstance(d.get("text"), str):
            out[-1] = {**out[-1], "text": out[-1].get("text", "") + d["text"]}
        else:
            out.append({k: v for k, v in d.items() if k != "part_index"})
    return out


def lmcc_core_message(response: lm15.Response) -> dict:
    return lm15.serde.message_to_dict(response.message)


# ------------------------------------------------------------------ decorator


def ai(func=None, *, model=None, adapter=None, reasoning=False, tools=(), max_steps=8,
       examples=(), retries=0, tool_errors="report", capabilities=None, **settings):
    """``@lmfn.ai`` — the parameters are the inputs, the return type is the
    output, the docstring is the instruction. Options: ``model``, lm15
    Config fields (``temperature``, ``max_tokens``, ...), ``reasoning``,
    ``tools``, ``max_steps``, ``examples``, ``retries``, ``tool_errors``,
    ``adapter``, ``capabilities``."""
    if model is not None:
        settings["model"] = model
    if capabilities is not None:
        settings["capabilities"] = capabilities

    def wrap(f):
        return Function(f, adapter=adapter, reasoning=reasoning, tools=tools, max_steps=max_steps,
                        examples=examples, retries=retries, tool_errors=tool_errors,
                        settings=settings)
    return wrap if func is None else wrap(func)
