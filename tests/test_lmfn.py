"""lmfn offline: a scripted router stands in for the provider."""

import dataclasses
import enum

import lm15
import pytest

import lmcc
import lmfn


class Resolution:
    def __init__(self, provider, model):
        self.provider, self.model = provider, model


class FakeRouter:
    """Replies in order; records every request."""

    def __init__(self, *replies, provider="openai"):
        self.replies = list(replies)
        self.requests = []
        self.provider = provider

    def resolve(self, model):
        return Resolution(self.provider, model)

    def complete(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, lm15.Response):
            return reply
        finish = "stop"
        if isinstance(reply, tuple):
            reply, finish = reply
        parts = reply if isinstance(reply, list) else [lm15.TextPart(reply)]
        return lm15.Response(id="r", model=request.model, message=lm15.Message.assistant(parts),
                             finish_reason=finish, usage=lm15.Usage(input_tokens=3, output_tokens=2,
                                                                    total_tokens=5))

    def stream(self, request):
        self.requests.append(request)
        text = self.replies.pop(0)
        yield lm15.StreamStartEvent(id="r", model=request.model)
        for i in range(0, len(text), 3):
            yield lm15.StreamDeltaEvent(delta=lm15.TextDelta(text=text[i:i + 3], part_index=0))
        yield lm15.StreamEndEvent(finish_reason="stop")


@pytest.fixture(autouse=True)
def model():
    with lmfn.configure(model="gpt-4.1-mini"):
        yield


def fake(*replies, provider="openai"):
    r = FakeRouter(*replies, provider=provider)
    lmfn.use_router(r)
    return r


# ------------------------------------------------------------- one call


def test_a_function_calls_the_model_and_returns_a_str():
    @lmfn.ai
    def summarize(text: str) -> str:
        """Summarize the text in one sentence."""
    r = fake("<answer>\nShort.\n</answer>")
    assert summarize("long text") == "Short."
    request = r.requests[0]
    assert request.model == "gpt-4.1-mini"
    assert "Summarize the text in one sentence." in request.system
    assert "<text>\nlong text\n</text>" in request.messages[0].parts[0].text


class Sentiment(enum.Enum):
    positive = "positive"
    negative = "negative"


@dataclasses.dataclass
class Review:
    sentiment: Sentiment
    stars: int


def test_a_dataclass_return_is_several_outputs_built_back():
    @lmfn.ai
    def read_review(text: str) -> Review:
        """Read the review."""
    fake("<sentiment>\nPositive\n</sentiment>\n<stars>\n4.\n</stars>")
    res = read_review.call("tasty")
    assert res.value == Review(Sentiment.positive, 4)
    assert [r["repair"] for r in res.repairs] == ["value", "value"]
    assert res.usage["total_tokens"] == 5


def test_an_input_named_answer_moves_the_output_to_result():
    @lmfn.ai
    def grade(answer: str) -> int:
        """Grade it."""
    r = fake("<result>\n7\n</result>")
    assert grade("x") == 7 and "<result>" in r.requests[0].system


# ------------------------------------------------------------- reasoning


def test_reasoning_uses_tags_on_a_model_without_thinking():
    @lmfn.ai(reasoning=True)
    def solve(problem: str) -> int:
        """Solve it."""
    r = fake("<think>7*3=21, 50-21=29</think>\n<answer>\n29\n</answer>")
    res = solve.call("pens")
    assert res.value == 29 and res.values["reasoning"] == "7*3=21, 50-21=29"
    assert "<think>" in r.requests[0].system and r.requests[0].config.reasoning is None


def test_reasoning_uses_the_native_channel_where_the_model_has_it():
    @lmfn.ai(reasoning=True, model="claude-sonnet-4-5")
    def solve(problem: str) -> int:
        """Solve it."""
    r = fake([lm15.ThinkingPart("7*3=21"), lm15.TextPart("<answer>\n29\n</answer>")], provider="anthropic")
    res = solve.call("pens")
    assert res.values == {"reasoning": "7*3=21", "answer": 29}
    assert r.requests[0].config.reasoning is not None and "<think>" not in r.requests[0].system


# ------------------------------------------------------------- tools


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"Sunny and 22C in {city}."


def test_the_tool_loop_with_native_calls():
    @lmfn.ai(tools=[get_weather])
    def assistant(question: str) -> str:
        """Answer; use a tool when you need facts."""
    r = fake([lm15.ToolCallPart(id="c1", name="get_weather", input={"city": "Montreal"})],
             "<answer>\nSunny, 22C.\n</answer>")
    res = assistant.call("weather?")
    assert res.value == "Sunny, 22C."
    assert [s.kind for s in res.turn.steps] == ["model", "tool", "model"]
    assert res.turn.steps[1].output[0]["text"] == "Sunny and 22C in Montreal."
    assert r.requests[0].tools and r.requests[0].tools[0].name == "get_weather"
    assert any(p.type == "tool_result" for m in r.requests[1].messages for p in m.parts)


def test_the_tool_loop_with_text_calls_on_a_model_without_native_tools():
    @lmfn.ai(tools=[get_weather], model="openrouter:qwen/qwen-2.5-7b-instruct")
    def assistant(question: str) -> str:
        """Answer; use a tool when you need facts."""
    r = fake('```tool\n{"name": "get_weather", "input": {"city": "Paris"}}\n```',
             "<answer>\nSunny.\n</answer>", provider="openrouter")
    assert assistant("weather?") == "Sunny."
    assert not r.requests[0].tools and "get_weather" in r.requests[0].system


def test_a_failing_tool_is_reported_to_the_model():
    def boom(x: str) -> str:
        """Always fails."""
        raise ValueError("nope")

    @lmfn.ai(tools=[boom])
    def f(q: str) -> str:
        """Do."""
    fake([lm15.ToolCallPart(id="c1", name="boom", input={"x": "a"})], "<answer>\nok\n</answer>")
    res = f.call("q")
    assert res.turn.steps[1].output[0]["text"] == "error: ValueError: nope"


def test_the_step_limit_stops_a_loop_and_keeps_the_turn():
    @lmfn.ai(tools=[get_weather], max_steps=2)
    def f(q: str) -> str:
        """Do."""
    call = [lm15.ToolCallPart(id="c1", name="get_weather", input={"city": "X"})]
    fake(call, [lm15.ToolCallPart(id="c2", name="get_weather", input={"city": "Y"})])
    with pytest.raises(lmfn.StepLimit) as err:
        f("q")
    assert len(err.value.turn.steps) == 4


# ------------------------------------------------------------- settings


def test_settings_innermost_wins():
    @lmfn.ai(temperature=0.1)
    def f(q: str) -> str:
        """Do."""
    r = fake(*["<answer>\nx\n</answer>"] * 3)
    f("a")
    with lmfn.configure(model="gpt-4.1", temperature=0.9, max_tokens=50):
        f("b")
    f.using(temperature=0.5)("c")
    assert [(q.model, q.config.temperature) for q in r.requests] == \
        [("gpt-4.1-mini", 0.1), ("gpt-4.1", 0.1), ("gpt-4.1-mini", 0.5)]
    assert r.requests[1].config.max_tokens == 50 and f.settings == {"temperature": 0.1}


def test_an_unknown_setting_is_a_type_error():
    with pytest.raises(TypeError):
        lmfn.ai(temprature=0.1)(lambda q: q)
    with pytest.raises(TypeError):
        lmfn.configure(modle="x")


def test_render_shows_the_request_without_calling():
    @lmfn.ai
    def f(q: str) -> str:
        """Do."""
    r = fake()
    req = f.render("hello")
    assert isinstance(req, lm15.Request) and r.requests == []


# ------------------------------------------------------------- examples, turns


def test_examples_become_example_turns():
    @lmfn.ai(examples=[("I love it", Sentiment.positive)])
    def classify(text: str) -> Sentiment:
        """Classify."""
    r = fake("<answer>\nnegative\n</answer>")
    assert classify("awful") == Sentiment.negative
    msgs = r.requests[0].messages
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1].parts[0].text == "<answer>\npositive\n</answer>"


def test_a_past_turn_continues_the_conversation():
    @lmfn.ai
    def chat(message: str) -> str:
        """Chat."""
    r = fake("<answer>\nHi Ana.\n</answer>", "<answer>\nAna.\n</answer>")
    first = chat.call("I am Ana.")
    assert chat.call("My name?", turns=[first.turn]).value == "Ana."
    assert [m.role for m in r.requests[1].messages] == ["user", "assistant", "user"]


# ------------------------------------------------------------- when the reply is wrong


def test_a_bad_reply_raises_the_refusal_by_default():
    @lmfn.ai
    def f(q: str) -> int:
        """Do."""
    fake("I think it is 4")
    with pytest.raises(lmcc.Refusal) as err:
        f("q")
    assert err.value.code == "parse-missing-fields"


def test_retries_send_the_hint_back_once():
    @lmfn.ai(retries=1)
    def f(q: str) -> int:
        """Do."""
    r = fake("I think it is 4", "<answer>\n4\n</answer>")
    res = f.call("q")
    assert res.value == 4 and res.attempts == 2
    follow_up = r.requests[1].messages[-1].parts[0].text
    assert follow_up.startswith("Your reply could not be read: reply is missing")


def test_a_truncated_reply_retries_with_more_tokens():
    @lmfn.ai(retries=1, max_tokens=100)
    def f(q: str) -> str:
        """Do."""
    r = fake(("<answer>\nThe cap", "length"), "<answer>\nParis\n</answer>")
    assert f("q") == "Paris"
    assert [q.config.max_tokens for q in r.requests] == [100, 200]


# ------------------------------------------------------------- streaming


def test_streaming_yields_field_events_and_a_result():
    @lmfn.ai
    def f(q: str) -> str:
        """Do."""
    fake("<answer>\nstreamed text\n</answer>")
    s = f.stream("q")
    events = list(s)
    assert "".join(e["text"] for e in events if e["kind"] == "field_delta") == "streamed text"
    assert s.result.value == "streamed text"


def test_a_cut_reply_says_when_thinking_used_the_budget():
    @lmfn.ai(max_tokens=100)
    def f(q: str) -> str:
        """Do."""
    cut = lm15.Response(id="r", model="m", message=lm15.Message.assistant([lm15.TextPart("<answer>\nPar")]),
                        finish_reason="length",
                        usage=lm15.Usage(input_tokens=1, output_tokens=100, total_tokens=101, reasoning_tokens=90))
    fake(cut)
    with pytest.raises(lmcc.Refusal) as err:
        f("q")
    assert err.value.code == "parse-truncated" and "spent 90 of its tokens thinking" in err.value.hint


def test_openai_models_do_not_get_stop_sequences():
    from lmfn import models
    assert models.capabilities("openai", "gpt-4.1-mini")["stop_sequences"] is False
    assert models.capabilities("anthropic", "claude-haiku-4-5")["stop_sequences"] is True
