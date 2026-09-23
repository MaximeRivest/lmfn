"""Sessions offline: the scripted router from test_lmfn."""

import pytest

import lmfn
from test_lmfn import fake


@pytest.fixture(autouse=True)
def model():
    with lmfn.configure(model="gpt-4.1-mini"):
        yield


@lmfn.ai
def chat(message: str) -> str:
    """Chat."""


@lmfn.ai
def rag(question: str, context: str) -> str:
    """Answer from the context."""


def say(*texts):
    return [f"<answer>\n{t}\n</answer>" for t in texts]


def test_a_session_remembers():
    r = fake(*say("Hi Ana.", "Ana."))
    s = lmfn.Session(chat)
    s("I am Ana.")
    assert s("My name?") == "Ana." and len(s) == 2
    assert [m.role for m in r.requests[1].messages] == ["user", "assistant", "user"]


def test_forgotten_inputs_leave_past_turns_but_not_the_current_call():
    r = fake(*say("Paris.", "2.1 million."))
    s = lmfn.Session(rag, forget=["context"])
    s("Capital of France?", context="BIG DOCUMENT ONE")
    s("Population?", context="BIG DOCUMENT TWO")
    sent = "\n".join(p.text for m in r.requests[1].messages for p in m.parts if p.type == "text")
    assert "BIG DOCUMENT ONE" not in sent and "BIG DOCUMENT TWO" in sent
    assert s.turns[0].inputs["context"] == "BIG DOCUMENT ONE"   # kept in the record


def test_a_window_sends_only_recent_turns():
    r = fake(*say("a", "b", "c"))
    s = lmfn.Session(chat, window=1)
    s("1"), s("2"), s("3")
    assert len(r.requests[2].messages) == 3


def test_undo_fork_add_reset():
    fake(*say("x", "y", "z"))
    s = lmfn.Session(chat)
    s("1"), s("2")
    assert [t.outputs["answer"] for t in s.undo()] == ["y"]
    f = s.fork()
    f("3")
    assert len(s) == 1 and len(f) == 2
    s.add({"message": "gold"}, {"answer": "gold answer"})
    assert s.turns[-1].outputs == {"answer": "gold answer"}
    s.reset()
    assert len(s) == 0


def test_save_and_load_continue_the_conversation(tmp_path):
    fake(*say("Hi.", "Still here."))
    s = lmfn.Session(chat, forget=[])
    s("hello")
    s.save(tmp_path / "s.json")
    loaded = lmfn.Session.load(tmp_path / "s.json", chat)
    assert loaded.turns == s.turns
    assert loaded("again?") == "Still here."


def test_load_refuses_another_signature(tmp_path):
    fake(*say("Hi."))
    s = lmfn.Session(chat)
    s("hello")
    s.save(tmp_path / "s.json")
    with pytest.raises(ValueError):
        lmfn.Session.load(tmp_path / "s.json", rag)


def test_score_and_examples():
    fake(*say("good", "bad", "good"))
    s = lmfn.Session(chat)
    s("a"), s("b"), s("c")
    s.score(lambda t: 1.0 if t.outputs["answer"] == "good" else 0.0)
    kept = s.examples(min_score=0.5)
    assert [e.outputs["answer"] for e in kept] == ["good", "good"]
    assert [len(e.past) for e in kept] == [0, 2]
    assert len(s.examples(min_score=0.5, stop_at_bad=True)) == 1


def test_forget_names_must_be_inputs():
    with pytest.raises(TypeError):
        lmfn.Session(chat, forget=["context"])
