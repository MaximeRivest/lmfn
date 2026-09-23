"""A session against real models: old context forgotten, save/load. Costs cents."""
import sys, tempfile, os
import lmfn


@lmfn.ai(max_tokens=2000)
def rag(question: str, context: str) -> str:
    """Answer the question in one short sentence, from the context and the conversation."""


for model in sys.argv[1:] or ["gpt-4.1-mini", "claude-haiku-4-5"]:
    with lmfn.configure(model=model):
        s = lmfn.Session(rag, forget=["context"])
        a1 = s("Who founded the bakery?", context="Le Pain Doré was founded in 1952 by Marie Tremblay in Québec City.")
        a2 = s("And in what year?", context="(no new documents)")
        path = os.path.join(tempfile.mkdtemp(), "s.json")
        s.save(path)
        s2 = lmfn.Session.load(path, rag)
        a3 = s2("Repeat the founder's name only.", context="(no new documents)")
        sent = s2.render("x", context="y")
        assert "1952 by Marie" not in str(sent.messages), "forgotten context was sent"
        # a1 is what the model remembers; the year survives only if a1 said it:
        # that is what forgetting the context costs, by design
        ok = "Marie" in a1 and "Marie" in a3
        year = "remembered" if "1952" in a2 else ("lost: the first answer did not state it" if "1952" not in a1
                                                   else "LOST DESPITE BEING IN THE ANSWER")
        print(("OK  " if ok else "FAIL"), model, "| year:", year, "|", a1, "|", a2, "|", a3)
