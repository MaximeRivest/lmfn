"""The verifiers harness, offline: units, then a real vf-eval run against a
fake model server (tests/fake_openai.py). Skipped when verifiers or the
example package is not installed."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1")
pytest.importorskip("alphabet_sort_lmfn")

import lmcc  # noqa: E402

import lmfn_verifiers as lv  # noqa: E402
from alphabet_sort_lmfn.program import ADAPTERS, Entry, sort_names  # noqa: E402

HERE = Path(__file__).parent


def config(**kw):
    return lv.LmfnHarnessConfig(id="lmfn-verifiers", program="alphabet_sort_lmfn.program:sort_names", **kw)


def test_the_program_runs_with_its_named_adapter_replayed_verbatim():
    fn = lv.load_program(config(adapter="json"))
    assert fn.adapter.name == "alphabet_json" and fn.adapter.replay == "verbatim"


def test_an_adapter_can_be_an_lmcc_json_file(tmp_path):
    from lmfn.core import _REGISTRY
    data_only = lmcc.adapter(name="alphabet_std_json", messages=[
        lmcc.system("{instruction}\n\nReply with one line:\nANSWER: {answer}"), lmcc.turns(),
        lmcc.user("Sort by {by} name. {% if first_turn %}Names{% else %}New names{% endif %}: {names}")],
        formats={"list[str]": "json", "list[Entry]": "json"})
    path = tmp_path / "a.json"
    path.write_text(json.dumps(data_only.dump(registry=_REGISTRY)))
    fn = lv.load_program(config(adapter=str(path)))
    assert fn.adapter.name == "alphabet_std_json" and fn.adapter.replay == "verbatim"
    import lmfn
    with lmfn.configure(model="m", router=_Router()):
        plan = fn.plan()
    assert plan.parse('ANSWER: [{"name": "AnnJones", "new": false}]') == {"answer": [Entry("AnnJones", False)]}


class _Router:
    def resolve(self, model):
        return lv._Resolution("verifiers", model)


def test_unknown_adapter_names_are_listed():
    with pytest.raises(KeyError, match="original"):
        lv.load_program(config(adapter="nope"))


def test_inputs_travel_in_the_user_turn_and_are_lifted():
    msg = lv.user_turn(names=["AnnJones"], by="LAST", first_turn=True)
    assert lv.decode_inputs(sort_names, msg) == {"names": ["AnnJones"], "by": "LAST", "first_turn": True}
    with pytest.raises(ValueError, match="user_turn"):
        lv.decode_inputs(sort_names, "please sort AnnJones")


def test_plain_text_works_for_a_function_with_one_text_input():
    import lmfn

    @lmfn.ai
    def chat(message: str) -> str:
        """Chat."""
    assert lv.decode_inputs(chat, "hello") == {"message": "hello"}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def fake_model():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "fake_openai.py"), str(port)])
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/v1"
    proc.terminate()


@pytest.mark.parametrize("adapter", ["original", "tags", "json"])
def test_vf_eval_rollouts_are_one_training_path_each(fake_model, adapter, tmp_path):
    from verifiers.v1.trace import Trace
    vf_eval = Path(sys.executable).parent / "vf-eval"
    run = subprocess.run(
        [str(vf_eval), "alphabet-sort-lmfn", "--env.agent.harness.id", "lmfn-verifiers",
         "--env.agent.harness.program", "alphabet_sort_lmfn.program:sort_names",
         "--env.agent.harness.adapter", adapter, "--model", "fake",
         "--client.base-url", fake_model, "--client.api-key-var", "FAKE_KEY",
         "--num-tasks", "30", "--no-push", "--no-rich", "--output-dir", str(tmp_path)],
        env={**os.environ, "FAKE_KEY": "x"}, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    rows = [json.loads(line) for line in next(tmp_path.rglob("traces.jsonl")).read_text().splitlines()]
    assert len(rows) == 30 and all(r["ok"] for r in rows)
    traces = [r["traces"][0] for r in rows]
    # verbatim replay keeps every multi-turn conversation one exact-prefix path
    assert all(len(Trace.model_validate(t).branches) == 1 for t in traces)
    recorded = [x for t in traces for x in t["info"]["lmfn"]["turns"]]
    assert len(recorded) == sum(r["task"]["data"]["info"]["num_turns"] for r in rows)
    assert any("refusal" in x for x in recorded)           # the fake's nonsense replies
    for row, t in zip(rows, traces):                      # the fake sorts right: 1.0 unless unreadable
        turns = t["info"]["lmfn"]["turns"]
        expected = sum("refusal" not in x for x in turns) / row["task"]["data"]["info"]["num_turns"]
        assert abs(t["rewards"]["alphabet_sort"]["score"] - expected) < 1e-9
