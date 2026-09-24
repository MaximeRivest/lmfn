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


def _v1_cli() -> bool:
    """vf-eval with the v1 flags (verifiers 0.3.2 dev); 0.3.1 ships the older CLI."""
    exe = Path(sys.executable).parent / "vf-eval"
    try:
        out = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--env.agent" in out.stdout + out.stderr or "harness" in out.stdout


needs_v1_cli = pytest.mark.skipif(not _v1_cli(), reason="needs the verifiers v1 vf-eval CLI")


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


@needs_v1_cli
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


@needs_v1_cli
@pytest.mark.parametrize("mode", ["drop", "keep"])
def test_a_thinking_teachers_reasoning_is_dropped_before_the_trace(fake_model, mode, tmp_path):
    pytest.importorskip("banking77_lmfn")
    from verifiers.v1.trace import Trace
    vf_eval = Path(sys.executable).parent / "vf-eval"
    run = subprocess.run(
        [str(vf_eval), "banking77-lmfn", "--env.agent.harness.id", "lmfn-verifiers",
         "--env.agent.harness.program", "banking77_lmfn.program:classify",
         "--env.agent.harness.adapter", "compact", "--env.taskset.task.teacher-reasoning", mode,
         "--model", "fake", "--client.base-url", fake_model, "--client.api-key-var", "FAKE_KEY",
         "--num-tasks", "3", "--no-push", "--no-rich", "--output-dir", str(tmp_path)],
        env={**os.environ, "FAKE_KEY": "x"}, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    for line in next(tmp_path.rglob("traces.jsonl")).read_text().splitlines():
        t = json.loads(line)["traces"][0]
        reply = [m for m in Trace.model_validate(t).branches[0].messages if m.role == "assistant"][0]
        assert reply.content == "Intent: card_arrival"
        assert (reply.reasoning_content is None) == (mode == "drop")


def test_strip_reasoning_removes_an_inline_think_block():
    from verifiers.v1.types import AssistantMessage, Response
    def response(content):
        return Response(id="r", created=0, model="m", finish_reason="stop",
                        message=AssistantMessage(content=content))
    assert lv.strip_reasoning(response("<think>hm</think>\nIntent: x")).message.content == "Intent: x"
    assert lv.strip_reasoning(response("Intent: x")) is None


@needs_v1_cli
def test_teacher_traces_export_as_sft_rows_without_reasoning(fake_model, tmp_path):
    pytest.importorskip("banking77_lmfn")
    vf_eval = Path(sys.executable).parent / "vf-eval"
    subprocess.run(
        [str(vf_eval), "banking77-lmfn", "--env.agent.harness.id", "lmfn-verifiers",
         "--env.agent.harness.program", "banking77_lmfn.program:classify",
         "--env.agent.harness.adapter", "compact", "--model", "fake",
         "--client.base-url", fake_model, "--client.api-key-var", "FAKE_KEY",
         "--num-tasks", "3", "--no-push", "--no-rich", "--output-dir", str(tmp_path)],
        env={**os.environ, "FAKE_KEY": "x"}, capture_output=True, text=True, timeout=300, check=True)
    rows = lv.sft_rows(next(tmp_path.rglob("traces.jsonl")))
    assert len(rows) == 3
    for row in rows:
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        assert row["completion"] == [{"role": "assistant", "content": "Intent: card_arrival"}]
