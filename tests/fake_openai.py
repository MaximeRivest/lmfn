"""A fake OpenAI Chat Completions server that plays alphabet-sort: it sorts
correctly in whichever adapter's layout the prompt uses, misspells the tag on
some turns (a repair) and answers nonsense on others (an unreadable reply).
Deterministic: the choice depends on the request's content. Offline tests only.

    python tests/fake_openai.py PORT
"""

import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NEW = "// new name!"


def _text(m):
    c = m.get("content")
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in (c or []))


def _key(name, by):
    cut = next((i for i in range(1, len(name)) if name[i].isupper()), len(name))
    return name[:cut] if by == "FIRST" else name[cut:]


def answer(messages):
    users = [_text(m) for m in messages if m["role"] == "user"]
    system = next((_text(m) for m in messages if m["role"] == "system"), "")
    style, by, seen = None, "LAST", []
    for u in users:
        for pattern, s in ((r"by (\w+) name: (.*)", "original"), (r"Sort by: (\w+)\nNew names: (.*)", "tags"),
                           (r"Sort by (\w+) name\. New names: (.*)", "json")):
            m = re.search(pattern, u)
            if m:
                style, by = s, m.group(1)
                seen.append([n.strip() for n in m.group(2).split(",")])
                break
    everything = [n for turn in seen for n in turn]
    new = set(seen[-1]) if len(seen) > 1 else set()
    ranked = sorted(everything, key=lambda n: _key(n, by))
    roll = int(hashlib.sha256(json.dumps(messages).encode()).hexdigest(), 16) % 6
    if roll == 0:
        return "I am not sure how to sort these."                       # unreadable
    lines = "\n".join(n + (f" {NEW}" if n in new else "") for n in ranked)
    if style == "json":
        return "ANSWER: " + json.dumps([{"name": n, "new": n in new} for n in ranked])
    tag = "combined_alphabetical_sorted" if style == "original" else "answer"
    open_tag = f"<{tag.title()}>" if roll == 1 else f"<{tag}>"          # a misspelling
    return f"{open_tag}\n{lines}\n</{tag}>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = answer(req["messages"])
        self._send({"id": "c1", "object": "chat.completion", "created": 0, "model": req.get("model", "fake"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
