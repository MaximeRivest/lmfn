"""Run a Markdown notebook's python cells in one namespace and write an HTML
page with each cell's code and real output (stdout, and the last expression:
polars frames as tables).

    python docs/render_notebook.py docs/distill-demo.md out.html
"""

import ast
import contextlib
import html
import io
import re
import sys
import time
import traceback

import markdown  # type: ignore

CSS = """
body{font:15px/1.55 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1d1d1f;background:#fafaf7}
h1{font-size:1.9rem}h2{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
pre{background:#f0f0ea;border-radius:6px;padding:.8rem 1rem;overflow-x:auto;font-size:13px}
pre.code{border-left:3px solid #7aa33a}pre.out{background:#fff;border:1px solid #e3e3dc;border-left:3px solid #999}
pre.err{background:#fff4f4;border-left:3px solid #c33}
table{border-collapse:collapse;font-size:13px;margin:.5rem 0}td,th{border:1px solid #ddd;padding:3px 8px;text-align:left}
th{background:#eee}.meta{color:#777;font-size:13px}a{color:#3a6a1a}
"""


def run(md_path: str) -> str:
    text = open(md_path).read()
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    ns: dict = {"__name__": "__notebook__"}
    parts = []
    for chunk in re.split(r"(```python\n.*?```)", text, flags=re.S):
        m = re.match(r"```python\n(.*?)```", chunk, flags=re.S)
        if not m:
            parts.append(markdown.markdown(chunk, extensions=["tables"]))
            continue
        code = m.group(1)
        parts.append(f'<pre class="code">{html.escape(code.rstrip())}</pre>')
        buf = io.StringIO()
        shown, cls = "", "out"
        t0 = time.time()
        try:
            tree = ast.parse(code)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(tree, "<cell>", "exec"), ns)
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns) if last else None
            if value is not None:
                shown = value._repr_html_() if hasattr(value, "_repr_html_") else html.escape(repr(value))
        except Exception:
            cls = "err"
            buf.write(traceback.format_exc(limit=2))
        out = buf.getvalue()
        if out:
            parts.append(f'<pre class="{cls}">{html.escape(out.rstrip())}</pre>')
        if shown:
            parts.append(f'<div class="out">{shown}</div>')
        parts.append(f'<div class="meta">{time.time() - t0:.1f}s</div>')
    stamp = time.strftime("%Y-%m-%d %H:%M")
    return (f"<!doctype html><meta charset=utf-8><title>distill demo</title><style>{CSS}</style>"
            f'<p class="meta">Rendered {stamp} by running every cell (docs/distill-demo.md).</p>'
            + "\n".join(parts))


if __name__ == "__main__":
    open(sys.argv[2], "w").write(run(sys.argv[1]))
