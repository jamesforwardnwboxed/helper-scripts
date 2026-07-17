#!/usr/bin/env python3
"""Export an OpenCode session and its subagents as a self-contained HTML ZIP.

Message rows contain message metadata. The user-visible text and tool-call
details are stored in related ``part`` rows, so this exporter includes both
tables and deliberately omits the lower-level event log.
"""

from __future__ import annotations

import argparse
import curses
import html
import json
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_opencode_cache import DEFAULT_DB, pick_session


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"raw": str(value)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def format_timestamp(value: Any) -> str:
    try:
        timestamp = float(value) / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return "Unknown time"


def format_number(value: Any) -> str:
    try:
        return f"{int(value or 0):,}"
    except (TypeError, ValueError):
        return "0"


def session_tree(
    connection: sqlite3.Connection, session_id: str
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        WITH RECURSIVE descendants(id) AS (
            SELECT id FROM session WHERE id = ?
            UNION ALL
            SELECT session.id
            FROM session
            JOIN descendants ON session.parent_id = descendants.id
        )
        SELECT session.*
        FROM session
        JOIN descendants ON descendants.id = session.id
        ORDER BY session.time_created, session.id
        """,
        (session_id,),
    )
    sessions = [dict(row) for row in rows]
    if not sessions:
        raise ValueError(f"session not found: {session_id}")
    return sessions


def session_records(
    connection: sqlite3.Connection, session_ids: list[str]
) -> dict[str, dict[str, Any]]:
    records = {
        session_id: {"messages": [], "parts_by_message": {}}
        for session_id in session_ids
    }
    placeholders = ",".join("?" for _ in session_ids)

    message_rows = connection.execute(
        f"""
        SELECT id, session_id, time_created, time_updated, data
        FROM message
        WHERE session_id IN ({placeholders})
        ORDER BY time_created, id
        """,
        session_ids,
    )
    for row in message_rows:
        records[row["session_id"]]["messages"].append(
            {
                "id": row["id"],
                "time_created": row["time_created"],
                "time_updated": row["time_updated"],
                "data": parse_json(row["data"]),
            }
        )

    part_rows = connection.execute(
        f"""
        SELECT id, message_id, session_id, time_created, time_updated, data
        FROM part
        WHERE session_id IN ({placeholders})
        ORDER BY time_created, id
        """,
        session_ids,
    )
    for row in part_rows:
        records[row["session_id"]]["parts_by_message"].setdefault(
            row["message_id"], []
        ).append(
            {
                "id": row["id"],
                "time_created": row["time_created"],
                "time_updated": row["time_updated"],
                "data": parse_json(row["data"]),
            }
        )

    return records


def json_block(value: Any, class_name: str = "") -> str:
    content = html.escape(safe_text(value))
    return f'<pre class="code {class_name}">{content}</pre>'


def part_body(part: dict[str, Any]) -> tuple[str, str, str]:
    data = part["data"]
    part_type = str(data.get("type") or "part")
    if part_type in {"text", "reasoning"}:
        return part_type, html.escape(str(data.get("text") or "")), ""

    if part_type == "tool":
        state = data.get("state") or {}
        tool_name = data.get("tool") or "tool"
        status = state.get("status") or "unknown"
        call_id = data.get("callID")
        input_value = state.get("input", data.get("input"))
        output_value = state.get("output", data.get("output"))
        if output_value is None:
            output_value = (state.get("metadata") or {}).get("output")
        details = [
            f'<div class="tool-heading"><strong>{html.escape(str(tool_name))}</strong>'
            f' <span class="status status-{html.escape(str(status))}">{html.escape(str(status))}</span></div>'
        ]
        if call_id:
            details.append(f'<div class="muted">Call ID: {html.escape(str(call_id))}</div>')
        if input_value is not None:
            details.append('<div class="label">Input</div>')
            details.append(json_block(input_value, "tool-input"))
        if output_value is not None:
            details.append('<div class="label">Output</div>')
            details.append(json_block(output_value, "tool-output"))
        return "tool", "".join(details), "tool-part"

    if part_type in {"step-start", "step-finish"}:
        details = {key: value for key, value in data.items() if key != "type"}
        return part_type, json_block(details), "compact-part"

    details = {key: value for key, value in data.items() if key != "type"}
    return part_type, json_block(details), "compact-part"


def render_message(
    message: dict[str, Any], parts: list[dict[str, Any]], message_index: int
) -> tuple[str, int]:
    data = message["data"]
    role = str(data.get("role") or "unknown")
    role_class = role.replace("_", "-")
    message_id = html.escape(message["id"])
    model = data.get("modelID")
    provider = data.get("providerID")
    model_label = f"{provider}/{model}" if provider and model else model or provider
    tokens = data.get("tokens") or {}
    cache = tokens.get("cache") or {}
    meta = [
        format_timestamp(message["time_created"]),
        f"{format_number(tokens.get('input'))} input",
        f"{format_number(cache.get('read'))} cached",
        f"{format_number(tokens.get('output'))} output",
    ]
    if model_label:
        meta.append(str(model_label))

    parts_html: list[str] = []
    tool_count = 0
    for part_index, part in enumerate(parts):
        part_type, body, extra_class = part_body(part)
        if part_type == "tool":
            tool_count += 1
        part_id = f"message-{message_index}-part-{part_index}"
        parts_html.append(
            f'<section class="part {html.escape(extra_class)}" id="{part_id}">'
            f'<div class="part-label">{html.escape(part_type)}</div>{body}</section>'
        )

    if not parts_html:
        parts_html.append('<div class="muted empty">No message parts found.</div>')

    return (
        f'<article class="message {html.escape(role_class)}" id="message-{message_index}" '
        f'data-search="{html.escape((role + " " + safe_text(data)).lower())}">'
        f'<header class="message-header"><div><span class="role">{html.escape(role)}</span>'
        f'<span class="message-id">{message_id}</span></div>'
        f'<div class="message-meta">{" · ".join(html.escape(item) for item in meta)}</div></header>'
        f'<div class="parts">{"".join(parts_html)}</div></article>',
        tool_count,
    )


def render_session(
    session: dict[str, Any], record: dict[str, Any], session_index: int
) -> tuple[str, dict[str, int]]:
    session_id = html.escape(session["id"])
    title = html.escape(session.get("title") or "Untitled session")
    parent = session.get("parent_id")
    message_html: list[str] = []
    role_counts: dict[str, int] = {}
    tool_count = 0
    for index, message in enumerate(record["messages"]):
        role = str(message["data"].get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        rendered, tools = render_message(
            message,
            record["parts_by_message"].get(message["id"], []),
            session_index * 100000 + index,
        )
        message_html.append(rendered)
        tool_count += tools

    tokens = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
    }
    cost = 0.0
    for message in record["messages"]:
        data = message["data"]
        if data.get("role") != "assistant":
            continue
        token_data = data.get("tokens") or {}
        cache = token_data.get("cache") or {}
        tokens["input"] += int(token_data.get("input") or 0)
        tokens["output"] += int(token_data.get("output") or 0)
        tokens["cache_read"] += int(cache.get("read") or 0)
        tokens["cache_write"] += int(cache.get("write") or 0)
        try:
            cost += float(data.get("cost") or 0)
        except (TypeError, ValueError):
            pass

    parent_label = f' · child of <code>{html.escape(str(parent))}</code>' if parent else ""
    return (
        f'<section class="session" id="session-{session_id}">'
        f'<div class="session-heading"><div><div class="eyebrow">'
        f'{"Subagent" if parent else "Main agent"}</div><h2>{title}</h2>'
        f'<div class="session-id">{session_id}{parent_label}</div></div>'
        f'<div class="session-stats"><strong>{len(record["messages"]):,}</strong> messages '
        f'<strong>{tool_count:,}</strong> tool calls</div></div>'
        f'<div class="session-summary"><span>{format_timestamp(session["time_created"])}</span>'
        f'<span>{format_number(tokens["input"])} input</span>'
        f'<span>{format_number(tokens["cache_read"])} cached</span>'
        f'<span>{format_number(tokens["output"])} output</span>'
        f'<span>${cost:.8f}</span></div>'
        f'<div class="messages">{"".join(message_html)}</div></section>',
        {"messages": len(record["messages"]), "tools": tool_count, "cost": cost, **tokens},
    )


CSS = r"""
:root { --ink:#17212b; --muted:#667786; --line:#dce5ea; --paper:#f7fafb; --card:#fff; --accent:#d96c45; --accent-dark:#8e3f28; --blue:#257a88; --shadow:0 14px 40px rgba(32,54,66,.08); }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:var(--paper); font:15px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
a { color:var(--accent-dark); text-decoration:none; }
a:hover { text-decoration:underline; }
.shell { display:grid; grid-template-columns:300px minmax(0,1fr); min-height:100vh; }
.sidebar { position:sticky; top:0; height:100vh; overflow:auto; padding:28px 20px; color:#eff6f7; background:#17333d; }
.brand { font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:#a9d4d4; font-weight:800; }
.sidebar h1 { margin:9px 0 22px; font:700 24px/1.15 Georgia, serif; color:#fff; }
.search { width:100%; border:1px solid #41616a; border-radius:9px; background:#21434d; padding:10px 12px; color:#fff; outline:none; }
.search::placeholder { color:#a6bec1; }
.search-status { min-height:18px; margin-top:7px; color:#91b2b7; font-size:11px; }
.nav-label { margin:25px 0 8px; color:#a9d4d4; font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.nav-item { display:block; padding:9px 10px; border-radius:8px; color:#d9eaeb; font-size:13px; }
.nav-item:hover { background:#244b55; text-decoration:none; }
.nav-item .nav-id { display:block; color:#91b2b7; font:11px ui-monospace, SFMono-Regular, Menlo, monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.main { max-width:1280px; width:100%; padding:45px clamp(24px,5vw,72px) 100px; }
.hero { display:flex; justify-content:space-between; gap:28px; align-items:flex-end; border-bottom:1px solid var(--line); padding-bottom:35px; margin-bottom:38px; }
.eyebrow { color:var(--accent); font-size:11px; font-weight:900; letter-spacing:.16em; text-transform:uppercase; }
h1,h2 { font-family:Georgia, "Times New Roman", serif; letter-spacing:-.025em; }
.hero h1 { max-width:850px; margin:10px 0 10px; font-size:clamp(32px,5vw,60px); line-height:1.02; }
.hero p { max-width:760px; margin:0; color:var(--muted); font-size:16px; }
.hero-date { color:var(--muted); font-size:12px; text-align:right; white-space:nowrap; }
.overview { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-bottom:45px; }
.stat { padding:17px 18px; border:1px solid var(--line); border-radius:12px; background:var(--card); box-shadow:var(--shadow); }
.stat strong { display:block; font:700 23px Georgia, serif; }
.stat span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
.session { margin:0 0 54px; scroll-margin-top:25px; }
.session-heading { display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:13px; }
.session h2 { margin:5px 0 2px; font-size:31px; line-height:1.1; }
.session-id { color:var(--muted); font:12px ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
.session-stats { color:var(--muted); font-size:12px; white-space:nowrap; }
.session-stats strong { color:var(--ink); }
.session-summary { display:flex; flex-wrap:wrap; gap:8px 18px; margin-bottom:15px; color:var(--muted); font-size:12px; }
.session-summary span:not(:first-child) { padding-left:18px; border-left:1px solid var(--line); }
.message { margin:14px 0; border:1px solid var(--line); border-radius:13px; background:var(--card); box-shadow:0 5px 18px rgba(32,54,66,.045); overflow:hidden; scroll-margin-top:20px; }
.message.search-hidden { display:none !important; }
.message.user { border-left:4px solid var(--blue); }
.message.assistant { border-left:4px solid var(--accent); }
.message-header { display:flex; justify-content:space-between; gap:20px; padding:13px 17px; background:#fbfcfc; border-bottom:1px solid var(--line); }
.role { font-weight:900; text-transform:uppercase; font-size:11px; letter-spacing:.12em; }
.message.user .role { color:var(--blue); }
.message.assistant .role { color:var(--accent-dark); }
.message-id { margin-left:10px; color:#9aaab2; font:10px ui-monospace, SFMono-Regular, Menlo, monospace; }
.message-meta { color:var(--muted); font-size:11px; text-align:right; }
.parts { padding:3px 17px 12px; }
.part { padding:15px 0 4px; }
.part + .part { border-top:1px solid #edf1f2; }
.part-label,.label { margin-bottom:6px; color:var(--muted); font-size:10px; font-weight:900; letter-spacing:.12em; text-transform:uppercase; }
.part-label { color:#9aaab2; }
.reasoning { color:#586975; font-family:Georgia, serif; font-size:15px; }
.tool-part { margin:9px 0 12px; padding:14px; border:1px solid #d5e5e7; border-radius:10px; background:#f4faf9; }
.tool-heading { display:flex; align-items:center; gap:9px; font:14px ui-monospace, SFMono-Regular, Menlo, monospace; }
.status { padding:2px 7px; border-radius:20px; background:#dcebec; color:#28636d; font:10px ui-sans-serif,system-ui,sans-serif; text-transform:uppercase; letter-spacing:.08em; }
.status-error, .status-failed { background:#f8dfd8; color:#9a3928; }
.status-completed { background:#dcefdc; color:#39713b; }
.code { margin:7px 0 13px; padding:12px 13px; border:1px solid #e0e8ea; border-radius:8px; background:#fbfcfc; color:#263943; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
.tool-input { background:#fffdf7; }
.tool-output { max-height:480px; }
.compact-part { opacity:.82; }
.muted { color:var(--muted); font-size:12px; }
.empty { padding:10px 0; }
code { font:11px ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (max-width:900px) { .shell { display:block; } .sidebar { position:relative; height:auto; } .overview { grid-template-columns:repeat(2,minmax(0,1fr)); } .hero,.session-heading,.message-header { display:block; } .hero-date,.message-meta { margin-top:12px; text-align:left; } }
@media (max-width:500px) { .main { padding:28px 15px 70px; } .overview { grid-template-columns:1fr 1fr; } .stat { padding:12px; } .stat strong { font-size:18px; } }
"""


JS = r"""
(function () {
  function initSearch() {
    var search = document.querySelector('.search');
    var status = document.querySelector('.search-status');
    var messages = Array.prototype.slice.call(document.querySelectorAll('.message'));
    if (!search || !status || !messages.length) return;

    function updateSearch() {
      var query = search.value.trim().toLowerCase();
      var matches = 0;
      messages.forEach(function (message) {
        var haystack = (message.textContent || '').toLowerCase();
        var visible = !query || haystack.indexOf(query) !== -1;
        message.classList.toggle('search-hidden', !visible);
        if (visible) matches += 1;
      });
      status.textContent = query
        ? matches + ' matching message' + (matches === 1 ? '' : 's')
        : messages.length + ' messages';
    }

    search.addEventListener('input', updateSearch);
    updateSearch();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSearch);
  } else {
    initSearch();
  }
}());
"""


def build_html(
    root_session: dict[str, Any],
    sessions: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> str:
    rendered_sessions: list[str] = []
    stats = {"messages": 0, "tools": 0, "cost": 0.0, "input": 0, "output": 0, "cache_read": 0}
    nav: list[str] = []
    for index, session in enumerate(sessions):
        rendered, session_stats = render_session(session, records[session["id"]], index)
        rendered_sessions.append(rendered)
        stats["messages"] += session_stats["messages"]
        stats["tools"] += session_stats["tools"]
        stats["cost"] += session_stats["cost"]
        stats["input"] += session_stats["input"]
        stats["output"] += session_stats["output"]
        stats["cache_read"] += session_stats["cache_read"]
        label = "↳ " if session.get("parent_id") else ""
        nav.append(
            f'<a class="nav-item" href="#session-{html.escape(session["id"])}">'
            f'{label}{html.escape(session.get("title") or "Untitled session")}'
            f'<span class="nav-id">{html.escape(session["id"])}</span></a>'
        )

    title = html.escape(root_session.get("title") or "OpenCode session export")
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · OpenCode export</title><style>{CSS}</style>
</head>
<body>
<div class="shell">
<aside class="sidebar"><div class="brand">OpenCode / session archive</div><h1>Execution log</h1>
<input class="search" type="search" placeholder="Search messages and tools..." aria-label="Search messages and tools">
<div class="search-status" aria-live="polite"></div>
<div class="nav-label">Sessions</div>{''.join(nav)}</aside>
<main class="main">
<header class="hero"><div><div class="eyebrow">Full session export</div><h1>{title}</h1>
<p>Main agent conversation and all recursively attached subagent sessions. Message and tool-call data is rendered from the SQLite message and part records.</p></div>
<div class="hero-date">Generated<br><strong>{html.escape(generated)}</strong></div></header>
<section class="overview">
<div class="stat"><strong>{len(sessions):,}</strong><span>sessions</span></div>
<div class="stat"><strong>{stats['messages']:,}</strong><span>messages</span></div>
<div class="stat"><strong>{stats['tools']:,}</strong><span>tool calls</span></div>
<div class="stat"><strong>{format_number(stats['cache_read'])}</strong><span>cached tokens</span></div>
<div class="stat"><strong>${stats['cost']:.8f}</strong><span>recorded cost</span></div>
</section>
{''.join(rendered_sessions)}
</main></div>
<script>{JS}</script>
</body></html>'''


def output_name() -> Path | None:
    default = "opencode-session-export.zip"
    try:
        answer = input(f"Save ZIP as [{default}]: ").strip()
    except EOFError:
        print("error: no output filename supplied", file=sys.stderr)
        return None
    filename = answer or default
    if not filename.lower().endswith(".zip"):
        filename += ".zip"
    path = Path(filename)
    if path.name != filename or path.is_absolute() or filename in {".", ".."}:
        print("error: enter a filename only; the ZIP is saved in the launch folder", file=sys.stderr)
        return None
    output = Path.cwd() / path
    if output.exists():
        try:
            overwrite = input(f"{output.name} already exists. Overwrite? [y/N]: ").strip().lower()
        except EOFError:
            return None
        if overwrite not in {"y", "yes"}:
            print("Export cancelled.")
            return None
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an OpenCode session as a styled HTML ZIP.")
    parser.add_argument("session_id", nargs="?", help="session ID; omit to open the picker")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db.expanduser()
    if not db_path.is_file():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 2

    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            session_id = args.session_id
            if session_id is None:
                try:
                    session_id = pick_session(connection)
                except curses.error as error:
                    print(f"error: interactive picker requires a terminal ({error})", file=sys.stderr)
                    return 2
                if session_id is None:
                    return 0

            sessions = session_tree(connection, session_id)
            records = session_records(connection, [session["id"] for session in sessions])
            root_session = sessions[0]
            document = build_html(root_session, sessions, records)
        finally:
            connection.close()
    except (sqlite3.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    destination = output_name()
    if destination is None:
        return 2
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("index.html", document)
    print(f"Exported {len(sessions)} session(s) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
