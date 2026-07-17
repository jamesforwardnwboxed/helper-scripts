#!/usr/bin/env python3
"""Analyze OpenCode token-cache usage for one session.

OpenCode stores cache counters on assistant messages as:

    tokens.cache.read
    tokens.cache.write

The database does not currently contain an explicit cache-invalidation event.
This script therefore reports "inferred invalidations": a transition from one
or more cache reads to a later token-bearing assistant message with no cache
read. Treat those transitions as a useful heuristic, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("~/.local/share/opencode/opencode.db").expanduser()


def integer(value: Any) -> int:
    """Convert nullable or numeric JSON values to a non-negative integer."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def percent(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator else 0.0


def model_name(value: Any) -> str | None:
    if isinstance(value, dict):
        provider = value.get("providerID")
        model = value.get("id") or value.get("modelID")
        if provider and model:
            return f"{provider}/{model}"
        return model or provider
    if isinstance(value, str):
        try:
            return model_name(json.loads(value))
        except json.JSONDecodeError:
            return value
    return None


def message_tokens(data: dict[str, Any]) -> dict[str, int]:
    tokens = data.get("tokens") or {}
    cache = tokens.get("cache") or {}
    return {
        "input": integer(tokens.get("input")),
        "output": integer(tokens.get("output")),
        "reasoning": integer(tokens.get("reasoning")),
        "total": integer(tokens.get("total")),
        "cache_read": integer(cache.get("read", tokens.get("cache_read"))),
        "cache_write": integer(cache.get("write", tokens.get("cache_write"))),
    }


def load_session(
    connection: sqlite3.Connection, session_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    session_row = connection.execute(
        """
        SELECT id, title, directory, version, model, time_created, time_updated,
               cost, tokens_input, tokens_output, tokens_reasoning,
               tokens_cache_read, tokens_cache_write
        FROM session
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if session_row is None:
        raise ValueError(f"session not found: {session_id}")

    session = dict(session_row)
    messages: list[dict[str, Any]] = []
    invalid_json_count = 0

    rows = connection.execute(
        """
        SELECT id, time_created, data
        FROM message
        WHERE session_id = ?
        ORDER BY time_created, id
        """,
        (session_id,),
    )
    for row in rows:
        try:
            data = json.loads(row["data"])
        except (TypeError, json.JSONDecodeError):
            invalid_json_count += 1
            continue

        if data.get("role") != "assistant":
            continue

        tokens = message_tokens(data)
        if not any(tokens.values()):
            continue

        messages.append(
            {
                "id": row["id"],
                "time_created": row["time_created"],
                "model_id": data.get("modelID"),
                "provider_id": data.get("providerID"),
                "finish": data.get("finish"),
                **tokens,
            }
        )

    return session, messages, invalid_json_count


def analyze(
    session: dict[str, Any], messages: list[dict[str, Any]], invalid_json_count: int
) -> dict[str, Any]:
    totals = {
        key: sum(message[key] for message in messages)
        for key in ("input", "output", "reasoning", "total", "cache_read", "cache_write")
    }
    read_messages = sum(message["cache_read"] > 0 for message in messages)
    write_messages = sum(message["cache_write"] > 0 for message in messages)

    # There is no explicit invalidation record in the current schema. Count
    # one inferred invalidation per contiguous cache-read gap after a hit.
    invalidations: list[dict[str, Any]] = []
    has_seen_cache_read = False
    in_read_gap = False
    previous_read_message_id: str | None = None
    for message in messages:
        if message["cache_read"] > 0:
            has_seen_cache_read = True
            in_read_gap = False
            previous_read_message_id = message["id"]
            continue

        meaningful = message["input"] > 0 or message["cache_write"] > 0
        if has_seen_cache_read and meaningful and not in_read_gap:
            invalidations.append(
                {
                    "message_id": message["id"],
                    "time_created": message["time_created"],
                    "previous_cache_read_message_id": previous_read_message_id,
                    "cache_write_tokens": message["cache_write"],
                    "reason": "cache-read reset",
                }
            )
            in_read_gap = True

    # OpenCode's `input` is the uncached input counter. Cache reads are a
    # separate counter, so this is the fraction of effective prompt tokens
    # served from cache, excluding cache-write accounting.
    effective_prompt_tokens = totals["input"] + totals["cache_read"]
    cache_read_share = percent(totals["cache_read"], effective_prompt_tokens)

    return {
        "session": {
            "id": session["id"],
            "title": session["title"],
            "directory": session["directory"],
            "version": session["version"],
            "model": model_name(session["model"]),
            "time_created": session["time_created"],
            "time_updated": session["time_updated"],
            "cost": session["cost"],
        },
        "message_count": len(messages),
        "invalid_json_count": invalid_json_count,
        "totals_from_messages": totals,
        "stored_session_totals": {
            "input": integer(session["tokens_input"]),
            "output": integer(session["tokens_output"]),
            "reasoning": integer(session["tokens_reasoning"]),
            "cache_read": integer(session["tokens_cache_read"]),
            "cache_write": integer(session["tokens_cache_write"]),
        },
        "derived": {
            "effective_prompt_tokens": effective_prompt_tokens,
            "cache_read_share_percent": round(cache_read_share, 2),
            "uncached_input_share_percent": round(
                percent(totals["input"], effective_prompt_tokens), 2
            ),
            "cache_read_messages": read_messages,
            "cache_write_messages": write_messages,
            "inferred_invalidations": len(invalidations),
            "invalidation_detection": (
                "heuristic: first meaningful no-cache-read message after a cache-read run"
            ),
        },
        "inferred_invalidation_events": invalidations,
        "messages": messages,
    }


def format_number(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def print_report(report: dict[str, Any], show_timeline: bool) -> None:
    session = report["session"]
    totals = report["totals_from_messages"]
    stored = report["stored_session_totals"]
    derived = report["derived"]

    print(f"Session: {session['id']}")
    print(f"Title:   {session['title']}")
    print(f"Model:   {session['model'] or 'unknown'}")
    print(f"Messages with token data: {report['message_count']}")
    print()
    print("Token usage")
    print(f"  Uncached input: {format_number(totals['input'])}")
    print(f"  Cache read:     {format_number(totals['cache_read'])}")
    print(f"  Cache write:    {format_number(totals['cache_write'])}")
    print(f"  Output:         {format_number(totals['output'])}")
    print(f"  Reasoning:      {format_number(totals['reasoning'])}")
    print(f"  Effective prompt: {format_number(derived['effective_prompt_tokens'])}")
    print()
    print("Cache efficiency")
    print(f"  Input served from cache: {derived['cache_read_share_percent']:.2f}%")
    print(f"  Input not served from cache: {derived['uncached_input_share_percent']:.2f}%")
    print(f"  Messages with cache reads:  {derived['cache_read_messages']}")
    print(f"  Messages with cache writes: {derived['cache_write_messages']}")
    print(f"  Inferred invalidations:     {derived['inferred_invalidations']}")
    print("  Invalidation definition:    cache-read reset after a prior cache hit")
    print()
    print("Stored session totals vs message totals")
    for key in ("input", "output", "reasoning", "cache_read", "cache_write"):
        message_value = totals[key]
        stored_value = stored[key]
        marker = "" if message_value == stored_value else " (DIFF)"
        print(
            f"  {key:11} messages={format_number(message_value):>12} "
            f"session={format_number(stored_value):>12}{marker}"
        )

    if report["invalid_json_count"]:
        print(f"\nWarning: skipped {report['invalid_json_count']} malformed message row(s).")

    if report["inferred_invalidation_events"]:
        print("\nInferred invalidation events")
        for event in report["inferred_invalidation_events"]:
            write = format_number(event["cache_write_tokens"])
            print(f"  {event['message_id']} (cache write: {write})")

    if show_timeline:
        print("\nMessage timeline")
        print("  id                                      input    cache-read  cache-write  output")
        for message in report["messages"]:
            print(
                f"  {message['id']:<40} {message['input']:>8,} "
                f"{message['cache_read']:>11,} {message['cache_write']:>12,} "
                f"{message['output']:>8,}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze OpenCode cache usage for a session."
    )
    parser.add_argument("session_id", help="OpenCode session ID, for example ses_...")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite database path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="write machine-readable JSON instead of the text report",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="include one row per assistant message in the text report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db.expanduser()
    if not db_path.is_file():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 2

    try:
        connection = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=5
        )
        connection.row_factory = sqlite3.Row
        try:
            session, messages, invalid_json_count = load_session(
                connection, args.session_id
            )
        finally:
            connection.close()
    except (sqlite3.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    report = analyze(session, messages, invalid_json_count)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, args.timeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
