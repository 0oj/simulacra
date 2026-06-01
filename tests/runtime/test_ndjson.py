"""Tests for runtime.NDJSONWriter."""
from __future__ import annotations

import io
import json

from simulacra.runtime.ndjson import NDJSONWriter


def test_writer_writes_one_message_per_line():
    buf = io.StringIO()
    w = NDJSONWriter(buf)
    w.write({"message_type": "metadata", "schema_version": "0.3.0", "dt": 0.01, "duration": 1.0})
    w.write({"message_type": "variable", "timestamp": 0.0, "x": 1.0})
    lines = buf.getvalue().rstrip("\n").split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["message_type"] == "metadata"
    assert json.loads(lines[1])["x"] == 1.0


def test_writer_close_is_a_noop():
    buf = io.StringIO()
    w = NDJSONWriter(buf)
    w.close()
    buf.write("after close\n")
    assert "after close" in buf.getvalue()


def test_writer_as_context_manager():
    buf = io.StringIO()
    with NDJSONWriter(buf) as w:
        w.write({"a": 1})
    assert json.loads(buf.getvalue().rstrip())["a"] == 1
