"""NDJSON writer: one JSON object per line."""
from __future__ import annotations

import json
from typing import Any, TextIO


class NDJSONWriter:
    """Writes one JSON-encoded message per line to a text stream.

    Does not close the underlying stream on close() by default — the
    stream is often sys.stdout or an externally-managed file handle,
    and closing is the caller's responsibility.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, message: dict[str, Any]) -> None:
        self._stream.write(json.dumps(message, separators=(",", ":")) + "\n")

    def close(self) -> None:
        return None

    def __enter__(self) -> "NDJSONWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
