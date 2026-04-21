"""Process-global side-channel for BFDTS visualization.

The planning tool writes the structured trace here; the FastAPI SSE handler
drains it immediately after emitting the `make_science_plan` tool_result event.
Keeping the trace out of the tool's string output means the LLM doesn't carry
a large JSON blob in its context on follow-up turns.

IMPORTANT: we can't use `threading.local()` here because LangGraph often
executes tool functions in a different thread from the one that's iterating
the SSE stream. A simple lock-guarded global slot works for single-user dev.
For true concurrency the next step is keying by the LangGraph thread_id.
"""

import threading
from typing import Optional

_lock = threading.Lock()
_trace: Optional[dict] = None


def set_trace(trace: dict) -> None:
    global _trace
    with _lock:
        _trace = trace


def pop_trace() -> Optional[dict]:
    global _trace
    with _lock:
        t = _trace
        _trace = None
        return t
