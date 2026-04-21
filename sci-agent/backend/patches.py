"""Runtime patches for upstream libraries.

langchain_openai's delta converter drops vLLM's `reasoning` field (used by the
`--reasoning-parser nemotron_v3` option). We surface it as
`additional_kwargs["reasoning"]` on every AIMessageChunk so the backend can
emit a parallel `reasoning_token` SSE stream for the Planning panel.
"""

from typing import Any, Mapping

import langchain_openai.chat_models.base as _oai_base

_original_convert = _oai_base._convert_delta_to_message_chunk


def _patched_convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type
):
    chunk = _original_convert(_dict, default_class)
    reasoning = _dict.get("reasoning") if isinstance(_dict, Mapping) else None
    if reasoning and hasattr(chunk, "additional_kwargs"):
        current = chunk.additional_kwargs or {}
        chunk.additional_kwargs = {**current, "reasoning": reasoning}
    return chunk


def apply_patches() -> None:
    _oai_base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk
