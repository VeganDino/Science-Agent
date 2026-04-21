import json
import uuid
from typing import Any, Generator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, HumanMessage

from agent import get_agent, get_agents_md_files
from tools._bfdts_trace import pop_trace

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None


_INTERNAL_NODES = {
    "deep_agent", "agent", "tools", "__interrupt__", "__end__",
    "model", "SkillsMiddleware", "PatchToolCallsMiddleware",
    "MemoryMiddleware", "before_agent", "after_agent",
}


def _is_internal_node(name: str) -> bool:
    return (
        name in _INTERNAL_NODES
        or name.endswith(".before_agent")
        or name.endswith(".after_agent")
        or "Middleware" in name
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    if isinstance(content, str):
        return content
    return ""


def _stream_agent(query: str, thread_id: str) -> Generator[str, None, None]:
    agent = get_agent()
    yield _sse("thread", {"thread_id": thread_id})

    active_nodes: set[str] = set()

    try:
        for mode, payload in agent.stream(
            {
                "messages": [HumanMessage(content=query)],
                "files": get_agents_md_files(),
            },
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                # payload = (message_chunk, metadata)
                msg_chunk, metadata = payload
                if not isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                    continue
                node = "agent"
                if isinstance(metadata, dict):
                    node = metadata.get("langgraph_node") or metadata.get("node") or "agent"
                # vLLM with --reasoning-parser nemotron_v3 streams CoT via the
                # `reasoning` field (surfaced by backend/patches.py into
                # additional_kwargs["reasoning"]). Emit it to the Plan panel.
                ak = getattr(msg_chunk, "additional_kwargs", None) or {}
                reasoning_delta = ak.get("reasoning")
                if reasoning_delta:
                    yield _sse("reasoning_token", {"node": node, "text": reasoning_delta})
                text = _extract_text(msg_chunk.content)
                if text:
                    yield _sse("token", {"node": node, "text": text})

            elif mode == "updates":
                chunk = payload
                if not isinstance(chunk, dict):
                    continue
                for node_name, node_data in chunk.items():
                    if not isinstance(node_data, dict):
                        continue

                    if not _is_internal_node(node_name) and node_name not in active_nodes:
                        active_nodes.add(node_name)
                        yield _sse("subagent", {"name": node_name})

                    raw = node_data.get("messages", [])
                    if hasattr(raw, "value"):
                        raw = raw.value
                    try:
                        messages = list(raw) if raw else []
                    except TypeError:
                        messages = []

                    for msg in messages:
                        if isinstance(msg, AIMessage) and not isinstance(msg, AIMessageChunk):
                            has_tool_calls = bool(getattr(msg, "tool_calls", None))
                            if has_tool_calls:
                                for tc in msg.tool_calls:
                                    yield _sse("tool_call", {
                                        "node": node_name,
                                        "name": tc.get("name", "?"),
                                        "args": tc.get("args", {}),
                                    })
                            text = _extract_text(msg.content)
                            if text.strip() or has_tool_calls:
                                yield _sse("message_end", {
                                    "node": node_name,
                                    "final": not has_tool_calls,
                                    "text": text,
                                })
                        elif isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, "name", "result")
                            yield _sse("tool_result", {
                                "node": node_name,
                                "name": tool_name,
                                "content": str(msg.content),
                            })
                            # Drain the BFDTS side-channel right after the
                            # planner's tool_result lands, so the UI can
                            # animate the exploration.
                            if tool_name == "make_science_plan":
                                trace = pop_trace()
                                if trace is not None:
                                    yield _sse("bfdts_trace", trace)
    except Exception as e:
        import traceback
        yield _sse("error", {"message": str(e), "traceback": traceback.format_exc()})
        return

    yield _sse("done", {})


@router.post("/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_agent(req.query, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
