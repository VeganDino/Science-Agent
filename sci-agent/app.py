import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import shutil
import gradio as gr
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from agent import get_agent, get_agents_md_files

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


def _fmt_tool_call(tc: dict, node: str) -> str:
    name = tc.get("name", "?")
    args = tc.get("args", {})
    args_str = str(args)
    if len(args_str) > 200:
        args_str = args_str[:200] + "…"
    prefix = f"[{node}] " if node not in ("deep_agent", "agent") else ""
    if name == "spawn_agent":
        role = args.get("role", "?")
        return f"🤖 **{prefix}spawn_agent** → `{role}`"
    if name == "task":
        agent_name = args.get("subagent_name", args.get("name", "?"))
        return f"📋 **{prefix}task** → `{agent_name}`"
    return f"🔧 **{prefix}{name}**\n```\n{args_str}\n```"


def _fmt_tool_result(msg: ToolMessage, node: str) -> str:
    content = str(msg.content)
    if len(content) > 400:
        content = content[:400] + "…"
    name = getattr(msg, "name", "result")
    prefix = f"[{node}] " if node not in ("deep_agent", "agent") else ""
    return f"   ↳ **{prefix}{name}**:\n```\n{content}\n```"


def run_agent(query: str, csv_file, thread_id: str):
    if not query.strip():
        yield thread_id, "", "Please enter a scientific question."
        return

    if not thread_id:
        thread_id = str(uuid.uuid4())

    if csv_file is not None:
        dest = WORKSPACE / Path(csv_file.name).name
        shutil.copy(csv_file.name, dest)
        query += f"\n\n[Attached CSV: {dest.name}]"

    agent = get_agent()

    log_lines = []
    final_answer = ""
    active_nodes = set()
    _INTERNAL_NODES = {
        "deep_agent", "agent", "tools", "__interrupt__", "__end__",
        "model", "SkillsMiddleware", "PatchToolCallsMiddleware",
        "MemoryMiddleware", "before_agent", "after_agent",
    }

    try:
        for chunk in agent.stream(
            {
                "messages": [HumanMessage(content=query)],
                "files": get_agents_md_files(),
            },
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            for node_name, node_data in chunk.items():
                if not isinstance(node_data, dict):
                    continue

                # Show when a real subagent node activates (skip DeepAgents internal middleware)
                _is_internal = (
                    node_name in _INTERNAL_NODES
                    or node_name.endswith(".before_agent")
                    or node_name.endswith(".after_agent")
                    or "Middleware" in node_name
                )
                if not _is_internal:
                    if node_name not in active_nodes:
                        active_nodes.add(node_name)
                        log_lines.append(f"\n---\n🤖 **Subagent: `{node_name}`** activated\n")

                raw = node_data.get("messages", [])
                # LangGraph may wrap state updates in Overwrite/other channel objects
                if hasattr(raw, "value"):
                    raw = raw.value
                try:
                    messages = list(raw) if raw else []
                except TypeError:
                    messages = []
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                log_lines.append(_fmt_tool_call(tc, node_name))
                        # Capture content regardless of whether tool_calls also present
                        # (Nemotron sometimes emits content + tool_calls in the same message)
                        raw_content = msg.content
                        if raw_content:
                            if isinstance(raw_content, list):
                                # Extract text blocks from list-of-dicts content
                                text_parts = [
                                    b.get("text", "") for b in raw_content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                ]
                                text = "\n".join(p for p in text_parts if p)
                            else:
                                text = str(raw_content)
                            if text.strip():
                                final_answer = text

                    elif isinstance(msg, ToolMessage):
                        log_lines.append(_fmt_tool_result(msg, node_name))

            yield thread_id, "\n\n".join(log_lines), final_answer

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        yield thread_id, f"**Error:**\n```\n{e}\n{tb}\n```", ""
        return

    yield thread_id, "\n\n".join(log_lines), final_answer


with gr.Blocks(title="Nemotron Science Agent") as demo:
    gr.Markdown("# Nemotron General Science Agent")
    gr.Markdown(
        "Powered by **DeepAgents** · **SciToolAgent** (485 tools) · **SciAgentGYM** (1414 functions) · **vLLM Nemotron-3-Nano-30B**"
    )

    # Per-browser-session thread ID — persists across queries, resets on "New Conversation"
    session_thread = gr.State(value="")

    with gr.Row():
        with gr.Column(scale=2):
            query_box = gr.Textbox(
                label="Scientific Question",
                placeholder="e.g. What hypotheses could explain the Warburg effect in cancer cells?",
                lines=4,
            )
            csv_upload = gr.File(label="Upload CSV (optional)", file_types=[".csv"])
            with gr.Row():
                submit_btn = gr.Button("Run Science Agent", variant="primary", scale=3)
                new_conv_btn = gr.Button("New Conversation", scale=1)

        with gr.Column(scale=3):
            output_box = gr.Markdown(
                label="Scientific Report",
                value="*Answer will appear here...*",
                min_height=400,
            )
            with gr.Accordion("Execution Log", open=False):
                log_box = gr.Markdown(
                    label="Tool calls · Subagent spawns · Code execution",
                    value="*Log will appear here...*",
                )

    submit_btn.click(
        fn=run_agent,
        inputs=[query_box, csv_upload, session_thread],
        outputs=[session_thread, log_box, output_box],
    )

    new_conv_btn.click(
        fn=lambda: (str(uuid.uuid4()), "", "*Answer will appear here...*", "*Log will appear here...*"),
        inputs=[],
        outputs=[session_thread, query_box, output_box, log_box],
    )

    gr.Examples(
        examples=[
            ["What hypotheses could explain the Warburg effect in cancer cells?", None],
            ["Calculate molecular descriptors for aspirin and assess its drug-likeness (Lipinski's rule of 5).", None],
            ["Design an experiment to test whether sleep deprivation impairs working memory.", None],
            ["What is the band gap and density of TiO2? Is it metallic?", None],
            ["Calculate the Doppler shift for a sound source moving at 30 m/s toward a stationary observer. Speed of sound is 343 m/s, source frequency 440 Hz.", None],
        ],
        inputs=[query_box, csv_upload],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
