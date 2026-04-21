import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import uuid
import shutil
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import io

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from agent import get_agent, get_agents_md_files

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

st.set_page_config(page_title="Nemotron Science Agent", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; color: #ececec; }
    [data-testid="collapsedControl"] { display: none; }

    .msg-user {
        background: #2f2f2f;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: auto;
        max-width: 85%;
        color: #ececec;
        font-size: 15px;
    }
    .msg-assistant {
        background: #1a1a2e;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 85%;
        color: #ececec;
        font-size: 15px;
        border-left: 3px solid #4f8ef7;
    }
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: #ececec !important;
        border: 1px solid #3a3a3a !important;
        border-radius: 12px !important;
        font-size: 15px !important;
    }
    .exec-block {
        background: #111827;
        border-left: 3px solid #4f8ef7;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
        font-size: 12px;
        font-family: monospace;
        color: #93c5fd;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .subagent-header {
        background: #1e293b;
        border-left: 3px solid #a78bfa;
        padding: 6px 12px;
        margin: 10px 0 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 12px;
        color: #c4b5fd;
        font-weight: bold;
    }
    .tool-badge {
        background: #1e3a5f;
        color: #7eb8f7;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 2px 3px;
    }
    .panel-title {
        font-size: 12px;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        border-bottom: 1px solid #222;
        padding-bottom: 5px;
    }
    h1 { color: #4f8ef7 !important; }
    h3 { color: #888 !important; font-weight: 400 !important; }
</style>
""", unsafe_allow_html=True)

# ── 상태 초기화 ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "executions" not in st.session_state:
    st.session_state.executions = []


# ── 툴 카테고리 분류 ─────────────────────────────────────────
_CHEM_TOOLS = {
    "name_to_smiles", "smiles_to_weight", "get_mol_formula", "get_crippen_descriptors",
    "calculate_tpsa", "get_hbd_count", "get_hba_count", "get_rotatable_bonds",
    "get_functional_groups", "mol_similarity", "check_safety", "predict_reaction",
    "retrosynthesis", "run_scitool",
}
_BIO_TOOLS = {
    "compute_protein_parameters", "compute_pi_mw", "translate_dna",
    "get_reverse_complement", "find_orf", "sequence_alignment",
}
_MAT_TOOLS = {
    "get_band_gap", "get_density", "get_formation_energy", "is_metal",
    "search_materials", "get_structure_info", "calculate_symmetry",
}
_GEN_TOOLS = {
    "download_papers", "paper_qa", "run_python", "ocr_image",
    "kg_search_tools", "kg_plan_chain", "kg_next_tools", "kg_category_tools",
    "make_science_plan", "spawn_agent",
}

def _tool_color(name: str) -> str:
    if name in _CHEM_TOOLS:   return "#f59e0b"  # 노랑 = 화학
    if name in _BIO_TOOLS:    return "#34d399"  # 초록 = 생물
    if name in _MAT_TOOLS:    return "#f87171"  # 빨강 = 재료
    if name in _GEN_TOOLS:    return "#94a3b8"  # 회색 = 일반
    return "#4f8ef7"                             # 파랑 = 기타 scitool


# ── 툴 그래프 생성 ───────────────────────────────────────────
def build_tool_graph(tool_calls: list) -> bytes | None:
    if not tool_calls:
        return None

    # 서브에이전트 → 툴 목록 매핑
    agent_tools: dict[str, list[str]] = {}
    for agent_node, tool_name in tool_calls:
        agent_tools.setdefault(agent_node, [])
        if tool_name not in agent_tools[agent_node]:
            agent_tools[agent_node].append(tool_name)

    G = nx.DiGraph()
    for agent_node, tools in agent_tools.items():
        G.add_node(agent_node, kind="agent")
        for tool_name in tools:
            G.add_node(tool_name, kind="tool")
            G.add_edge(agent_node, tool_name)

    agent_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "agent"]
    tool_nodes  = [n for n, d in G.nodes(data=True) if d.get("kind") == "tool"]
    tool_colors = [_tool_color(n) for n in tool_nodes]

    pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(max(8, len(G.nodes()) * 1.4), max(5, len(agent_nodes) * 2)))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    nx.draw_networkx_nodes(G, pos, nodelist=agent_nodes, node_color="#a78bfa",
                           node_size=2500, ax=ax, alpha=0.95)
    nx.draw_networkx_nodes(G, pos, nodelist=tool_nodes, node_color=tool_colors,
                           node_size=1800, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#555", arrows=True,
                           arrowstyle="-|>", arrowsize=15, ax=ax,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=7, ax=ax)

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color="#a78bfa", label="Subagent"),
        mpatches.Patch(color="#f59e0b", label="Chemistry"),
        mpatches.Patch(color="#34d399", label="Biology"),
        mpatches.Patch(color="#f87171", label="Materials"),
        mpatches.Patch(color="#94a3b8", label="General"),
        mpatches.Patch(color="#4f8ef7", label="SciTool"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.2,
              labelcolor="white", facecolor="#1a1a1a", fontsize=8)
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Agent 실행 (스트리밍) ────────────────────────────────────
def run_agent_stream(query: str):
    agent = get_agent()
    thread_id = str(uuid.uuid4())

    log_lines = []
    tool_calls = []
    final_answer = ""
    active_nodes = set()
    _INTERNAL_NODES = {
        "deep_agent", "agent", "tools", "__interrupt__", "__end__",
        "model", "SkillsMiddleware", "PatchToolCallsMiddleware",
        "MemoryMiddleware", "before_agent", "after_agent",
    }

    def fmt_tool_call(tc, node):
        name = tc.get("name", "?")
        args = tc.get("args", {})
        args_str = str(args)
        if len(args_str) > 200:
            args_str = args_str[:200] + "…"
        prefix = f"[{node}] " if node not in ("deep_agent", "agent") else ""
        if name == "spawn_agent":
            role = args.get("role", "?")
            return f"🤖 {prefix}spawn_agent → `{role}`", name
        if name == "task":
            agent_name = args.get("subagent_name", args.get("name", "?"))
            return f"📋 {prefix}task → `{agent_name}`", name
        return f"🔧 {prefix}{name}\n{args_str}", name

    def fmt_tool_result(msg, node):
        content = str(msg.content)
        if len(content) > 400:
            content = content[:400] + "…"
        name = getattr(msg, "name", "result")
        prefix = f"[{node}] " if node not in ("deep_agent", "agent") else ""
        return f"   ↳ {prefix}{name}:\n{content}"

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=query)], "files": get_agents_md_files()},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for node_name, node_data in chunk.items():
            if not isinstance(node_data, dict):
                continue

            _is_internal = (
                node_name in _INTERNAL_NODES
                or node_name.endswith(".before_agent")
                or node_name.endswith(".after_agent")
                or "Middleware" in node_name
            )
            if not _is_internal:
                if node_name not in active_nodes:
                    active_nodes.add(node_name)
                    log_lines.append(("subagent", f"🤖 Subagent: {node_name}"))

            raw = node_data.get("messages", [])
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
                            text, tname = fmt_tool_call(tc, node_name)
                            log_lines.append(("tool_call", text))
                            tool_calls.append((node_name, tname))
                    elif msg.content:
                        final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                elif isinstance(msg, ToolMessage):
                    log_lines.append(("tool_result", fmt_tool_result(msg, node_name)))

        yield log_lines, tool_calls, final_answer


# ── 레이아웃 ─────────────────────────────────────────────────
col_chat, col_exec = st.columns([6, 4], gap="large")

with col_chat:
    st.markdown("<h1 style='text-align:center; margin-bottom:2px'>🔬 Nemotron Science Agent</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; margin-bottom:20px'>Powered by DeepAgents · SciToolAgent · vLLM Nemotron-30B</h3>", unsafe_allow_html=True)

    # 채팅 히스토리
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; color:#444; margin-top:60px; font-size:14px'>
            Ask a scientific question to get started.<br><br>
            <span style='font-size:12px; color:#333'>
            e.g. "What hypotheses could explain the Warburg effect in cancer cells?"
            </span>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='msg-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg-assistant'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("", placeholder="Ask a scientific question...", height=80, label_visibility="collapsed")
        col1, col2 = st.columns(2)
        with col1:
            csv_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
        with col2:
            image_file = st.file_uploader("Upload Image/PDF for OCR (optional)", type=["png", "jpg", "jpeg", "pdf", "tiff", "bmp"])
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        query = user_input.strip()

        if csv_file is not None:
            dest = WORKSPACE / csv_file.name
            dest.write_bytes(csv_file.read())
            query += f"\n\n[Attached CSV: {dest}]"

        if image_file is not None:
            dest = WORKSPACE / image_file.name
            dest.write_bytes(image_file.read())
            query += f"\n\n[Attached file for OCR: {dest} — use ocr_image('{dest}') to extract text from it]"

        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        # 실행 중 우측 패널 실시간 업데이트
        with col_exec:
            st.markdown("<div class='panel-title'>Execution Panel</div>", unsafe_allow_html=True)
            status_placeholder = st.empty()
            log_placeholder = st.empty()
            graph_placeholder = st.empty()

        answer_placeholder = col_chat.empty()
        answer_placeholder.markdown("<div class='msg-assistant'>🤖 ⏳ Agent thinking...</div>", unsafe_allow_html=True)

        final_answer = ""
        log_lines = []
        tool_calls = []

        try:
            for log_lines, tool_calls, final_answer in run_agent_stream(query):
                # 실행 로그 업데이트
                log_html = ""
                for kind, text in log_lines:
                    if kind == "subagent":
                        log_html += f"<div class='subagent-header'>{text}</div>"
                    elif kind == "tool_call":
                        log_html += f"<div class='exec-block'>{text}</div>"
                    elif kind == "tool_result":
                        log_html += f"<div class='exec-block' style='color:#86efac'>{text}</div>"
                log_placeholder.markdown(log_html, unsafe_allow_html=True)

                # 툴 그래프 업데이트
                if tool_calls:
                    graph_img = build_tool_graph(tool_calls)
                    if graph_img:
                        graph_placeholder.image(graph_img, use_container_width=True)

                # 답변 미리보기
                if final_answer:
                    answer_placeholder.markdown(f"<div class='msg-assistant'>🤖 {final_answer}</div>", unsafe_allow_html=True)

        except Exception as e:
            import traceback
            final_answer = f"Error: {e}\n{traceback.format_exc()}"
            answer_placeholder.markdown(f"<div class='msg-assistant'>🤖 {final_answer}</div>", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.session_state.executions.append({
            "question": user_input.strip(),
            "log_lines": log_lines,
            "tool_calls": tool_calls,
            "graph_img": build_tool_graph(tool_calls),
        })
        st.rerun()


# ── 우측: 실행 패널 (히스토리) ──────────────────────────────
with col_exec:
    st.markdown("<div class='panel-title'>Execution Panel</div>", unsafe_allow_html=True)

    if not st.session_state.executions:
        st.markdown("<div style='color:#333; font-size:13px; text-align:center; margin-top:40px'>Execution details will appear here.</div>", unsafe_allow_html=True)
    else:
        latest = st.session_state.executions[-1]

        # 사용된 툴 배지
        unique_tools = list(dict.fromkeys([name for _, name in latest["tool_calls"]]))
        if unique_tools:
            st.markdown("<div class='panel-title'>Tools Used</div>", unsafe_allow_html=True)
            badges = " ".join([f"<span class='tool-badge'>{t}</span>" for t in unique_tools])
            st.markdown(badges, unsafe_allow_html=True)

        # 툴 실행 그래프
        if latest["graph_img"]:
            st.markdown("<br><div class='panel-title'>Tool Execution Graph</div>", unsafe_allow_html=True)
            st.image(latest["graph_img"], use_container_width=True)

        # 실행 로그
        if latest["log_lines"]:
            st.markdown("<br><div class='panel-title'>Execution Log</div>", unsafe_allow_html=True)
            with st.expander("Show full log", expanded=True):
                log_html = ""
                for kind, text in latest["log_lines"]:
                    if kind == "subagent":
                        log_html += f"<div class='subagent-header'>{text}</div>"
                    elif kind == "tool_call":
                        log_html += f"<div class='exec-block'>{text}</div>"
                    elif kind == "tool_result":
                        log_html += f"<div class='exec-block' style='color:#86efac'>{text}</div>"
                st.markdown(log_html, unsafe_allow_html=True)

        # 이전 실행 기록
        if len(st.session_state.executions) > 1:
            st.markdown("<br><div class='panel-title'>Previous Runs</div>", unsafe_allow_html=True)
            for ex in reversed(st.session_state.executions[:-1]):
                with st.expander(f"Q: {ex['question'][:50]}..."):
                    tools = list(dict.fromkeys([n for _, n in ex["tool_calls"]]))
                    if tools:
                        badges = " ".join([f"<span class='tool-badge'>{t}</span>" for t in tools])
                        st.markdown(badges, unsafe_allow_html=True)
                    if ex["graph_img"]:
                        st.image(ex["graph_img"], use_container_width=True)
