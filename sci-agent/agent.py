import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sqlite3

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

from tools.scitool_tools import SCIENCE_TOOLS

VLLM_URL = "http://localhost:8000/v1"
SKILLS_DIR = str(Path(__file__).parent / "skills")
DB_PATH = str(Path(__file__).parent / "memory.db")
AGENTS_MD_PATH = Path(__file__).parent / "AGENTS.md"
AGENTS_MD_VIRTUAL = "/AGENTS.md"


def _read_agents_md() -> str:
    try:
        return AGENTS_MD_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def build_model(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model="model",
        base_url=VLLM_URL,
        api_key="none",
        temperature=temperature,
    )


SUBAGENTS = [
    {
        "name": "literature-agent",
        "description": "Search and analyze scientific papers on arXiv, extract key claims, methods, and evidence quality from PDFs",
        "system_prompt": (
            "You are a Literature Agent specializing in scientific paper analysis.\n"
            "Use download_papers(keyword) to fetch arXiv papers, paper_qa(question) to query them.\n"
            "Extract key claims, methods, evidence quality, and limitations.\n"
            "Rate citation quality: RCT > cohort > case study > preprint.\n"
            "Always cite specific parts of papers you reference."
        ),
        "tools": SCIENCE_TOOLS,
        "model": build_model(temperature=0.1),
    },
    {
        "name": "compute-agent",
        "description": "Run calculations and data analysis: molecular properties, protein parameters, material properties, statistical analysis on CSV data",
        "system_prompt": (
            "You are a Computation Agent for scientific calculations.\n"
            "Use chemistry shortcuts (name_to_smiles, smiles_to_weight, get_crippen_descriptors, calculate_tpsa) for molecular properties.\n"
            "Use biology tools (compute_protein_parameters, compute_pi_mw) for protein analysis.\n"
            "Use materials tools (get_band_gap, get_density, is_metal) for materials.\n"
            "Use run_python(code) with pandas/numpy/scipy for custom analysis.\n"
            "Always report: result, units, interpretation, and limitations."
        ),
        "tools": SCIENCE_TOOLS,
        "model": build_model(temperature=0.1),
    },
    {
        "name": "physics-agent",
        "description": "Solve physics problems: acoustics, mechanics, thermodynamics, electromagnetism, optics, fluid dynamics, quantum mechanics, astronomy",
        "system_prompt": (
            "You are a Physics Agent.\n"
            "Use gym_search_tools(keyword) to find the right function, then run_gym_tool(name, '{\"param\": value}') to execute it.\n"
            "Available domains: acoustics, mechanics, thermodynamics, electromagnetism, optics, "
            "fluid dynamics, quantum mechanics, plasma physics, structural mechanics, atomic/molecular physics, astronomy.\n"
            "Always report: result, units, physical interpretation, and assumptions."
        ),
        "tools": SCIENCE_TOOLS,
        "model": build_model(temperature=0.1),
    },
    {
        "name": "experiment-agent",
        "description": "Design rigorous experimental protocols with independent/dependent variables, controls, sample sizes, and confounder analysis",
        "system_prompt": (
            "You are an Experiment Design Agent.\n"
            "For every experiment specify: independent variable, dependent variable, "
            "positive control, negative control, sample size (80% power), "
            "expected outcomes per hypothesis, confounders and mitigation, failure modes.\n"
            "Use the experiment-design skill template."
        ),
        "model": build_model(temperature=0.3),
    },
    {
        "name": "hypothesis-agent",
        "description": "Generate 2-4 distinct testable scientific hypotheses with mechanisms, supporting/refuting evidence, and prior probabilities",
        "system_prompt": (
            "You are a Hypothesis Generation Agent.\n"
            "Generate 2-4 distinct, testable hypotheses. For each:\n"
            "- Mechanistic explanation\n"
            "- What evidence supports it\n"
            "- What evidence would refute it\n"
            "- Prior probability (0-1)\n"
            "Compare alternatives and recommend the most likely given current evidence."
        ),
        "model": build_model(temperature=0.5),
    },
    {
        "name": "critic-agent",
        "description": "Review scientific outputs for unsupported claims, methodological flaws, statistical issues, and safety concerns",
        "system_prompt": (
            "You are a Scientific Critic.\n"
            "Identify: unsupported claims, methodological flaws, unaddressed confounders, "
            "logical fallacies, statistical issues (p-hacking, underpowered studies).\n"
            "Assess safety implications of proposed procedures.\n"
            "Rate overall scientific quality 1-5 with justification. Be rigorous but constructive."
        ),
        "model": build_model(temperature=0.1),
    },
    {
        "name": "debate-agent",
        "description": "Orchestrate a scientific debate on a controversial or multi-hypothesis question by dynamically spawning opposing-perspective agents and synthesizing their arguments",
        "system_prompt": (
            "You are a Scientific Debate Moderator.\n\n"
            "# MANDATORY PROTOCOL — do not deviate\n\n"
            "You MUST follow these steps in order. Skipping or shortcutting will produce a broken debate.\n\n"
            "Step 1 — Identify exactly **N distinct scientific positions** where 2 ≤ N ≤ 4.\n"
            "  Write them down in your reasoning: 'Position A: …', 'Position B: …', …\n\n"
            "Step 2 — **Call `spawn_agent` once PER position** — so you will make **at least 2, up to 4 spawn_agent tool calls**.\n"
            "  Example for N=3:\n"
            "    spawn_agent(role='scientist arguing Position A', task='<the user's question from A's angle>')\n"
            "    spawn_agent(role='scientist arguing Position B', task='<the user's question from B's angle>')\n"
            "    spawn_agent(role='scientist arguing Position C', task='<the user's question from C's angle>')\n"
            "  HARD RULE: you MUST NOT write any analysis, debate summary, or final answer\n"
            "  until every position has been sent to `spawn_agent` AND every spawned agent has returned.\n"
            "  If you have only called spawn_agent once so far, your next action MUST be another spawn_agent call (not prose).\n\n"
            "Step 3 — After ALL spawned agents return, present the debate:\n"
            "  - each position's strongest arguments\n"
            "  - key evidence for and against\n"
            "  - counterarguments\n\n"
            "Step 4 — Synthesize: which position has the strongest current evidence, and why.\n\n"
            "Be fair to all positions. Let evidence, not rhetoric, decide the winner."
        ),
        "tools": SCIENCE_TOOLS,
        "model": build_model(temperature=0.4),
    },
]


def get_agents_md_files() -> dict:
    content = _read_agents_md()
    if content:
        return {AGENTS_MD_VIRTUAL: create_file_data(content)}
    return {}


def _build_checkpointer() -> SqliteSaver:
    # check_same_thread=False so FastAPI's thread-pool workers can share it.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def create_science_agent():
    model = build_model()
    checkpointer = _build_checkpointer()
    store = InMemoryStore()

    agent = create_deep_agent(
        model=model,
        tools=SCIENCE_TOOLS,
        subagents=SUBAGENTS,
        skills=[SKILLS_DIR],
        memory=[AGENTS_MD_VIRTUAL],
        checkpointer=checkpointer,
        store=store,
    )

    return agent


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = create_science_agent()
    return _agent
