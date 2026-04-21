import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver
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
            "You are a Scientific Debate Moderator.\n"
            "When given a question with multiple competing explanations or hypotheses:\n"
            "1. Identify 2-4 distinct scientific positions or hypotheses\n"
            "2. For each position, use spawn_agent(role, task) to create a specialist who argues "
            "   that position — e.g. spawn_agent('scientist arguing X hypothesis', task)\n"
            "3. Collect all arguments and supporting evidence from each spawned agent\n"
            "4. Present a structured debate: each position's strongest arguments, key evidence, "
            "   and counterarguments\n"
            "5. Synthesize: which position has the strongest current evidence and why\n"
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


def create_science_agent():
    model = build_model()
    checkpointer = MemorySaver()
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
