"""Dynamically spawn specialist sub-agents on demand."""
from langchain_core.tools import tool

VLLM_URL = "http://localhost:8000/v1"

_ROLE_HINTS = {
    "computational chemist": (
        "Use name_to_smiles, smiles_to_weight, get_crippen_descriptors, calculate_tpsa, "
        "get_functional_groups, mol_similarity, check_safety, predict_reaction, retrosynthesis. "
        "Start with name_to_smiles if given a compound name."
    ),
    "bioinformatician": (
        "Use compute_protein_parameters, compute_pi_mw, translate_dna, get_reverse_complement, "
        "find_orf, sequence_alignment, run_scitool for biology tools. "
        "Use kg_search_tools('protein') or kg_search_tools('dna') to find additional tools."
    ),
    "materials scientist": (
        "Use get_band_gap, get_density, get_formation_energy, is_metal, search_materials, "
        "get_structure_info, calculate_symmetry for Materials Project queries. "
        "Use gym_search_tools('crystal') or gym_search_tools('spectroscopy') for structure analysis."
    ),
    "physicist": (
        "Use gym_search_tools(keyword) to find physics functions, then run_gym_tool(name, json_args). "
        "Available domains: acoustics, mechanics, thermodynamics, electromagnetism, optics, "
        "fluid dynamics, quantum mechanics, plasma physics, structural mechanics."
    ),
    "astronomer": (
        "Use gym_search_tools(keyword) to find astronomy functions, then run_gym_tool(name, json_args). "
        "Available: stellar physics, orbital mechanics, cosmology, spectroscopy, telescope calculations."
    ),
    "statistician": (
        "Use gym_search_tools(keyword) for statistical functions or run_python(code) with "
        "pandas/numpy/scipy/sklearn. Always report: test statistic, p-value, effect size, "
        "confidence interval, and assumptions checked."
    ),
    "literature reviewer": (
        "Use download_papers(keyword) to fetch arXiv papers, paper_qa(question) to query them. "
        "Extract key claims, methods, evidence quality. Rate: RCT > cohort > case study > preprint."
    ),
    "experiment designer": (
        "Design rigorous protocols: state IV, DV, controls (positive + negative), "
        "sample size for 80% power, expected outcomes, confounders, failure modes."
    ),
    "hypothesis generator": (
        "Generate 2-4 distinct testable hypotheses. For each: mechanism, supporting evidence, "
        "refuting evidence, prior probability. Compare and recommend the most likely."
    ),
    "scientific critic": (
        "Identify unsupported claims, methodological flaws, confounders, logical fallacies, "
        "statistical issues (p-hacking, underpowered). Rate quality 1-5 with justification."
    ),
    "data analyst": (
        "Use run_python(code) with pandas/numpy/scipy/sklearn/matplotlib. "
        "Read CSVs, run stats, generate plots. Always show code and intermediate results."
    ),
}

_DEFAULT_HINT = (
    "Use search_all_tools(keyword) to discover relevant tools from both SciToolAgent and SciAgentGYM. "
    "Use run_scitool(tool_name, input) for SciToolAgent tools, "
    "run_gym_tool(tool_name, json_args) for SciAgentGYM functions."
)


@tool
def spawn_agent(role: str, task: str) -> str:
    """Dynamically spawn a specialist agent to handle a specific sub-task.

    Use this when a focused expert is needed for part of the problem.

    role: specialist description — choose from or describe your own:
      Chemistry:    'computational chemist'
      Biology:      'bioinformatician'
      Materials:    'materials scientist'
      Physics:      'physicist'
      Astronomy:    'astronomer'
      Statistics:   'statistician'
      Literature:   'literature reviewer'
      Design:       'experiment designer'
      Hypothesis:   'hypothesis generator'
      Review:       'scientific critic'
      Data:         'data analyst'

    task: the specific question or task for the specialist
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    from tools.scitool_tools import SCIENCE_TOOLS

    # Find role-specific tool hints (case-insensitive partial match)
    hint = _DEFAULT_HINT
    for key, val in _ROLE_HINTS.items():
        if key in role.lower():
            hint = val
            break

    model = ChatOpenAI(
        model="model",
        base_url=VLLM_URL,
        api_key="none",
        temperature=0.2,
    )

    system_prompt = (
        f"You are a specialist {role}.\n"
        f"Tool guidance: {hint}\n"
        "Complete the assigned task thoroughly using available tools.\n"
        "Report findings with: results, units, interpretation, and limitations.\n"
        "Be concise but complete."
    )

    agent = create_react_agent(model, SCIENCE_TOOLS, prompt=system_prompt)

    try:
        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                return f"[{role}]\n{msg.content}"
        return f"[{role}] No output produced."
    except Exception as e:
        import traceback
        return f"[{role}] Error: {e}\n{traceback.format_exc()}"
