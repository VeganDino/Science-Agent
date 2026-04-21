"""Unified search across SciToolAgent KG and SciAgentGYM toolkit index."""
from langchain_core.tools import tool


@tool
def search_all_tools(keyword: str) -> str:
    """Search ALL available tools across both SciToolAgent (chemistry/biology/materials)
    and SciAgentGYM (physics/astronomy/statistics) with a single keyword.

    Use this as the first step when you don't know which tool to use.
    Returns tools from both sources ranked by relevance.

    Examples:
      search_all_tools('molecular weight')
      search_all_tools('doppler')
      search_all_tools('protein folding')
      search_all_tools('band gap')
      search_all_tools('thermodynamics')
    """
    from tools.kg_planner import search_tools_by_description
    from tools.gym_tools import _build_gym_index

    kw = keyword.lower()
    lines = [f"=== Tool search: '{keyword}' ===\n"]

    # ── SciToolAgent KG results ───────────────────────────────────────────────
    kg_results = search_tools_by_description(keyword, top_k=8)
    if kg_results:
        lines.append(f"[SciToolAgent KG — {len(kg_results)} matches]")
        for r in kg_results:
            inp = ", ".join(r.get("inputs", []))
            out = ", ".join(r.get("outputs", []))
            lines.append(f"  {r['tool']}  in:[{inp}] → out:[{out}]")
            desc = r.get("description", "")
            if desc:
                lines.append(f"    {desc[:80]}")
        lines.append("")

    # ── SciAgentGYM results ───────────────────────────────────────────────────
    gym_index = _build_gym_index()
    scored = []
    for name, info in gym_index.items():
        score = 0
        if kw in name.lower():
            score += 3
        if kw in info["docstring"].lower():
            score += 2
        if kw in info["subject"] or kw in info["topic"]:
            score += 1
        if score:
            scored.append((score, name, info))
    scored.sort(reverse=True)

    if scored:
        lines.append(f"[SciAgentGYM — {len(scored)} matches]")
        for _, name, info in scored[:10]:
            params = ", ".join(info["params"])
            tag = f"[{info['subject']}/{info['topic']}]"
            lines.append(f"  {name}({params})  {tag}")
            if info["docstring"]:
                lines.append(f"    {info['docstring'][:80]}")
        if len(scored) > 10:
            lines.append(f"  ... and {len(scored) - 10} more")
        lines.append("")

    if not kg_results and not scored:
        return (
            f"No tools found for '{keyword}'.\n"
            "Try broader keywords or use kg_category_tools / gym_search_tools directly."
        )

    lines.append(
        "To call a SciToolAgent tool: run_scitool(tool_name, input)\n"
        "To call a SciAgentGYM function: run_gym_tool(tool_name, json_args)"
    )
    return "\n".join(lines)


@tool
def plan_science_workflow(goal: str) -> str:
    """Given a scientific goal, suggest a complete workflow using tools from both
    SciToolAgent and SciAgentGYM.

    goal: describe what you want to compute or analyze in plain English
    Examples:
      plan_science_workflow('calculate drug-likeness of a compound from its name')
      plan_science_workflow('analyze protein sequence and predict structure')
      plan_science_workflow('compute acoustic properties of a material')
    """
    from tools.kg_planner import search_tools_by_description, find_tool_chain
    from tools.gym_tools import _build_gym_index

    kw = goal.lower()
    lines = [f"=== Workflow plan: '{goal}' ===\n"]

    # Extract key terms for search (simple word-level split)
    words = [w for w in kw.split() if len(w) > 3]

    # KG chain planning — try common input→output pairs
    chain_candidates = [
        ("molecule name", "molecular weight"),
        ("molecule name", "smiles"),
        ("smiles", "molecular weight"),
        ("smiles", "drug-likeness"),
        ("protein sequence", "molecular weight"),
        ("protein sequence", "pdb"),
        ("material formula", "band gap"),
        ("material formula", "density"),
    ]

    kg_chains_found = []
    for start, end in chain_candidates:
        if any(w in kw for w in start.split()) or any(w in kw for w in end.split()):
            chains = find_tool_chain(start, end, max_depth=4)
            if chains:
                kg_chains_found.append((start, end, chains[0]))

    if kg_chains_found:
        lines.append("[SciToolAgent KG — Tool chains]")
        for start, end, chain in kg_chains_found[:3]:
            lines.append(f"  {start} → {end}:")
            for step in chain:
                lines.append(f"    → run_scitool('{step}', ...)")
        lines.append("")

    # GYM functions relevant to the goal
    gym_index = _build_gym_index()
    scored = []
    for name, info in gym_index.items():
        score = sum(
            (3 if w in name.lower() else 0) +
            (2 if w in info["docstring"].lower() else 0) +
            (1 if w in info["subject"] or w in info["topic"] else 0)
            for w in words
        )
        if score:
            scored.append((score, name, info))
    scored.sort(reverse=True)

    if scored:
        lines.append("[SciAgentGYM — Relevant functions]")
        for _, name, info in scored[:8]:
            params = ", ".join(info["params"])
            lines.append(f"  run_gym_tool('{name}', '{{...}}')  # {info['docstring'][:60]}")
        lines.append("")

    # Direct shortcut tools
    shortcut_keywords = {
        "smiles": ["name_to_smiles", "smiles_to_weight", "get_crippen_descriptors", "calculate_tpsa"],
        "protein": ["compute_protein_parameters", "compute_pi_mw"],
        "dna": ["translate_dna", "find_orf", "get_reverse_complement"],
        "band gap": ["get_band_gap", "get_density", "is_metal"],
        "paper": ["download_papers", "paper_qa"],
        "python": ["run_python"],
    }
    shortcuts = []
    for kw_match, tools in shortcut_keywords.items():
        if kw_match in kw:
            shortcuts.extend(tools)

    if shortcuts:
        lines.append("[Shortcut tools available]")
        for s in shortcuts:
            lines.append(f"  {s}(...)")
        lines.append("")

    if not kg_chains_found and not scored and not shortcuts:
        lines.append(
            "No specific workflow found. Try:\n"
            "  search_all_tools(keyword) to explore available tools\n"
            "  kg_search_tools(keyword) for SciToolAgent KG\n"
            "  gym_search_tools(keyword) for SciAgentGYM"
        )

    return "\n".join(lines)


UNIFIED_TOOLS = [search_all_tools, plan_science_workflow]
