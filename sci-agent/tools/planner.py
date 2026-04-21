"""KG-informed planning tool — KG BFDTS is mandatory, intent templates are hints only."""
from langchain_core.tools import tool


# ── Domain / intent detection ─────────────────────────────────────────────────

_DOMAIN_SIGNALS = {
    "chemistry": [
        "smiles", "molecule", "compound", "drug", "lipinski", "logp", "tpsa",
        "reaction", "synthesis", "retrosynthesis", "functional group", "fingerprint",
        "molar", "inchi", "selfies", "aspirin", "caffeine", "benzene",
        "pharmacokinetic", "admet", "toxicity", "patent", "similarity",
    ],
    "biology": [
        "protein", "peptide", "sequence", "dna", "rna", "amino acid", "gene",
        "genome", "blast", "alignment", "isoelectric", "translation", "orf",
        "codon", "expression", "transcription", "uniprot", "pdb",
        "cancer", "cell", "cells", "warburg", "metabol", "glycolysis",
        "mitochondria", "enzyme", "receptor", "pathway", "signaling",
    ],
    "materials": [
        "material", "crystal", "band gap", "density", "formation energy",
        "lattice", "symmetry", "space group", "cif", "perovskite", "oxide",
        "metallic", "conductor", "semiconductor", "tio2", "fe2o3", "lio",
    ],
    "physics": [
        "force", "energy", "pressure", "velocity", "acceleration", "momentum",
        "temperature", "frequency", "wave", "quantum", "field", "acoustic",
        "optic", "thermodynamic", "doppler", "entropy", "heat", "gravity",
        "electromagnetic", "photon", "electron", "nuclear", "plasma",
    ],
    "astronomy": [
        "star", "planet", "orbit", "galaxy", "cosmology", "redshift",
        "luminosity", "magnitude", "telescope", "spectral", "stellar",
        "solar", "black hole", "nebula",
    ],
    "statistics": [
        "regression", "correlation", "hypothesis test", "p-value", "anova",
        "distribution", "variance", "standard deviation", "bayesian", "confidence",
        "sample size", "power", "mean", "median", "chi-square",
    ],
}

# Maps (domain, intent) → shortcut tool steps.
# These are HINTS only — KG BFDTS is always the primary source.
_INTENT_HINTS = {
    ("chemistry", "drug-likeness"): {
        "triggers": ["lipinski", "drug-like", "admet", "rule of 5", "ro5"],
        "shortcuts": [
            ("name_to_smiles", "compound_name → SMILES"),
            ("smiles_to_weight", "SMILES → MW (≤500 Da)"),
            ("get_crippen_descriptors", "SMILES → LogP (≤5)"),
            ("get_hbd_count", "SMILES → H-bond donors (≤5)"),
            ("get_hba_count", "SMILES → H-bond acceptors (≤10)"),
            ("calculate_tpsa", "SMILES → TPSA (≤140 Å²)"),
        ],
        "note": "Lipinski RO5: MW≤500, LogP≤5, HBD≤5, HBA≤10, TPSA≤140.",
    },
    ("chemistry", "molecular-properties"): {
        "triggers": ["molecular descriptor", "descriptor", "property", "weight", "formula"],
        "shortcuts": [
            ("name_to_smiles", "compound_name → SMILES"),
            ("get_mol_formula", "SMILES → formula"),
            ("smiles_to_weight", "SMILES → MW"),
            ("get_crippen_descriptors", "SMILES → LogP, MR"),
            ("calculate_tpsa", "SMILES → TPSA"),
        ],
    },
    ("chemistry", "reaction"): {
        "triggers": ["reaction", "product", "reactant", "synthesis", "yield"],
        "shortcuts": [
            ("name_to_smiles", "compound_name → SMILES"),
            ("predict_reaction", "reactants SMILES → products"),
        ],
    },
    ("chemistry", "retrosynthesis"): {
        "triggers": ["retrosynthesis", "retrosynthetic", "synthetic route", "precursor"],
        "shortcuts": [
            ("name_to_smiles", "product_name → SMILES"),
            ("retrosynthesis", "product SMILES → pathways"),
        ],
    },
    ("biology", "protein-analysis"): {
        "triggers": ["protein", "peptide", "isoelectric", "instability", "gravy"],
        "shortcuts": [
            ("compute_protein_parameters", "sequence → MW, pI, instability, GRAVY"),
            ("compute_pi_mw", "sequence → pI, MW"),
        ],
    },
    ("biology", "dna-analysis"): {
        "triggers": ["dna", "translate", "orf", "reverse complement"],
        "shortcuts": [
            ("translate_dna", "DNA → protein"),
            ("get_reverse_complement", "DNA → reverse complement"),
            ("find_orf", "DNA → ORFs"),
        ],
    },
    ("materials", "properties"): {
        "triggers": ["band gap", "metallic", "density", "formation energy", "semiconductor"],
        "shortcuts": [
            ("get_band_gap", "formula → band gap (eV)"),
            ("is_metal", "formula → metallic yes/no"),
            ("get_density", "formula → density (g/cm³)"),
            ("get_formation_energy", "formula → formation energy per atom"),
        ],
    },
}

# KG chain templates — (start_type, end_type, trigger_keywords)
# Types must match actual KG input/output type names exactly.
_KG_CHAIN_TEMPLATES = [
    # Chemistry: name → SMILES → properties
    ("molecule name", "smiles",
     ["smiles", "compound", "molecule", "aspirin", "caffeine", "drug", "lipinski",
      "descriptor", "property", "weight", "logp", "tpsa", "formula", "reaction",
      "synthesis", "similarity", "retrosynthesis"]),
    ("smiles", "molecular weight",
     ["molecular weight", "smiles", "mass"]),
    ("smiles", "3d structure",
     ["3d", "pdb", "structure", "conformer", "geometry"]),
    ("smiles", "retrosynthetic pathway",
     ["retrosynthesis", "retrosynthetic", "pathway", "precursor"]),
    ("smiles", "molecular similarity",
     ["similarity", "similar", "tanimoto", "analog"]),
    ("smiles", "functional groups",
     ["functional group", "group", "moiety"]),
    # Biology: sequence → properties
    ("protein sequence", "molecular weight",
     ["protein", "peptide", "amino acid", "protein weight"]),
    ("protein sequence", "isoelectric point",
     ["protein", "peptide", "isoelectric", "isoelectric point"]),
    ("protein sequence", "3d structure",
     ["protein folding", "protein structure", "peptide pdb"]),
    ("dna sequence", "protein sequence",
     ["dna", "nucleotide", "translate", "codon"]),
    # Materials: formula → properties (KG type is "molecule formula" not "material formula")
    ("molecule formula", "band gap",
     ["band gap", "bandgap", "tio2", "material", "semiconductor", "metallic"]),
    ("molecule formula", "density",
     ["density", "tio2", "material", "crystal"]),
    ("molecule formula", "formation energy",
     ["formation energy", "tio2", "material", "stability"]),
    ("molecule formula", "crystal structure",
     ["crystal", "structure", "lattice", "space group", "tio2", "material"]),
]

_SUBAGENT_TRIGGERS = {
    "literature-agent":   ["paper", "study", "research", "pubmed", "arxiv", "evidence", "literature"],
    "compute-agent":      ["descriptor", "fingerprint", "cheminformatics", "computational"],
    "physics-agent":      ["force", "energy", "pressure", "velocity", "temperature", "frequency",
                           "wave", "quantum", "field", "acoustic", "optic", "thermodynamic", "doppler"],
    "experiment-agent":   ["experiment", "design", "protocol", "control", "sample size"],
    "hypothesis-agent":   ["hypothesis", "explain", "mechanism", "cause", "why", "warburg"],
    "debate-agent":       ["debate", "controversial", "competing", "versus", " vs ", "alternative"],
    "critic-agent":       ["review", "validate", "verify", "critique", "quality"],
}

_DOMAIN_GENERIC = {
    "molecular", "biological", "chemical", "material", "physical", "scientific",
    "structure", "data", "result", "value", "number", "based", "from", "into",
}

# Queries that are conceptual/reasoning — KG data-transformation chains don't apply.
# For these, the plan skips BFDTS and goes straight to literature + agent tools.
_CONCEPTUAL_SIGNALS = [
    "hypothes", "explain", "mechanism", "cause", "why ", "evidence",
    "review", "discuss", "debate", "theory", "understand", "reasoning",
    "what is the role", "how does", "what causes", "what could", "what might",
    "warburg", "critique", "implication", "significance",
]


def _detect_domain(kw: str) -> list[str]:
    hits = []
    for domain, signals in _DOMAIN_SIGNALS.items():
        if any(s in kw for s in signals):
            hits.append(domain)
    return hits or ["general"]


def _is_conceptual(kw: str) -> bool:
    return any(s in kw for s in _CONCEPTUAL_SIGNALS)


@tool
def make_science_plan(goal: str) -> str:
    """Build a structured execution plan by consulting the Tool Knowledge Graph and GYM index.

    Call this FIRST before executing any tools. KG BFDTS is always consulted.
    Returns a step-by-step plan with tool chains from the KG graph plus GYM functions.

    goal: plain-English description of what you want to achieve
    """
    from tools.kg_planner import (
        search_tools_by_description,
        bfdts_tool_chain,
        describe_decision_tree,
        decision_tree_to_dict,
        _build_indices,
    )
    from tools._bfdts_trace import set_trace

    _, _type_to_tools, _tool_to_outputs = _build_indices()

    def _chain_steps(chain, start_type, end_type):
        """Annotate each hop with its concrete input/output type."""
        steps = []
        current = start_type
        for tool in chain:
            outputs = _tool_to_outputs.get(tool, [])
            # Prefer the target type if this tool produces it; else first output.
            if end_type in outputs:
                out = end_type
            elif outputs:
                out = outputs[0]
            else:
                out = ""
            steps.append({"tool": tool, "input_type": current, "output_type": out})
            current = out
        return steps
    from tools.gym_tools import _build_gym_index

    kw = goal.lower()
    STOPWORDS = {
        "calculate", "compute", "get", "find", "what", "give", "show", "tell",
        "using", "with", "from", "into", "that", "this", "for", "the", "and",
        "analysis", "analyze", "determine", "obtain", "perform", "assess",
        "measure", "estimate", "evaluate", "predict",
    }
    words = [w.strip("().,?!") for w in kw.split()
             if len(w.strip("().,?!")) > 3 and w.strip("().,?!") not in STOPWORDS]

    sections = [f"## Science Plan: {goal}\n"]

    # ── 1. Domain detection ───────────────────────────────────────────────────
    domains = _detect_domain(kw)
    primary_domain = domains[0]

    all_intents = []
    for (d, intent_key), hint in _INTENT_HINTS.items():
        if d == primary_domain and any(t in kw for t in hint["triggers"]):
            all_intents.append((intent_key, hint))

    sections.append(
        f"**Domain**: {', '.join(domains)}  |  "
        f"**Intent**: {', '.join(i for i, _ in all_intents) or 'general'}\n"
    )

    # ── 2. KG BFDTS ───────────────────────────────────────────────────────────
    tool_info, _, _ = _build_indices()
    conceptual = _is_conceptual(kw)

    kw_words = set(kw.replace("(", " ").replace(")", " ").split())

    def _triggers_match(triggers: list[str]) -> bool:
        for t in triggers:
            if " " in t:
                if t in kw:
                    return True
            else:
                if t in kw_words:
                    return True
        return False

    # Each item: (start_type, end_type, chain_list, tree)
    kg_chain_candidates: list[tuple[str, str, list[str], object]] = []
    seen_chain_key: set[tuple] = set()

    sections.append("### Step 1 — KG BFDTS")

    if conceptual:
        # Conceptual/reasoning queries have no KG data-transformation chains.
        # Route directly to literature + specialist agents.
        sections.append(
            "*Conceptual query detected — KG data-transformation chains are not applicable.*\n"
            "*Approach: literature search + specialist agent reasoning.*"
        )
        sections.append("**Recommended tools for this query type:**")
        sections.append("- `download_papers(keyword)` — fetch recent papers from arXiv/PubMed")
        sections.append("- `paper_qa(question)` — answer questions from downloaded papers")
        sections.append("- `search_all_tools(keyword)` — search KG+GYM for any relevant tools")
    else:
        for start, end, triggers in _KG_CHAIN_TEMPLATES:
            if _triggers_match(triggers):
                solutions, tree = bfdts_tool_chain(start, end, max_depth=5, max_branches=3)
                # BFS returns shortest-first; collect top-3 unique per template
                for chain in solutions[:3]:
                    chain_key = tuple(chain)
                    if chain_key not in seen_chain_key:
                        seen_chain_key.add(chain_key)
                        kg_chain_candidates.append((start, end, chain, tree))

        if kg_chain_candidates:
            # Sort all candidates globally by chain length (shortest wins ties).
            kg_chain_candidates.sort(key=lambda x: len(x[2]))
            top_k = kg_chain_candidates[:3]

            sections.append(
                f"**Candidate chains (top {len(top_k)}, shortest first via BFDTS):**"
            )
            for i, (start, end, chain, _) in enumerate(top_k, 1):
                sections.append(
                    f"{i}. `{start}` → `{end}`  "
                    f"({len(chain)} steps): " +
                    " → ".join(f"`{t}`" for t in chain)
                )

            sections.append("")
            sections.append("**Execution strategy — execute ALL listed chains, then pick the best:**")
            sections.append(
                "1. Run each candidate chain above to completion, sequentially.\n"
                "2. Collect each chain's final output(s).\n"
                "3. In your synthesis, report how results compare:\n"
                "   - If chains agree (within tolerance) → **high confidence**, use the shortest chain's value.\n"
                "   - If chains disagree → **flag the discrepancy**, prefer the shortest chain, and note which alternatives produced which values.\n"
                "4. Cite the winning chain (by number above) in the final report."
            )

            # Decision tree for the shortest candidate
            start, end, chain, tree = top_k[0]
            tree_str = describe_decision_tree(tree)
            if tree_str.strip():
                sections.append(f"\n**Decision tree for chain #1 (`{start}` → `{end}`):**\n```")
                sections.append(tree_str[:700])
                sections.append("```")

            # Side-channel trace for UI animation (not included in tool output string)
            set_trace(
                {
                    "goal": goal,
                    "start": start,
                    "end": end,
                    "tree": decision_tree_to_dict(tree),
                    "candidates": [
                        {
                            "index": i + 1,
                            "from": s,
                            "to": e,
                            "step_count": len(c),
                            "tools": c,
                            "steps": _chain_steps(c, s, e),
                        }
                        for i, (s, e, c, _) in enumerate(top_k)
                    ],
                }
            )
        else:
            # KG keyword fallback
            domain_words = [w for w in words if w not in _DOMAIN_GENERIC]
            seen: set[str] = set()
            kg_keyword_hits = []
            for w in domain_words:
                for r in search_tools_by_description(w, top_k=3):
                    if r["tool"] not in seen:
                        seen.add(r["tool"])
                        tool_cat = r.get("category", "").lower()
                        if primary_domain == "chemistry" and any(
                            x in tool_cat for x in ["dna", "biology", "protein", "material"]
                        ):
                            continue
                        if primary_domain == "biology" and any(
                            x in tool_cat for x in ["material", "chemical"]
                        ):
                            continue
                        kg_keyword_hits.append(r)
            kg_keyword_hits = kg_keyword_hits[:6]
            if kg_keyword_hits and primary_domain in {"chemistry", "biology", "materials"}:
                sections.append("**KG keyword search (no BFDTS chain found):**")
                for r in kg_keyword_hits:
                    inp = ", ".join(r.get("inputs", []))
                    out = ", ".join(r.get("outputs", []))
                    sections.append(f"- `{r['tool']}` [{inp}→{out}]: {r.get('description','')[:70]}")
            else:
                sections.append(f"*No applicable KG chains for this {primary_domain} query.*")
    sections.append("")

    # ── 3. Shortcut tool hints (intent-matched) ───────────────────────────────
    shortcut_steps: list[str] = []
    if all_intents:
        sections.append("### Step 2 — Shortcut Tools (intent-matched)")
        seen_shortcuts: set[str] = set()
        for intent_key, hint in all_intents:
            sections.append(f"**{intent_key}:**")
            for tool_name, desc in hint["shortcuts"]:
                sections.append(f"- `{tool_name}` — {desc}")
                if tool_name not in seen_shortcuts:
                    seen_shortcuts.add(tool_name)
                    shortcut_steps.append(tool_name)
            if "note" in hint:
                sections.append(f"> {hint['note']}")
        sections.append("")

    # ── 4. GYM function discovery ─────────────────────────────────────────────
    gym_index = _build_gym_index()
    DOMAIN_TO_SUBJECT = {
        "physics":    ["physics"],
        "astronomy":  ["astronomy"],
        "statistics": ["statistics"],
        "chemistry":  ["chemistry"],
        "materials":  ["materials_science"],
        "biology":    ["life_science"],
    }
    allowed_subjects = DOMAIN_TO_SUBJECT.get(primary_domain, sum(DOMAIN_TO_SUBJECT.values(), []))

    scored = []
    for name, info in gym_index.items():
        if info["subject"] not in allowed_subjects:
            continue
        score = sum(
            (5 if w in name.lower() else 0) +
            (2 if w in info["docstring"].lower() else 0) +
            (1 if w in info["topic"] else 0)
            for w in words
        )
        if score >= 5 and any(w in name.lower() for w in words):
            scored.append((score, name, info))
    scored.sort(reverse=True)

    if scored:
        step_label = "Step 3" if all_intents else "Step 2"
        sections.append(f"### {step_label} — SciAgentGYM Functions")
        for _, name, info in scored[:6]:
            params = ", ".join(info["params"])
            tag = f"[{info['subject']}/{info['topic']}]"
            sections.append(f"- `{name}({params})` {tag}: {info['docstring'][:70]}")
        sections.append("")

    # ── 5. Subagent recommendations — only for complex/non-computational queries ──
    # Don't recommend agents for pure calculation tasks (shortcut_steps handle it)
    _COMPLEX_SIGNALS = [
        "paper", "study", "research", "evidence", "literature",
        "hypothesis", "explain", "mechanism", "why", "debate",
        "review", "validate", "critique", "experiment", "design",
        "warburg", "force", "wave", "quantum", "doppler",
    ]
    is_complex = any(s in kw for s in _COMPLEX_SIGNALS) or not shortcut_steps
    recommended = [
        agent for agent, triggers in _SUBAGENT_TRIGGERS.items()
        if any(t in kw for t in triggers)
    ] if is_complex else []

    if recommended:
        sections.append("### Suggested Subagents")
        for a in recommended:
            sections.append(f"- `{a}`")
        sections.append("")

    # ── 6. Execution Plan — minimal, no duplicates ────────────────────────────
    sections.append("### Execution Plan")
    sections.append(f"> Target: ≤{max(6, len(shortcut_steps) + 2)} tool calls\n")
    step = 1

    if conceptual:
        sections.append(f"{step}. `download_papers(keyword)`")
        step += 1
        sections.append(f"{step}. `paper_qa(question)`")
        step += 1
        for a in recommended[:2]:
            sections.append(f"{step}. `spawn_agent('{a}', ...)`")
            step += 1
    else:
        if shortcut_steps:
            # Shortcuts exist → skip KG chain steps (they overlap with shortcuts)
            # Just reference KG as context, don't list as separate steps
            if kg_chain_candidates:
                kg_note = ", ".join(f"`{c[0]}`" for c in kg_chain_candidates[:2])
                sections.append(f"*(KG chains identified: {kg_note} — covered by shortcuts below)*")
            added: set[str] = set()
            for tool_name in shortcut_steps:
                if tool_name not in added:
                    added.add(tool_name)
                    sections.append(f"{step}. `{tool_name}(...)`")
                    step += 1
        elif kg_chain_candidates:
            # No shortcuts → use KG chain steps directly
            for start, end, chain, _ in kg_chain_candidates[:3]:
                chain_str = " → ".join(f"`run_scitool('{t}', ...)`" for t in chain)
                sections.append(f"{step}. [{start}→{end}]: {chain_str}")
                step += 1
        elif scored:
            # Physics/GYM only — no shortcuts, no KG chains
            for _, name, _ in scored[:3]:
                sections.append(f"{step}. `run_gym_tool('{name}', '{{...}}')`")
                step += 1

        for a in recommended[:1]:
            sections.append(f"{step}. `spawn_agent('{a}', ...)`")
            step += 1

    sections.append(f"{step}. Synthesize → structured scientific report")

    if len(sections) <= 4:
        return (
            f"No tools found in KG or GYM for: '{goal}'.\n"
            "Use search_all_tools(keyword) to explore manually."
        )

    return "\n".join(sections)
