"""LangChain @tool wrappers for SciToolAgent functions."""
import sys
from pathlib import Path
from langchain_core.tools import tool

SCITOOL_PATH = Path("/home/ubuntu/workspace/SciToolAgent/ToolsAgent")
if str(SCITOOL_PATH) not in sys.path:
    sys.path.insert(0, str(SCITOOL_PATH))

# ── lazy loaders ────────────────────────────────────────────────────────────

_chem = _bio = _mat = _gen = None

def _chemical():
    global _chem
    if _chem is None:
        from ToolsFuns.Chemical.tool_name_dict import CHEMICAL_TOOLS_DICT
        _chem = CHEMICAL_TOOLS_DICT
    return _chem

def _biology():
    global _bio
    if _bio is None:
        from ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DICT
        _bio = BIOLOGY_TOOLS_DICT
    return _bio

def _material():
    global _mat
    if _mat is None:
        from ToolsFuns.Material.tool_name_dict import MATERIAL_TOOLS_DICT
        _mat = MATERIAL_TOOLS_DICT
    return _mat

def _general():
    global _gen
    if _gen is None:
        from ToolsFuns.General.tool_name_dict import GENERAL_TOOLS_DICT
        _gen = GENERAL_TOOLS_DICT
    return _gen

def _call(registry_fn, name: str, arg: str) -> str:
    try:
        fn = registry_fn().get(name)
        if fn is None:
            return f"Tool '{name}' not available."
        return str(fn(arg))
    except Exception as e:
        return f"Tool error: {e}"


# ── Generic dispatcher ───────────────────────────────────────────────────────

@tool
def run_scitool(tool_name: str, tool_input: str) -> str:
    """Run any SciToolAgent tool by name.
    tool_name: exact SciToolAgent tool name (e.g. 'SMILESToWeight', 'ComputeProtPara')
    tool_input: the input string for the tool
    """
    from tools.registry import _tools_mapping, _load
    _load()
    category = _tools_mapping.get(tool_name, "")
    if "Chemical" in category:
        return _call(_chemical, tool_name, tool_input)
    elif "Biology" in category:
        return _call(_biology, tool_name, tool_input)
    elif "Material" in category:
        return _call(_material, tool_name, tool_input)
    elif "General" in category:
        return _call(_general, tool_name, tool_input)
    # try all
    for fn in (_chemical, _biology, _material, _general):
        try:
            result = fn().get(tool_name)
            if result:
                return str(result(tool_input))
        except Exception:
            pass
    return f"Tool '{tool_name}' not found in any category."


# ── Chemistry shortcuts ──────────────────────────────────────────────────────

@tool
def name_to_smiles(compound_name: str) -> str:
    """Convert a compound name (e.g. 'aspirin', 'caffeine') to its SMILES string."""
    return _call(_chemical, "NameToSMILES", compound_name)

@tool
def smiles_to_weight(smiles: str) -> str:
    """Calculate the molecular weight of a compound from its SMILES string."""
    return _call(_chemical, "SMILESToWeight", smiles)

@tool
def get_mol_formula(smiles: str) -> str:
    """Get the molecular formula from a SMILES string."""
    return _call(_chemical, "GetMolFormula", smiles)

@tool
def get_crippen_descriptors(smiles: str) -> str:
    """Calculate LogP and molar refractivity (MR) from SMILES using Crippen method."""
    return _call(_chemical, "GetCrippenDescriptors", smiles)

@tool
def calculate_tpsa(smiles: str) -> str:
    """Calculate topological polar surface area (TPSA) from SMILES."""
    return _call(_chemical, "CalculateTPSA", smiles)

@tool
def get_hbd_count(smiles: str) -> str:
    """Get the number of hydrogen bond donors (HBD) from a SMILES string. Used in Lipinski's Rule of 5."""
    return _call(_chemical, "GetHBDNum", smiles)

@tool
def get_hba_count(smiles: str) -> str:
    """Get the number of hydrogen bond acceptors (HBA) from a SMILES string. Used in Lipinski's Rule of 5."""
    return _call(_chemical, "GetHBANum", smiles)

@tool
def get_rotatable_bonds(smiles: str) -> str:
    """Get the number of rotatable bonds from a SMILES string."""
    return _call(_chemical, "GetRotatableBondsNum", smiles)

@tool
def get_functional_groups(smiles: str) -> str:
    """List functional groups present in a molecule given its SMILES."""
    return _call(_chemical, "FuncGroups", smiles)

@tool
def mol_similarity(smiles_pair: str) -> str:
    """Calculate Tanimoto similarity between two molecules.
    Input format: 'SMILES1 SMILES2' (space-separated)
    """
    return _call(_chemical, "MolSimilarity", smiles_pair)

@tool
def check_safety(smiles: str) -> str:
    """Get safety summary for a molecule (toxicity, hazards) from its SMILES."""
    return _call(_chemical, "SafetySummary", smiles)

@tool
def predict_reaction(reactants: str) -> str:
    """Predict reaction products from reactants SMILES (IBM RXN4Chemistry)."""
    return _call(_chemical, "RXNPredict", reactants)

@tool
def retrosynthesis(product_smiles: str) -> str:
    """Predict retrosynthetic pathway for a target molecule SMILES."""
    return _call(_chemical, "RXNRetrosynthetic", product_smiles)


# ── Biology shortcuts ────────────────────────────────────────────────────────

@tool
def compute_protein_parameters(sequence: str) -> str:
    """Compute protein physicochemical parameters: MW, pI, instability index,
    GRAVY score, and amino acid composition from a protein sequence."""
    return _call(_biology, "ComputeProtPara", sequence)

@tool
def compute_pi_mw(sequence: str) -> str:
    """Calculate isoelectric point (pI) and molecular weight of a protein/peptide sequence."""
    return _call(_biology, "ComputePiMw", sequence)

@tool
def translate_dna(dna_sequence: str) -> str:
    """Translate a DNA sequence to its amino acid (protein) sequence."""
    return _call(_biology, "TranslateDNAtoAminoAcidSequence", dna_sequence)

@tool
def get_reverse_complement(dna_sequence: str) -> str:
    """Get the reverse complement of a DNA sequence."""
    return _call(_biology, "GetReverseComplement", dna_sequence)

@tool
def find_orf(dna_sequence: str) -> str:
    """Find open reading frames (ORFs) in a DNA sequence."""
    return _call(_biology, "ORFFind", dna_sequence)

@tool
def sequence_alignment(sequences: str) -> str:
    """Perform global pairwise sequence alignment.
    Input format: 'SEQUENCE1 SEQUENCE2' (space-separated)
    """
    return _call(_biology, "DoubleSequenceGlobalAlignment", sequences)


# ── Materials shortcuts ──────────────────────────────────────────────────────

@tool
def get_band_gap(formula: str) -> str:
    """Get the band gap of a material by its chemical formula using Materials Project."""
    return _call(_material, "GetBandGapByFormula", formula)

@tool
def get_density(formula: str) -> str:
    """Get the density of a material by its chemical formula using Materials Project."""
    return _call(_material, "GetDensityByFormula", formula)

@tool
def get_formation_energy(formula: str) -> str:
    """Get formation energy per atom of a material by chemical formula."""
    return _call(_material, "GetFormationEnergyPerAtomByFormula", formula)

@tool
def is_metal(formula: str) -> str:
    """Check if a material is metallic by its chemical formula."""
    return _call(_material, "IsMetalByFormula", formula)

@tool
def search_materials(elements: str) -> str:
    """Search Materials Project for materials containing specified elements.
    Input: comma-separated element symbols, e.g. 'Fe,O' or 'Li,Mn,O'
    """
    return _call(_material, "SearchMaterialsContainingElements", elements)

@tool
def get_structure_info(formula_or_cif: str) -> str:
    """Get crystal structure information for a material formula or CIF content."""
    return _call(_material, "GetStructureInfo", formula_or_cif)

@tool
def calculate_symmetry(structure: str) -> str:
    """Calculate space group and symmetry of a crystal structure."""
    return _call(_material, "CalculateSymmetry", structure)


# ── OCR tool ─────────────────────────────────────────────────────────────────

import os as _os
import json as _json

OCR_SERVICE_URL = _os.environ.get("OCR_SERVICE_URL", "http://localhost:8788")
_WORKSPACE_DIR = Path(__file__).parent.parent / "workspace"


def _stringify_ocr_result(result) -> str:
    """NemotronOCRV2 may return a string, list[str], list[dict], or nested dict.
    Reduce to a readable string for the agent."""
    if result is None:
        return "(no result)"
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("content")
                parts.append(str(txt) if txt else _json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(no text detected)"
    if isinstance(result, dict):
        txt = result.get("text") or result.get("content")
        if txt:
            return str(txt)
        return _json.dumps(result, ensure_ascii=False)[:5000]
    return str(result)


@tool
def ocr_image(image_path: str, merge_level: str = "paragraph") -> str:
    """Extract text from an image or PDF using the Nemotron OCR V2 service (English).

    image_path: absolute path, or a filename inside sci-agent/workspace/.
    merge_level: "line" | "paragraph" | "page" — granularity of the merged text.
    Returns the extracted text.
    """
    import requests
    try:
        path = Path(image_path)
        if not path.is_absolute():
            path = _WORKSPACE_DIR / image_path
        if not path.exists():
            return f"OCR error: file not found at {path}"
        with open(path, "rb") as f:
            resp = requests.post(
                f"{OCR_SERVICE_URL}/ocr",
                files={"file": (path.name, f)},
                data={"merge_level": merge_level},
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"OCR error: {data['error']}"
        return _stringify_ocr_result(data.get("result"))
    except Exception as e:
        return f"OCR service error: {e}"


# ── General tools ─────────────────────────────────────────────────────────────

@tool
def download_papers(keyword: str) -> str:
    """Download recent papers from arXiv by keyword and return titles and URLs."""
    return _call(_general, "DownloadPapers", keyword)

@tool
def paper_qa(question: str) -> str:
    """Answer a question using downloaded PDF papers in the TempFiles directory."""
    return _call(_general, "PaperQA", question)

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output. Useful for calculations, statistics,
    and data analysis. pandas, numpy, scipy, sklearn, matplotlib are available."""
    import io, sys, traceback
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        import pandas as pd, numpy as np, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
        from sklearn.linear_model import LinearRegression
        exec(code, {"pd": pd, "np": np, "plt": plt, "stats": stats,
                    "LinearRegression": LinearRegression, "__builtins__": __builtins__})
        return buf.getvalue() or "(no output)"
    except Exception:
        return traceback.format_exc()
    finally:
        sys.stdout = old


# ── KG-powered tools ─────────────────────────────────────────────────────────

@tool
def kg_search_tools(keyword: str) -> str:
    """Search the Tool Knowledge Graph for tools matching a keyword.
    Returns tool names, descriptions, input/output types.
    Use this to discover which tool to call for a given task.
    Example: kg_search_tools('molecular weight') or kg_search_tools('protein folding')
    """
    from tools.kg_planner import search_tools_by_description
    results = search_tools_by_description(keyword, top_k=8)
    if not results:
        return f"No tools found for keyword: {keyword}"
    lines = [f"Tools matching '{keyword}':"]
    for r in results:
        inp = ", ".join(r.get("inputs", []))
        out = ", ".join(r.get("outputs", []))
        lines.append(f"  {r['tool']} | in: [{inp}] → out: [{out}]")
        lines.append(f"    {r.get('description', '')}")
    return "\n".join(lines)


@tool
def kg_plan_chain(start_input_type: str, target_output_type: str) -> str:
    """Plan a tool chain using the Knowledge Graph.
    Given a starting data type and desired output type, returns the sequence of tools to call.
    Examples:
      kg_plan_chain('molecule name', 'molecular weight')
      kg_plan_chain('smiles', '3d structure')
      kg_plan_chain('protein sequence', 'pdb')
    """
    from tools.kg_planner import find_tool_chain, describe_tool_chain
    chains = find_tool_chain(start_input_type, target_output_type, max_depth=4)
    if not chains:
        return f"No tool chain found from '{start_input_type}' to '{target_output_type}'.\nTry kg_search_tools to find individual tools."
    lines = [f"Tool chains from [{start_input_type}] → [{target_output_type}]:"]
    for i, chain in enumerate(chains[:5], 1):
        lines.append(f"\nChain {i}:")
        lines.append(describe_tool_chain(chain))
    return "\n".join(lines)


@tool
def kg_next_tools(output_type: str) -> str:
    """Given a tool's output type, suggest which tools to call next.
    Example: kg_next_tools('SMILES') to see what tools accept SMILES as input.
    """
    from tools.kg_planner import suggest_next_tools
    results = suggest_next_tools(output_type, exclude_security=True)
    if not results:
        return f"No tools found that accept '{output_type}' as input."
    lines = [f"Tools that accept '{output_type}' as input:"]
    for r in results[:10]:
        out = ", ".join(r.get("outputs", []))
        lines.append(f"  {r['tool']} → [{out}]: {r.get('description', '')[:60]}")
    return "\n".join(lines)


@tool
def kg_category_tools(category: str) -> str:
    """List all available tools in a category from the Knowledge Graph.
    Categories: Chemical, Biological, Material, General
    """
    from tools.kg_planner import get_tools_by_category
    results = get_tools_by_category(category)
    if not results:
        return f"No tools found for category: {category}"
    lines = [f"{category} tools ({len(results)} total):"]
    for r in results[:20]:
        lines.append(f"  {r['tool']}: {r.get('description', '')[:60]}")
    if len(results) > 20:
        lines.append(f"  ... and {len(results)-20} more. Use kg_search_tools for specific tools.")
    return "\n".join(lines)


# ── Tool list for agent registration ─────────────────────────────────────────

from tools.dynamic_agent import spawn_agent
from tools.gym_tools import GYM_TOOLS
from tools.unified_search import UNIFIED_TOOLS
from tools.planner import make_science_plan

SCIENCE_TOOLS = [
    # KG-informed planning (call first)
    make_science_plan,
    # Dynamic agent spawning (for debate, ad-hoc specialists, etc.)
    spawn_agent,
    # Unified cross-source search
    *UNIFIED_TOOLS,
    # KG navigation (SciToolAgent)
    kg_search_tools, kg_plan_chain, kg_next_tools, kg_category_tools,
    # Generic dispatcher
    run_scitool,
    # Chemistry shortcuts
    name_to_smiles, smiles_to_weight, get_mol_formula,
    get_crippen_descriptors, calculate_tpsa,
    get_hbd_count, get_hba_count, get_rotatable_bonds,
    get_functional_groups, mol_similarity, check_safety,
    predict_reaction, retrosynthesis,
    # Biology shortcuts
    compute_protein_parameters, compute_pi_mw,
    translate_dna, get_reverse_complement, find_orf, sequence_alignment,
    # Materials shortcuts
    get_band_gap, get_density, get_formation_energy,
    is_metal, search_materials, get_structure_info, calculate_symmetry,
    # General
    download_papers, paper_qa, run_python, ocr_image,
    # SciAgentGYM (physics/chemistry/materials/astronomy/statistics)
    *GYM_TOOLS,
]
