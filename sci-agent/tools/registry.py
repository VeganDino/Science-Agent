"""Tool registry backed by SciToolAgent's tool list and Knowledge Graph."""
import json
import sys
from pathlib import Path

SCITOOL_ROOT = Path("/home/ubuntu/workspace/SciToolAgent")
KG_PATH = SCITOOL_ROOT / "KG" / "storage_graph_large" / "graph_store.json"

# Add SciToolAgent to path so we can import its tools_dict
sys.path.insert(0, str(SCITOOL_ROOT / "ToolsAgent"))

_kg: dict[str, list] = {}
_tools_mapping: dict[str, str] = {}  # tool_name -> category


def _load():
    global _kg, _tools_mapping
    if _kg:
        return

    # Load KG
    with open(KG_PATH) as f:
        _kg = json.load(f).get("graph_dict", {})

    # Load tool-to-category mapping from SciToolAgent
    try:
        from ToolsFuns.tools_dict import TOOLS_MAPPING
        for category, names in TOOLS_MAPPING.items():
            for name in names:
                _tools_mapping[name] = category
    except ImportError:
        pass


def get_tool_info(tool_name: str) -> dict:
    _load()
    triplets = _kg.get(tool_name, [])
    info = {"tool_id": tool_name, "category": _tools_mapping.get(tool_name, "Unknown")}
    for subj, obj in triplets:
        if subj == "is a":
            info["type"] = obj
        elif subj == "has the functionality that":
            info["description"] = obj
        elif subj == "inputs":
            info.setdefault("inputs", []).append(obj)
        elif subj == "outputs":
            info.setdefault("outputs", []).append(obj)
        elif subj == "is sourced from":
            info["source"] = obj
        elif subj in ("does not need", "requires"):
            info["security_check"] = subj != "does not need"
    return info


def get_all_tool_names() -> list[str]:
    _load()
    return list(_tools_mapping.keys())


def get_tools_by_category(category: str) -> list[str]:
    _load()
    return [name for name, cat in _tools_mapping.items() if category.lower() in cat.lower()]


def search_tools_by_keyword(keyword: str) -> list[dict]:
    _load()
    keyword_lower = keyword.lower()
    results = []
    for tool_name, triplets in _kg.items():
        if tool_name not in _tools_mapping:
            continue
        for pred, obj in triplets:
            if keyword_lower in obj.lower() or keyword_lower in tool_name.lower():
                results.append(get_tool_info(tool_name))
                break
    return results[:10]


def get_input_tools(input_type: str) -> list[str]:
    _load()
    results = []
    for tool_name, triplets in _kg.items():
        if tool_name not in _tools_mapping:
            continue
        for pred, obj in triplets:
            if pred == "inputs" and input_type.lower() in obj.lower():
                results.append(tool_name)
    return results


def get_output_tools(output_type: str) -> list[str]:
    _load()
    results = []
    for tool_name, triplets in _kg.items():
        if tool_name not in _tools_mapping:
            continue
        for pred, obj in triplets:
            if pred == "outputs" and output_type.lower() in obj.lower():
                results.append(tool_name)
    return results


_TASK_KEYWORDS = {
    "literature_review": ["paper", "search", "download", "qa", "query", "find"],
    "hypothesis_generation": ["predict", "design", "generate", "random"],
    "experiment_design": ["design", "protocol", "reaction", "synthesis", "fold", "predict"],
    "data_analysis": ["calculate", "compute", "analyze", "statistics", "weight", "descriptor"],
    "calculation": ["calculate", "compute", "weight", "formula", "molecular", "energy", "property"],
    "critique": ["check", "validate", "safety", "toxicity", "patent", "verify"],
}


def plan_tool_chain(domain: str, task_description: str) -> list[dict]:
    """Return up to 5 relevant tools for a given domain and task."""
    _load()
    domain_map = {
        "chemistry": "Chemical",
        "biology": "Biology",
        "materials": "Material",
        "general": "General",
        "physics": "General",
        "medicine": "Biology",
        "environmental": "Chemical",
    }
    category = domain_map.get(domain.lower(), "General")
    candidates = get_tools_by_category(category)

    # Expand keywords: combine task_description words + task-type synonyms
    base_keywords = task_description.lower().replace("_", " ").split()
    task_synonyms = _TASK_KEYWORDS.get(task_description.lower(), [])
    keywords = set(base_keywords + task_synonyms)

    scored = []
    for name in candidates:
        info = get_tool_info(name)
        desc = (info.get("description", "") + " " + name).lower()
        score = sum(1 for kw in keywords if kw in desc)
        scored.append((score, info))

    scored.sort(key=lambda x: -x[0])
    # Return top 5, fall back to first 5 in category if all scores are 0
    top = [info for _, info in scored[:5]]
    if all(s == 0 for s, _ in scored[:5]):
        top = [get_tool_info(n) for n in candidates[:5]]
    return top
