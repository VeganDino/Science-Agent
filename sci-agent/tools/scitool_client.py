"""Direct adapter for SciToolAgent tools — no HTTP service needed."""
import sys
from pathlib import Path

SCITOOL_AGENT_PATH = Path("/home/ubuntu/workspace/SciToolAgent/ToolsAgent")
if str(SCITOOL_AGENT_PATH) not in sys.path:
    sys.path.insert(0, str(SCITOOL_AGENT_PATH))

_chemical: dict = {}
_biology: dict = {}
_material: dict = {}
_general: dict = {}


def _load_category(category: str) -> dict:
    try:
        if category == "Chemical":
            global _chemical
            if not _chemical:
                from ToolsFuns.Chemical.tool_name_dict import CHEMICAL_TOOLS_DICT
                _chemical = CHEMICAL_TOOLS_DICT
            return _chemical
        elif category == "Biology":
            global _biology
            if not _biology:
                from ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DICT
                _biology = BIOLOGY_TOOLS_DICT
            return _biology
        elif category == "Material":
            global _material
            if not _material:
                from ToolsFuns.Material.tool_name_dict import MATERIAL_TOOLS_DICT
                _material = MATERIAL_TOOLS_DICT
            return _material
        elif category == "General":
            global _general
            if not _general:
                from ToolsFuns.General.tool_name_dict import GENERAL_TOOLS_DICT
                _general = GENERAL_TOOLS_DICT
            return _general
    except Exception as e:
        print(f"[SciToolClient] Failed to load {category} tools: {e}")
    return {}


def call_tool(tool_name: str, tool_args: str, category: str = "") -> dict:
    from tools.registry import _tools_mapping, _load
    _load()

    if not category:
        category_key = _tools_mapping.get(tool_name, "")
        # "ChemicalTools" → "Chemical"
        category = category_key.replace("Tools", "") if category_key else ""

    tool_dict = _load_category(category)
    fn = tool_dict.get(tool_name)

    if fn is None:
        return {"success": False, "error": f"Tool '{tool_name}' not available (missing dependency or not found)"}

    try:
        result = fn(tool_args)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}
