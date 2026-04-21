"""LangChain tool wrappers for SciAgentGYM toolkit functions.

Uses AST-based indexing for fast startup, with on-demand dynamic loading per call.
"""
import ast
import sys
import json
import importlib.util
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any
from langchain_core.tools import tool

GYM_ROOT = Path("/home/ubuntu/workspace/SciAgentGYM")
TOOLKITS_ROOT = GYM_ROOT / "toolkits"

if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))

_EXCLUDE = {"__", "test_", "convert_", "_tools_gym"}


@lru_cache(maxsize=1)
def _build_gym_index() -> Dict[str, Dict]:
    """Scan all toolkit Python files with AST — no imports, fast startup."""
    index: Dict[str, Dict] = {}

    for py_file in sorted(TOOLKITS_ROOT.rglob("*.py")):
        if any(py_file.name.startswith(p) for p in _EXCLUDE):
            continue

        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            continue

        rel_path = py_file.relative_to(GYM_ROOT)
        parts = rel_path.parts  # ('toolkits', subject, topic, file.py)
        subject = parts[1] if len(parts) > 1 else ""
        topic = parts[2] if len(parts) > 2 else ""

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue

            # First line of docstring only
            docstring = ""
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                raw = val.s if isinstance(val, ast.Str) else (
                    val.value if isinstance(val, ast.Constant) and isinstance(val.value, str) else ""
                )
                docstring = raw.strip().splitlines()[0] if raw else ""

            params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]

            # On name collision keep the first occurrence (earlier files win)
            if node.name not in index:
                index[node.name] = {
                    "file": str(rel_path),
                    "subject": subject,
                    "topic": topic,
                    "docstring": docstring,
                    "params": params,
                }

    return index


def _load_and_call(func_name: str, kwargs: Dict[str, Any]) -> str:
    """Dynamically load only the file that defines func_name, then call it."""
    index = _build_gym_index()
    info = index.get(func_name)
    if not info:
        return f"Function '{func_name}' not found. Use gym_search_tools to discover available tools."

    file_path = GYM_ROOT / info["file"]
    module_name = f"_gym__{func_name}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return f"Cannot load module from {info['file']}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        func = getattr(module, func_name, None)
        if func is None:
            return f"Function '{func_name}' not found in {info['file']}."

        result = func(**kwargs)
        return str(result) if result is not None else "(no output)"
    except Exception as e:
        import traceback
        return f"Error calling {func_name}: {e}\n{traceback.format_exc()}"


# ── Public LangChain tools ────────────────────────────────────────────────────

@tool
def gym_search_tools(keyword: str) -> str:
    """Search SciAgentGYM for physics/chemistry/materials/life_science/astronomy/statistics functions.
    Returns matching function names, parameters, and descriptions.
    Examples:
      gym_search_tools('sound pressure level')
      gym_search_tools('doppler shift')
      gym_search_tools('crystal structure')
      gym_search_tools('thermodynamics')
    """
    index = _build_gym_index()
    kw = keyword.lower()
    scored = []

    for name, info in index.items():
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

    if not scored:
        subjects = sorted({v["subject"] for v in index.values()})
        return (
            f"No functions found for '{keyword}'.\n"
            f"Available subjects: {', '.join(subjects)}\n"
            f"Total functions indexed: {len(index)}"
        )

    lines = [f"SciAgentGYM matches for '{keyword}' ({len(scored)} found):"]
    for _, name, info in scored[:15]:
        params = ", ".join(info["params"])
        tag = f"[{info['subject']}/{info['topic']}]"
        lines.append(f"  {name}({params})  {tag}")
        if info["docstring"]:
            lines.append(f"    {info['docstring'][:90]}")

    if len(scored) > 15:
        lines.append(f"  ... and {len(scored) - 15} more. Refine your keyword.")

    return "\n".join(lines)


@tool
def run_gym_tool(tool_name: str, tool_args: str) -> str:
    """Call a SciAgentGYM toolkit function by name.

    tool_name: exact function name (use gym_search_tools to find it)
    tool_args: JSON string of keyword arguments
               e.g. '{"freq_emit": 1000, "freq_received": 1050, "v_sound": 1540}'
               Use '{}' for no arguments.
    """
    try:
        kwargs = json.loads(tool_args.strip()) if tool_args.strip() else {}
    except json.JSONDecodeError as e:
        return f"Invalid JSON in tool_args: {e}\nExpected format: '{{\"key\": value}}'"

    return _load_and_call(tool_name, kwargs)


GYM_TOOLS = [gym_search_tools, run_gym_tool]
