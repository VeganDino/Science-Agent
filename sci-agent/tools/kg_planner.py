"""KG-based tool chain planner using SciToolAgent's graph_store.json."""
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Optional

KG_PATH = Path("/home/ubuntu/workspace/SciToolAgent/KG/storage_graph_large/graph_store.json")

# ── Graph loading ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_graph() -> dict:
    with open(KG_PATH) as f:
        return json.load(f)["graph_dict"]


@lru_cache(maxsize=1)
def _build_indices():
    graph = _load_graph()

    tool_info: dict[str, dict] = {}          # tool_name -> {desc, inputs, outputs, category, security}
    type_to_tools: dict[str, list] = defaultdict(list)   # input_type -> [tool_names]
    tool_to_outputs: dict[str, list] = defaultdict(list) # tool_name -> [output_types]

    for node, triplets in graph.items():
        predicates = {p: [] for p in set(p for p, _ in triplets)}
        for p, o in triplets:
            predicates[p] = predicates.get(p, []) + [o]

        if "is a" in predicates:
            # This is a tool node
            tool_info[node] = {
                "description": (predicates.get("has the functionality that") or [""])[0],
                "inputs":      predicates.get("inputs", []),
                "outputs":     predicates.get("outputs", []),
                "category":    (predicates.get("is a") or [""])[0],
                "source":      (predicates.get("is sourced from") or [""])[0],
                "security":    bool(predicates.get("needs")),
            }
            for inp in predicates.get("inputs", []):
                type_to_tools[inp.lower()].append(node)
            for out in predicates.get("outputs", []):
                tool_to_outputs[node].append(out.lower())

        elif "is the input of" in predicates:
            # data type node — already captured above via tool "inputs"
            pass

    return tool_info, type_to_tools, tool_to_outputs


# ── Public API ────────────────────────────────────────────────────────────────

def get_tool_info(tool_name: str) -> dict:
    tool_info, _, _ = _build_indices()
    return tool_info.get(tool_name, {})


def tools_for_input_type(input_type: str) -> list[str]:
    """Return all tools that accept input_type."""
    _, type_to_tools, _ = _build_indices()
    return type_to_tools.get(input_type.lower(), [])


def outputs_of_tool(tool_name: str) -> list[str]:
    """Return output types produced by tool_name."""
    _, _, tool_to_outputs = _build_indices()
    return tool_to_outputs.get(tool_name, [])


@dataclass
class DecisionNode:
    """A node in the DFDTS decision tree."""
    tool: str
    depth: int
    input_types: list
    output_types: list
    is_solution: bool = False
    children: list = field(default_factory=list)  # alternative next-step tools


def dfdts_tool_chain(
    start_input_type: str,
    target_output_type: str,
    max_depth: int = 5,
    max_branches: int = 3,
    max_solutions: int = 8,
    exclude_security: bool = True,
) -> tuple[list[list[str]], Optional[DecisionNode]]:
    """Depth-First Decision Tree Search through the KG.

    Explores tool chains depth-first: follows one path as deep as possible
    before backtracking to try sibling branches.

    Returns:
        (solutions, decision_tree_root)
        - solutions: list of complete tool chain paths reaching target
        - decision_tree_root: DecisionNode tree showing all explored branches
    """
    tool_info, type_to_tools, tool_to_outputs = _build_indices()
    start = start_input_type.lower()
    target = target_output_type.lower()

    solutions: list[list[str]] = []

    # Sentinel root node (represents the starting state, no tool yet)
    root = DecisionNode(
        tool="[start]",
        depth=0,
        input_types=[start_input_type],
        output_types=[start_input_type],
    )

    def dfs(current_types: frozenset, depth: int, path: list, visited: set, parent: DecisionNode):
        if depth >= max_depth or len(solutions) >= max_solutions:
            return

        branches_tried = 0
        for t in sorted(current_types):
            for tool in type_to_tools.get(t, []):
                if tool in visited:
                    continue
                if exclude_security and tool_info.get(tool, {}).get("security"):
                    continue
                if branches_tried >= max_branches and depth > 0:
                    break

                new_outputs = frozenset(tool_to_outputs.get(tool, []))
                info = tool_info.get(tool, {})
                node = DecisionNode(
                    tool=tool,
                    depth=depth + 1,
                    input_types=[t],
                    output_types=list(new_outputs),
                    is_solution=target in new_outputs,
                )
                parent.children.append(node)
                branches_tried += 1

                new_path = path + [tool]
                if target in new_outputs:
                    solutions.append(new_path)
                else:
                    # Depth-first: recurse before trying sibling tools
                    dfs(new_outputs, depth + 1, new_path, visited | {tool}, node)

                if len(solutions) >= max_solutions:
                    return

    dfs(frozenset([start]), 0, [], set(), root)
    return solutions, root


def describe_decision_tree(root: DecisionNode, indent: int = 0) -> str:
    """Render a DecisionNode tree as a readable string."""
    lines = []
    prefix = "  " * indent
    marker = "✓ " if root.is_solution else "→ "
    if root.tool != "[start]":
        out = ", ".join(root.output_types[:3])
        lines.append(f"{prefix}{marker}{root.tool} → [{out}]")
    for child in root.children:
        lines.extend(describe_decision_tree(child, indent + 1).splitlines())
    return "\n".join(lines)


def find_tool_chain(
    start_input_type: str,
    target_output_type: str,
    max_depth: int = 4,
    exclude_security: bool = True,
) -> list[list[str]]:
    """BFS through KG to find all tool chains from start_input_type to target_output_type."""
    tool_info, type_to_tools, tool_to_outputs = _build_indices()
    start = start_input_type.lower()
    target = target_output_type.lower()

    # BFS: state = (current_output_types_set, path_of_tools)
    queue = deque([(frozenset([start]), [])])
    visited = set()
    chains = []

    while queue and len(chains) < 10:
        current_types, path = queue.popleft()
        state = (current_types, tuple(path))
        if state in visited or len(path) >= max_depth:
            continue
        visited.add(state)

        for t in current_types:
            for tool in type_to_tools.get(t, []):
                if tool in path:
                    continue
                if exclude_security and tool_info.get(tool, {}).get("security"):
                    continue
                new_path = path + [tool]
                new_outputs = frozenset(tool_to_outputs.get(tool, []))
                if target in new_outputs:
                    chains.append(new_path)
                else:
                    queue.append((new_outputs, new_path))

    return chains


def search_tools_by_description(keyword: str, top_k: int = 8) -> list[dict]:
    """Keyword search over tool descriptions in the KG."""
    tool_info, _, _ = _build_indices()
    kw = keyword.lower()
    scored = []
    for name, info in tool_info.items():
        score = 0
        if kw in name.lower():
            score += 3
        if kw in info.get("description", "").lower():
            score += 2
        inp_out = " ".join(info.get("inputs", []) + info.get("outputs", []))
        if kw in inp_out.lower():
            score += 1
        if score:
            scored.append((score, name, info))
    scored.sort(reverse=True)
    return [{"tool": name, **info} for _, name, info in scored[:top_k]]


def suggest_next_tools(current_output_type: str, exclude_security: bool = True) -> list[dict]:
    """Given a tool's output type, suggest which tools to call next."""
    tool_info, type_to_tools, _ = _build_indices()
    candidates = type_to_tools.get(current_output_type.lower(), [])
    results = []
    for t in candidates:
        info = tool_info.get(t, {})
        if exclude_security and info.get("security"):
            continue
        results.append({"tool": t, **info})
    return results


def get_tools_by_category(category_keyword: str) -> list[dict]:
    """Return all tools in a category (Chemical, Biological, Material, General)."""
    tool_info, _, _ = _build_indices()
    kw = category_keyword.lower()
    return [
        {"tool": name, **info}
        for name, info in tool_info.items()
        if kw in info.get("category", "").lower()
    ]


def describe_tool_chain(chain: list[str]) -> str:
    """Format a tool chain as a readable string."""
    if not chain:
        return "(no chain found)"
    tool_info, _, tool_to_outputs = _build_indices()
    parts = []
    for tool in chain:
        info = tool_info.get(tool, {})
        inp = ", ".join(info.get("inputs", []))
        out = ", ".join(info.get("outputs", []))
        desc = info.get("description", "")
        parts.append(f"  {tool}({inp}) → [{out}]  # {desc}")
    return "\n".join(parts)
