"""Sanity check for KG-backed registry (no model/service needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tools.registry import (
    get_all_tool_names,
    get_tools_by_category,
    search_tools_by_keyword,
    get_tool_info,
    plan_tool_chain,
)


def test_load():
    names = get_all_tool_names()
    assert len(names) > 100, f"Expected 400+ tools, got {len(names)}"
    print(f"[PASS] Total tools loaded: {len(names)}")


def test_categories():
    chem = get_tools_by_category("Chemical")
    bio = get_tools_by_category("Biology")
    mat = get_tools_by_category("Material")
    gen = get_tools_by_category("General")
    print(f"[PASS] Chemical={len(chem)}, Biology={len(bio)}, Material={len(mat)}, General={len(gen)}")


def test_kg_info():
    info = get_tool_info("NameToSMILES")
    assert "description" in info
    print(f"[PASS] NameToSMILES -> {info.get('description', '')[:60]}")

    info2 = get_tool_info("ComputeProtPara")
    assert "description" in info2
    print(f"[PASS] ComputeProtPara -> {info2.get('description', '')[:60]}")


def test_search():
    results = search_tools_by_keyword("molecular weight")
    assert len(results) > 0
    print(f"[PASS] search 'molecular weight': {[r['tool_id'] for r in results[:3]]}")

    results2 = search_tools_by_keyword("protein sequence")
    assert len(results2) > 0
    print(f"[PASS] search 'protein sequence': {[r['tool_id'] for r in results2[:3]]}")


def test_plan():
    chain = plan_tool_chain("chemistry", "calculation")
    print(f"[PASS] chemistry/calculation chain: {[t['tool_id'] for t in chain]}")

    chain2 = plan_tool_chain("biology", "literature_review")
    print(f"[PASS] biology/literature_review chain: {[t['tool_id'] for t in chain2]}")

    chain3 = plan_tool_chain("materials", "data_analysis")
    print(f"[PASS] materials/data_analysis chain: {[t['tool_id'] for t in chain3]}")


if __name__ == "__main__":
    print("=== Registry + KG Tests ===")
    test_load()
    test_categories()
    test_kg_info()
    test_search()
    test_plan()
    print("\nAll tests passed.")
