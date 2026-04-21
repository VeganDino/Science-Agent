"""Quick sanity check for tool functions (no model needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tools.science_tools import (
    t_test, regression, smiles_validator, rdkit_descriptor,
    pubchem_lookup, sequence_analyzer, python_exec,
)
from tools.registry import get_tools_for_domain, plan_tool_chain


def test_statistics():
    result = t_test([1.2, 2.3, 3.1, 2.8, 3.5], [2.1, 2.9, 3.8, 3.2, 4.0])
    assert "p_value" in result
    print(f"[PASS] t_test: p={result['p_value']:.4f}, significant={result['significant']}")

    result = regression([[1],[2],[3],[4],[5]], [2.1, 3.9, 6.2, 7.8, 10.1])
    assert "r_squared" in result
    print(f"[PASS] regression: R²={result['r_squared']:.4f}")


def test_chemistry():
    result = smiles_validator("CC(=O)Oc1ccccc1C(=O)O")
    assert result["valid"]
    print(f"[PASS] smiles_validator: {result['canonical_smiles']}")

    result = rdkit_descriptor("CC(=O)Oc1ccccc1C(=O)O")
    assert "molecular_weight" in result
    print(f"[PASS] rdkit_descriptor: MW={result['molecular_weight']:.2f}, LogP={result['logp']:.2f}")


def test_biology():
    result = sequence_analyzer("ATGCGATCGATCG", "DNA")
    assert "gc_content" in result
    print(f"[PASS] sequence_analyzer: length={result['length']}, GC={result['gc_content']}%")


def test_python_exec():
    result = python_exec("result = 2 + 2\nprint(result)")
    assert result["error"] == ""
    print(f"[PASS] python_exec: output='{result['result'].strip()}'")


def test_registry():
    chem_tools = get_tools_for_domain("chemistry")
    assert len(chem_tools) > 0
    print(f"[PASS] registry: {len(chem_tools)} chemistry tools found")

    chain = plan_tool_chain("chemistry", "calculation")
    print(f"[PASS] tool_chain: {chain}")


if __name__ == "__main__":
    print("=== Tool Sanity Tests ===")
    test_statistics()
    test_chemistry()
    test_biology()
    test_python_exec()
    test_registry()
    print("\nAll tests passed.")
