import io
import sys
import traceback
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def python_exec(code: str) -> dict:
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    result = {}
    try:
        local_ns = {}
        exec(code, {"__builtins__": __builtins__, "pd": pd, "np": np, "plt": plt, "stats": stats}, local_ns)
        result["result"] = stdout_capture.getvalue() or str(local_ns.get("result", ""))
        result["error"] = ""
    except Exception:
        result["result"] = ""
        result["error"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    return result


def t_test(group_a: list, group_b: list | None = None, test_type: str = "two_sample") -> dict:
    a = np.array(group_a)
    if test_type == "one_sample" or group_b is None:
        t_stat, p_value = stats.ttest_1samp(a, 0)
    else:
        b = np.array(group_b)
        t_stat, p_value = stats.ttest_ind(a, b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def regression(X: list, y: list, mode: str = "linear") -> dict:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import r2_score
    X_arr = np.array(X)
    y_arr = np.array(y)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if mode == "logistic":
        model = LogisticRegression()
        model.fit(X_arr, y_arr)
        coefs = model.coef_.tolist()
        r2 = float(model.score(X_arr, y_arr))
    else:
        model = LinearRegression()
        model.fit(X_arr, y_arr)
        coefs = model.coef_.tolist()
        r2 = float(r2_score(y_arr, model.predict(X_arr)))
    return {
        "coefficients": coefs,
        "r_squared": r2,
        "summary": f"R²={r2:.4f}, coefs={coefs}",
    }


def plot_generator(plot_type: str, data: dict, title: str = "") -> dict:
    fig, ax = plt.subplots()
    if plot_type == "bar":
        ax.bar(data.get("x", []), data.get("y", []))
    elif plot_type == "scatter":
        ax.scatter(data.get("x", []), data.get("y", []))
    elif plot_type == "line":
        ax.plot(data.get("x", []), data.get("y", []))
    elif plot_type == "hist":
        ax.hist(data.get("values", []), bins=data.get("bins", 10))
    if title:
        ax.set_title(title)
    out_dir = Path(__file__).parent.parent / "data"
    out_path = out_dir / "plot_output.png"
    fig.savefig(out_path)
    plt.close(fig)
    return {"image_path": str(out_path)}


def pdf_reader(file_path: str) -> dict:
    import pdfplumber
    text = ""
    pages = 0
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = len(pdf.pages)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        return {"text": "", "pages": 0, "error": str(e)}
    return {"text": text[:8000], "pages": pages}


def csv_loader(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
        return {
            "columns": df.columns.tolist(),
            "shape": list(df.shape),
            "summary": df.describe().to_string(),
            "dataframe_json": df.head(20).to_json(),
        }
    except Exception as e:
        return {"columns": [], "shape": [], "summary": "", "dataframe_json": "", "error": str(e)}


def smiles_validator(smiles: str) -> dict:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "canonical_smiles": ""}
        return {"valid": True, "canonical_smiles": Chem.MolToSmiles(mol)}
    except Exception as e:
        return {"valid": False, "canonical_smiles": "", "error": str(e)}


def rdkit_descriptor(smiles: str) -> dict:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "h_bond_donors": Descriptors.NumHDonors(mol),
            "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
        }
    except Exception as e:
        return {"error": str(e)}


def pubchem_lookup(compound_name: str) -> dict:
    try:
        import pubchempy as pcp
        compounds = pcp.get_compounds(compound_name, "name")
        if not compounds:
            return {"error": f"No compound found for: {compound_name}"}
        c = compounds[0]
        return {
            "cid": c.cid,
            "smiles": c.isomeric_smiles,
            "iupac_name": c.iupac_name,
            "molecular_formula": c.molecular_formula,
        }
    except Exception as e:
        return {"error": str(e)}


def sequence_analyzer(sequence: str, seq_type: str = "DNA") -> dict:
    try:
        from Bio.Seq import Seq
        from Bio.SeqUtils import gc_fraction
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        seq = Seq(sequence.upper())
        result = {"length": len(seq)}
        if seq_type.upper() in ("DNA", "RNA"):
            result["gc_content"] = round(gc_fraction(seq) * 100, 2)
            result["complement"] = str(seq.complement())
        elif seq_type.upper() == "PROTEIN":
            analysis = ProteinAnalysis(str(seq))
            result["molecular_weight"] = round(analysis.molecular_weight(), 2)
            result["isoelectric_point"] = round(analysis.isoelectric_point(), 2)
        return result
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY = {
    "python_exec": python_exec,
    "t_test": t_test,
    "regression": regression,
    "plot_generator": plot_generator,
    "pdf_reader": pdf_reader,
    "csv_loader": csv_loader,
    "smiles_validator": smiles_validator,
    "rdkit_descriptor": rdkit_descriptor,
    "pubchem_lookup": pubchem_lookup,
    "sequence_analyzer": sequence_analyzer,
}


def run_tool(tool_id: str, **kwargs) -> dict:
    fn = TOOL_REGISTRY.get(tool_id)
    if fn is None:
        return {"error": f"Tool '{tool_id}' not found"}
    try:
        return fn(**kwargs)
    except Exception as e:
        return {"error": str(e)}
