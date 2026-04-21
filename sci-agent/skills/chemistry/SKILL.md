---
name: chemistry
description: "Handle chemistry tasks: molecular property calculation, SMILES conversion, reaction prediction, drug-likeness analysis, molecular similarity, functional group identification, and cheminformatics using RDKit and PubChem tools."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_scitool
---

# Chemistry Skill

Use this skill for any chemistry-related task. Available SciToolAgent tools for chemistry:

## Molecule Conversion
- `NameToSMILES(compound_name)` — Convert compound name to SMILES
- `SMILESToInChI(smiles)` — SMILES to InChI
- `InChIKeyToSMILES(inchikey)` — InChIKey to SMILES

## Molecular Properties
- `SMILESToWeight(smiles)` — Molecular weight from SMILES
- `GetMolFormula(smiles)` — Molecular formula
- `GetExactMolceularWeight(smiles)` — Exact molecular weight
- `GetCrippenDescriptors(smiles)` — LogP and MR
- `CalculateTPSA(smiles)` — Topological polar surface area
- `GetHBANum(smiles)`, `GetHBDNum(smiles)` — H-bond acceptors/donors
- `GetRotatableBondsNum(smiles)` — Rotatable bonds

## Safety & Drug-likeness
- `SafetySummary(smiles)` — Safety summary
- `CheckExplosiveness(smiles)` — Explosiveness check
- `CheckPatent(smiles)` — Patent status

## Similarity & Fingerprints
- `MolSimilarity(smiles1, smiles2)` — Tanimoto similarity
- `FuncGroups(smiles)` — Functional groups
- `BuildMorganFpFromSmiles(smiles)` — Morgan fingerprint

## Reactions
- `RXNPredict(reactants)` — Forward reaction prediction
- `RXNRetrosynthetic(product)` — Retrosynthesis

## Drug-likeness (Lipinski's Rule of 5)
Call these shortcut tools directly (no `run_scitool` needed):
1. `name_to_smiles(compound_name)` → SMILES
2. `smiles_to_weight(smiles)` → MW (must be ≤500 Da)
3. `get_crippen_descriptors(smiles)` → LogP (must be ≤5)
4. `get_hbd_count(smiles)` → H-bond donors (must be ≤5)
5. `get_hba_count(smiles)` → H-bond acceptors (must be ≤10)
6. `calculate_tpsa(smiles)` → TPSA (oral: ≤140 Å²)
7. Summarize pass/fail for each rule

## Workflow
1. Start with `name_to_smiles` (shortcut) if given a compound name
2. Use property shortcuts directly (no run_scitool wrapping needed)
3. Check safety for novel compounds
4. For less common tools, use `run_scitool(tool_name, input_string)`
