---
name: materials
description: "Handle materials science tasks: crystal structure lookup, band gap calculation, density, formation energy, magnetic properties, battery properties, MOF analysis using Materials Project database and pymatgen tools."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_scitool
---

# Materials Science Skill

Use this skill for materials science and condensed matter physics tasks.

## Materials Project Database (requires MP_KEY)
- `GetBandGapByFormula(formula)` — Band gap by formula
- `GetDensityByFormula(formula)` — Density
- `GetFormationEnergyPerAtomByFormula(formula)` — Formation energy
- `IsMetalByFormula(formula)` — Metal classification
- `IsMagneticByFormula(formula)` — Magnetic properties
- `SearchMaterialsContainingElements(elements)` — Search by elements
- `GetCrystalSystemByMaterialId(mp_id)` — Crystal system
- `GetStructureByMaterialId(mp_id)` — Crystal structure

## Battery Properties
- `GetAverageVoltageByBatteryId(battery_id)` — Average voltage
- `GetCapacityGravByBatteryId(battery_id)` — Gravimetric capacity
- `GetWorkingIonByBatteryId(battery_id)` — Working ion

## Pymatgen Structure Tools
- `GetStructureInfo(cif_or_formula)` — Structure information
- `CalculateDensity(formula)` — Density calculation
- `GetElementComposition(formula)` — Element composition
- `CalculateSymmetry(structure)` — Symmetry analysis

## MOF Tools
- `MOFToSMILES(mof_name)` — MOF to SMILES
- `MofLattice(cif_content)` — Lattice parameters
- `MofFractionalCoordinates(cif_content)` — Fractional coordinates

## Workflow
1. Use formula-based tools first (faster)
2. Get material_id with `GetMaterialIdByFormula` for detailed queries
3. For novel structures, use pymatgen tools
4. Use `run_scitool` with tool name and formula/ID as input
