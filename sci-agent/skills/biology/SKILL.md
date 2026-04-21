---
name: biology
description: "Handle biology tasks: protein property analysis, peptide calculations, DNA/RNA sequence analysis, codon optimization, protein solubility prediction, sequence alignment, ORF finding, and molecular biology calculations."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_scitool
---

# Biology Skill

Use this skill for biology and biochemistry tasks.

## Protein Analysis
- `ComputeProtPara(sequence)` — pI, MW, instability index, GRAVY
- `ComputePiMw(sequence)` — Isoelectric point and molecular weight
- `ComputeExtinctionCoefficient(sequence)` — Extinction coefficient
- `ComputeProtScale(sequence)` — Hydrophilicity/hydrophobicity profile
- `AminoAcidStatistics(sequence)` — Amino acid composition

## Peptide Tools
- `PeptideWeightCalculator(sequence)` — Peptide weight
- `PeptideFormulaCalculator(sequence)` — Peptide formula
- `ConvertingPeptide2SMILES(sequence)` — Peptide to SMILES
- `ProteaseDigestion(sequence)` — Protease digestion sites

## DNA/RNA Tools
- `TranslateDNAtoAminoAcidSequence(dna)` — DNA to protein
- `GetReverseComplement(dna)` — Reverse complement
- `ORFFind(dna)` — Open reading frames
- `RepeatDNASequenceSearch(dna)` — Repeat sequences
- `DNAMolecularWeightCalculator(dna)` — DNA molecular weight
- `CpGIslandPrediction(dna)` — CpG island prediction

## Sequence Alignment
- `DoubleSequenceGlobalAlignment(seq1, seq2)` — Global alignment
- `DoubleSequenceLocalAlignment(seq1, seq2)` — Local alignment
- `SequenceSimilarityCalculator(seq1, seq2)` — Similarity score

## Workflow
1. For protein queries: start with `ComputeProtPara` for basic properties
2. For DNA queries: use `TranslateDNAtoAminoAcidSequence` then protein tools
3. Use `run_scitool` with tool name and sequence as input string
