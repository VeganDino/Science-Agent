# Nemotron Science Agent

> **Democratizing Rigorous Scientific Inquiry**
> A multi-agent LLM system with 1,900 verified scientific tools, built for the NVIDIA Nemotron Hackathon (April 2026).

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron-76B900)](https://www.nvidia.com/)
[![vLLM](https://img.shields.io/badge/Serving-vLLM-blue)](https://github.com/vllm-project/vllm)
[![LangGraph](https://img.shields.io/badge/Agents-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)

**Team:** Suhyeong Park · Sungmin Sung · Seunghwan Kim

---

## Table of Contents

- [Motivation](#motivation)
- [What Makes It Different](#what-makes-it-different)
- [System Architecture](#system-architecture)
- [Key Innovation: Knowledge Graph Planner](#key-innovation-knowledge-graph-planner)
- [The 7-Agent Digital Lab Team](#the-7-agent-digital-lab-team)
- [Tech Stack](#tech-stack)
- [Environment Setup](#environment-setup)
- [Training & Indexing Instructions](#training--indexing-instructions)
- [Evaluation](#evaluation)
- [Demo Run Guide](#demo-run-guide)
- [Example Walkthrough](#example-walkthrough)
- [Project Structure](#project-structure)
- [Who This Is For](#who-this-is-for)
- [Limitations & Roadmap](#limitations--roadmap)
- [License](#license)

---

## Motivation

Modern science needs more than one LLM.

- **General LLMs hallucinate scientific facts.** They cannot chain verified tools, producing plausible-sounding answers with no audit trail.
- **Domain tools are siloed and expert-only.** Schrödinger, Materials Project, and similar platforms are rigorous but require specialist training.
- **Real science crosses disciplines.** Drug discovery alone requires chemistry, biology, and statistics — no single tool or person covers it all.

The Nemotron Science Agent closes this gap.

---

## What Makes It Different

**Plausible vs. Verifiable.** Ask: *"What is the molecular weight of penicillin G?"*

| General LLM | Nemotron Science Agent |
|---|---|
| Outputs a number directly | `NameToSMILES("penicillin G")` |
| No tool calls, no verification | → `SMILESToWeight(...)` |
| Opaque — cannot audit the source | → **334.39 g/mol** with traceable log |

Every answer is backed by a verifiable chain of tool calls.

---

## System Architecture

```
┌─────────────────────┐
│  User Query         │  PDF · CSV · question
└──────────┬──────────┘
           ▼
┌─────────────────────────────────────┐
│  Main Orchestrator                  │
│  Nemotron-3-Nano-30B · vLLM         │  ← model-swappable (zero code change)
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  DeepAgents + LangGraph ReAct       │
│  SqliteSaver · thread_id isolation  │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────┐
│  7 Specialist Sub-Agents                                    │
│  literature · compute · physics · experiment ·              │
│  hypothesis · critic · debate                               │
└──────────┬──────────────────────────────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│  SciAgent Tool Layer                │
│  1,900 verified tools               │
│  typed I/O edges · AST-indexed      │
└─────────────────────────────────────┘
```

---

## Key Innovation: Knowledge Graph Planner

1,900 tools is too many to wire by hand. We let the graph do it.

1. **Every query starts with a plan.**
   `make_science_plan(goal)` — a hard rule enforced in `AGENTS.md`.
2. **BFSDT walks the typed-edge knowledge graph.**
   *Breadth-First Search Decision Tree* treats tool I/O types as graph edges and finds valid execution paths.
3. **Returns a concrete execution plan.**
   Tool names + step-count target, ready for sub-agents to execute.

**Example — goal: molecule name → molecular weight**

```
"penicillin G" → NameToSMILES → SMILESToWeight → 334.39 g/mol
```

---

## The 7-Agent Digital Lab Team

| Agent | Role Metaphor | Primary Use |
|---|---|---|
| `literature-agent` | Literature reviewer | arXiv + PDF QA, evidence-grade rating |
| `compute-agent` | Computational chemist | Molecular / protein / material properties |
| `physics-agent` | Physicist | 11 subdomains — acoustics to quantum |
| `experiment-agent` | Experimentalist | Protocols, controls, sample size for 80% power |
| `hypothesis-agent` | Hypothesis generator | 2–4 testable hypotheses with priors |
| `critic-agent` | Peer reviewer | Flags flaws, statistical issues, safety |
| `debate-agent` | Debate moderator | Spawns opposing scientists, synthesizes |

---

## Tech Stack

| Layer | Tech | Purpose |
|---|---|---|
| **LLM Core** | Nemotron (via vLLM) | High-quality reasoning, multi-role prompting |
| **Agent Orchestration** | LangGraph + DeepAgents | Stateful multi-agent control |
| **Tool Execution** | Science Tool (1,900 tools) | Verifiable scientific computation |
| **Document Processing** | pdfplumber + Nemotron OCR V2 | Text PDFs and scanned papers |
| **Serving Layer** | FastAPI + vLLM server | Low-latency SSE streaming |
| **Frontend** | Next.js | Demo + production UI |
| **Memory** | SQLite (SqliteSaver) | Per-thread conversational state |

---

## Environment Setup

### System requirements

- **OS:** Ubuntu 22.04 LTS (or any Linux with CUDA support)
- **GPU:** NVIDIA GPU with ≥ 24 GB VRAM (A100 / H100 / RTX 4090 recommended)
- **CUDA:** 12.1+
- **Python:** 3.10 or later
- **Node.js:** 18+ (for frontend)
- **Disk:** ≥ 100 GB free (for model weights + tool index)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/nemotron-science-agent.git
cd nemotron-science-agent
```

### 2. Create a Python environment

```bash
# Using conda (recommended)
conda create -n sci-agent python=3.10 -y
conda activate sci-agent

# Or using venv
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install vLLM separately (match your CUDA version)
pip install vllm==0.6.3
```

### 4. Install Node.js dependencies (frontend)

```bash
cd frontend
npm install
cd ..
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Required variables in `.env`:

```bash
# Model serving
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
MODEL_NAME=nvidia/Nemotron-3-Nano-30B

# Backend API
API_HOST=0.0.0.0
API_PORT=8080

# Memory database
MEMORY_DB_PATH=./data/memory.db

# Tool index
TOOL_INDEX_PATH=./data/tool_index.json

# Optional: API keys for external tools
ARXIV_API_KEY=
PUBMED_API_KEY=
```

---

## Training & Indexing Instructions

> **Note:** This project does *not* fine-tune the base LLM. "Training" here refers to **building the tool knowledge graph** and **indexing scientific tools** that the agents rely on. Nemotron is used zero-shot via vLLM.

### Step 1. Index the 1,900 scientific tools (AST parsing)

```bash
python scripts/build_tool_index.py \
    --tools-dir ./sci_tools \
    --output ./data/tool_index.json
```

This parses every tool's Python source via AST, extracts typed I/O signatures, and produces a JSON index the planner consumes.

### Step 2. Build the typed-edge knowledge graph

```bash
python scripts/build_knowledge_graph.py \
    --index ./data/tool_index.json \
    --output ./data/kg.pkl
```

Creates the tool I/O type-edge graph used by the BFSDT planner.

### Step 3. Validate the graph (sanity check)

```bash
python scripts/validate_kg.py --graph ./data/kg.pkl
```

Confirms:
- All 1,900 tools indexed
- No broken type edges
- Planner can reach all terminal output types

### Step 4. (Optional) Precompute OCR cache for known PDFs

```bash
python scripts/ocr_pretrain.py --pdf-dir ./corpus/ --cache ./data/ocr_cache/
```

---

## Evaluation

### Run the full evaluation suite

```bash
python eval/run_eval.py \
    --config eval/configs/default.yaml \
    --output eval/results/
```

### Available benchmarks

| Benchmark | Domain | Metric |
|---|---|---|
| `molecular_weight_100` | Computational chemistry | Exact match (±0.01 g/mol) |
| `drug_likeness_lipinski` | Drug discovery | Classification accuracy |
| `physics_qa` | Physics reasoning | Tool-chain F1 |
| `literature_retrieval` | arXiv QA | Recall@5, evidence grade |
| `hypothesis_generation` | Scientific reasoning | Human-rated quality (1–5) |
| `critic_flag_detection` | Peer review | Precision / recall of flagged flaws |

### Evaluate a single task

```bash
python eval/run_task.py \
    --task molecular_weight_100 \
    --model Nemotron-3-Nano-30B \
    --verbose
```

### Compare against a baseline

```bash
python eval/compare.py \
    --baseline gpt-4o-mini \
    --candidate Nemotron-3-Nano-30B \
    --tasks all
```

Results are written to `eval/results/<timestamp>/` as JSON + an auto-generated Markdown summary.

---

## Demo Run Guide

Three services must be running: **(A)** the vLLM model server, **(B)** the FastAPI agent backend, **(C)** the Next.js frontend.

### A. Start the vLLM Nemotron server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Nemotron-3-Nano-30B \
    --served-model-name model \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

Verify with:

```bash
curl http://localhost:8000/v1/models
```

### B. Start the FastAPI agent backend

In a new terminal:

```bash
conda activate sci-agent
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

Health check:

```bash
curl http://localhost:8080/health
# {"status":"ok","tools_indexed":1900,"kg_loaded":true}
```

### C. Start the Next.js frontend

In another terminal:

```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

### Try a sample query

In the UI — or via curl — send:

```bash
curl -X POST http://localhost:8080/api/ask \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Evaluate ibuprofen drug-likeness and summarize safety concerns.",
      "thread_id": "demo-session-1"
    }'
```

Expected output: an 8-section structured report (Key Finding, Methods, Tool Results, Interpretation, Limitations, Confidence, Next Steps, Evidence Log) streamed via SSE.

### Docker (optional, one-shot)

```bash
docker-compose up --build
```

Brings up all three services with sensible defaults. Ensure the NVIDIA Container Toolkit is installed.

---

## Example Walkthrough

**Query:** *"Evaluate ibuprofen's drug-likeness and summarize safety concerns."*

### Execution trace

```
1. make_science_plan(goal)                              → Plans the tool chain
2. NameToSMILES("ibuprofen")                            → SMILES string
3. GetCrippenDescriptors + CalculateTPSA + Lipinski     → Compute descriptors
4. SafetySummary                                        → Extract safety concerns
5. critic-agent review                                  → Flag issues & confidence
6. Final report                                         → 8 structured sections
```

### Report structure (every answer)

| Section | Purpose |
|---|---|
| **Key Finding** | The headline answer |
| **Methods** | What was done |
| **Tool Results** | Raw outputs |
| **Interpretation** | What the results mean |
| **Limitations** | What this analysis cannot say |
| **Confidence** | Quantified uncertainty |
| **Next Steps** | Recommended follow-up |
| **Evidence Log** | Full, auditable tool-call trace |

Every answer follows this fixed structure — readable by a researcher, checkable by a reviewer.

---

## Project Structure

```
nemotron-science-agent/
├── backend/              # FastAPI application
│   ├── main.py           # API entrypoint
│   ├── agents/           # 7 specialist sub-agents
│   ├── orchestrator/     # LangGraph + DeepAgents
│   └── planner/          # BFSDT knowledge-graph planner
├── sci_tools/            # 1,900 scientific tools
├── scripts/              # Indexing & graph-building scripts
├── eval/                 # Benchmarks & evaluation harness
├── frontend/             # Next.js UI
├── data/                 # Generated indexes, memory.db
├── corpus/               # Sample PDFs for OCR testing
├── AGENTS.md             # Agent rules (hard constraints)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Who This Is For

Not pharma only — rigorous science is for everyone who asks questions.

- **Graduate students** entering a new field
- **Interdisciplinary researchers** crossing domain boundaries
- **Science educators** demonstrating verified reasoning
- **Journalists & policymakers** fact-checking scientific claims
- **Small R&D teams** without full specialist staff
- **Citizen scientists** asking rigorous questions

> We don't replace scientists. We give every curious person a specialist team.

---

## Limitations & Roadmap

### Current limitations

- No user authentication or multi-tenancy — only `thread_id` isolation
- Deployment scripts not yet formalized
- OCR microservice (Nemotron OCR V2, PDF rasterization via PyMuPDF) still being hardened

### Next up

- **PubMed + ClinicalTrials.gov** integration for biomedical depth
- **Multimodal input** — structure images, spectra, diagrams
- **Formal CI/CD** and production deployment scripts

> Research prototype works. The gap to production is standard SaaS engineering — not fundamental AI research.

---

## License

To be determined. Built for the NVIDIA Nemotron Hackathon, April 2026.

---

## Acknowledgements

- **NVIDIA** for the Nemotron model family and the hackathon
- **vLLM, LangGraph, DeepAgents** open-source communities
- The scientific tool ecosystem — RDKit, pdfplumber, PyMuPDF, and many more
