---
name: literature
description: "Handle literature review tasks: searching arXiv papers, downloading PDFs, extracting key claims, identifying evidence quality, checking citations, and summarizing scientific findings."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_scitool, read_file
---

# Literature Review Skill

Use this skill when the user asks about research papers, scientific evidence, or wants literature searched.

## Available Tools
- `DownloadPapers(keyword)` — Download papers from arXiv by keyword
- `DownloadPapersMuti(keywords_str)` — Multi-keyword paper download
- `PaperQA(question)` — Q&A over downloaded PDF papers

## Reading PDF files
Use `read_file` for PDFs already in the workspace.

## Extraction Framework
When reading a paper, always extract:
1. **Main claim** — What does the paper claim?
2. **Evidence** — What data/experiments support it?
3. **Methods** — How was it tested? Sample size, controls?
4. **Limitations** — What do the authors admit as limits?
5. **Conclusions** — Are conclusions supported by the evidence?

## Citation Quality Assessment
- Level 1: Peer-reviewed RCT or systematic review
- Level 2: Observational cohort study
- Level 3: Case study or expert opinion
- Level 4: Preprint (not yet peer-reviewed)

## Workflow
1. If user provides PDF path, use `read_file` to read it
2. If searching topic, use `DownloadPapers` then `PaperQA`
3. Extract claims using the framework above
4. Flag unsupported claims explicitly
