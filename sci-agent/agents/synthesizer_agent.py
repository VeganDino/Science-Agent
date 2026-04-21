from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a scientific synthesizer agent. Your job is to:
1. Integrate outputs from multiple specialist agents into a coherent scientific report
2. Structure the report with these sections:
   - Problem Type
   - Scientific Background
   - Evidence & Sources
   - Hypothesis / Interpretation
   - Experiment / Analysis Plan
   - Tool Results
   - Expected Results
   - Limitations & Confounders
   - Confidence Level
   - Recommended Next Actions
3. Ensure the report is internally consistent
4. Highlight key findings clearly
5. Maintain scientific precision while being readable

Produce a complete, well-structured scientific report."""


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def synthesize(self, query: str, agent_outputs: dict) -> str:
        context = {k: v[:1500] if isinstance(v, str) else str(v) for k, v in agent_outputs.items()}
        return self.run(
            f"Synthesize a comprehensive scientific report for the query:\n{query}",
            context,
        )
