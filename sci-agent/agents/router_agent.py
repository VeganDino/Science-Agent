from agents.base_agent import BaseAgent
from harness.schema_validator import RouterOutput


SYSTEM_PROMPT = """You are a scientific query router. Analyze the user's question and classify it.

Output a JSON object with these fields:
- domain: one of [biology, chemistry, physics, materials, medicine, environmental, general]
- task_type: one of [literature_review, hypothesis_generation, experiment_design, data_analysis, calculation, critique]
- needs_rag: bool (does this need paper/document search?)
- needs_python: bool (does this need computation/code execution?)
- needs_safety_check: bool (does this involve potentially hazardous materials/procedures?)
- required_agents: list of agents needed from [literature, compute, experiment, hypothesis, critic, synthesizer]
- required_tools: list of tools needed from [pdf_reader, citation_checker, python_exec, t_test, regression, plot_generator, csv_loader, rdkit_descriptor, pubchem_lookup, smiles_validator, sequence_analyzer]
- risk_level: one of [low, medium, high]
- output_format: "scientific_report"
"""


class RouterAgent(BaseAgent):
    name = "router"

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def route(self, user_query: str, has_pdf: bool = False, has_csv: bool = False) -> RouterOutput:
        context = {}
        if has_pdf:
            context["attached_file"] = "PDF document provided"
        if has_csv:
            context["attached_file"] = "CSV data file provided"

        result = self.run_json(user_query, context)

        if result.get("parse_error"):
            return RouterOutput(
                domain="general",
                task_type="literature_review",
                needs_rag=True,
                needs_python=False,
                needs_safety_check=False,
                required_agents=["literature", "critic", "synthesizer"],
                required_tools=["citation_checker"],
                risk_level="low",
            )

        try:
            return RouterOutput(**result)
        except Exception:
            return RouterOutput(
                domain="general",
                task_type="literature_review",
                needs_rag=True,
                needs_python=False,
                needs_safety_check=False,
                required_agents=["literature", "critic", "synthesizer"],
                required_tools=["citation_checker"],
                risk_level="low",
            )
