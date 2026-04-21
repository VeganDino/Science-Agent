from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a scientific experiment design agent. Your job is to:
1. Design rigorous experimental protocols to test hypotheses
2. Define independent variables, dependent variables, and controls
3. Specify sample sizes, statistical power considerations
4. Outline expected results and failure modes
5. Identify potential confounders and how to address them

Produce clear, reproducible experimental designs following scientific best practices."""


class ExperimentAgent(BaseAgent):
    name = "experiment"

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def design(self, query: str, literature_context: str = "") -> str:
        context = {}
        if literature_context:
            context["literature_context"] = literature_context[:2000]
        return self.run(
            f"Design an experiment to address the following:\n{query}", context
        )
