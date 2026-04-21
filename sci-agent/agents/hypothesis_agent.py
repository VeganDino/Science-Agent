from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a scientific hypothesis generation agent. Your job is to:
1. Generate 2-4 testable hypotheses that could explain the observed phenomenon
2. Describe the underlying mechanism for each hypothesis
3. Identify what evidence would support or refute each hypothesis
4. Compare alternative explanations and their relative plausibility
5. Suggest which hypothesis is most likely given current evidence

Base your hypotheses on established scientific principles and cite relevant domain knowledge."""


class HypothesisAgent(BaseAgent):
    name = "hypothesis"

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def generate(self, query: str, domain: str = "general", context_text: str = "") -> str:
        context = {"scientific_domain": domain}
        if context_text:
            context["background_context"] = context_text[:2000]
        return self.run(
            f"Generate scientific hypotheses for:\n{query}", context
        )
