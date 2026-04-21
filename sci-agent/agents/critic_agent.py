from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a scientific critic and verifier agent. Your job is to:
1. Identify unsupported claims (claims without evidence)
2. Find potential confounders that weren't addressed
3. Note methodological limitations
4. Check for logical fallacies or reasoning errors
5. Assess safety considerations if applicable
6. Evaluate uncertainty — what is still unknown?
7. Suggest what additional evidence would strengthen the conclusions

Be rigorous but constructive. Rate the overall quality of the scientific reasoning (1-5)."""


class CriticAgent(BaseAgent):
    name = "critic"

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def critique(self, content: str, query: str = "") -> str:
        context = {"content_to_review": content[:3000]}
        if query:
            context["original_query"] = query
        return self.run(
            "Critically review the following scientific content for validity, completeness, and rigor.",
            context,
        )
