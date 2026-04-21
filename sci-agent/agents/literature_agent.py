from agents.base_agent import BaseAgent
from tools.science_tools import pdf_reader


SYSTEM_PROMPT = """You are a scientific literature agent. Your job is to:
1. Extract key claims and evidence from provided text
2. Identify the main conclusions and their supporting evidence
3. Note any methodological details (sample size, controls, statistical methods)
4. Flag any unsupported claims or limitations mentioned

Be precise and cite specific parts of the text. Format your response clearly."""


class LiteratureAgent(BaseAgent):
    name = "literature"

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def analyze(self, query: str, text: str = "", file_path: str = "") -> str:
        if file_path:
            result = pdf_reader(file_path)
            text = result.get("text", "")
            if not text:
                return f"Failed to read PDF: {result.get('error', 'unknown error')}"

        if text:
            context = {"document_text": text[:4000]}
        else:
            context = {}

        return self.run(
            f"Analyze the following scientific query and provide evidence-based insights:\n{query}",
            context,
        )
