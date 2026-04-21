import json
import re
from pathlib import Path
from harness.model import generate


def _load_prompt(name: str) -> str:
    path = Path(__file__).parent.parent / "prompts" / f"{name}.md"
    if path.exists():
        return path.read_text()
    return ""


def _extract_json(text: str) -> dict | None:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class BaseAgent:
    name: str = "base"

    def _system_prompt(self) -> str:
        return _load_prompt(self.name) or f"You are a {self.name} agent."

    def run(self, user_input: str, context: dict | None = None) -> str:
        prompt = self._build_prompt(user_input, context or {})
        return generate(prompt, self._system_prompt())

    def run_json(self, user_input: str, context: dict | None = None) -> dict:
        prompt = self._build_prompt(user_input, context or {})
        prompt += "\n\nRespond ONLY with a valid JSON block wrapped in ```json ... ```."
        raw = generate(prompt, self._system_prompt())
        parsed = _extract_json(raw)
        if parsed is None:
            return {"raw": raw, "parse_error": True}
        return parsed

    def _build_prompt(self, user_input: str, context: dict) -> str:
        if context:
            ctx_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            return f"Context:\n{ctx_str}\n\nTask:\n{user_input}"
        return user_input
