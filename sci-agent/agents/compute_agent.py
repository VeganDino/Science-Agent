from agents.base_agent import BaseAgent
from tools.science_tools import run_tool
import json


SYSTEM_PROMPT = """You are a scientific computation agent. Your job is to:
1. Identify what calculations or statistical analyses are needed
2. Select appropriate tools (t_test, regression, python_exec, etc.)
3. Execute computations and interpret results
4. Explain what the numbers mean in scientific context

Always show your calculations and explain the statistical significance of results."""


class ComputeAgent(BaseAgent):
    name = "compute"

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def analyze_csv(self, query: str, file_path: str) -> str:
        csv_result = run_tool("csv_loader", file_path=file_path)
        if "error" in csv_result:
            return f"Failed to load CSV: {csv_result['error']}"

        context = {
            "csv_summary": csv_result["summary"],
            "columns": str(csv_result["columns"]),
            "shape": str(csv_result["shape"]),
        }
        analysis_plan = self.run_json(
            f"Plan the statistical analysis for: {query}", context
        )

        results = []
        if "tool_calls" in analysis_plan:
            for call in analysis_plan["tool_calls"]:
                tool_id = call.get("tool")
                args = call.get("args", {})
                if tool_id:
                    tool_result = run_tool(tool_id, **args)
                    results.append({"tool": tool_id, "result": tool_result})

        context["analysis_results"] = json.dumps(results)
        return self.run(f"Interpret these results for: {query}", context)

    def compute(self, query: str, tool_id: str, tool_args: dict) -> str:
        tool_result = run_tool(tool_id, **tool_args)
        context = {"tool": tool_id, "tool_result": json.dumps(tool_result)}
        return self.run(f"Interpret this computation result for: {query}", context)
