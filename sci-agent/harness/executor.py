import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.router_agent import RouterAgent
from agents.literature_agent import LiteratureAgent
from agents.compute_agent import ComputeAgent
from agents.experiment_agent import ExperimentAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.critic_agent import CriticAgent
from agents.synthesizer_agent import SynthesizerAgent
from harness.schema_validator import RouterOutput
from tools.registry import plan_tool_chain, search_tools_by_keyword, get_tool_info
from tools.scitool_tools import run_scitool
from tools.gym_tools import _load_and_call as _gym_call, _build_gym_index
from tools.unified_search import search_all_tools, plan_science_workflow

MAX_CRITIC_RETRIES = 2


class ScienceHarness:
    def __init__(self):
        self.router = RouterAgent()
        self.literature = LiteratureAgent()
        self.compute = ComputeAgent()
        self.experiment = ExperimentAgent()
        self.hypothesis = HypothesisAgent()
        self.critic = CriticAgent()
        self.synthesizer = SynthesizerAgent()

    def _select_tools(self, plan: RouterOutput) -> list[dict]:
        """Use KG to select tools relevant to the query."""
        tools = plan_tool_chain(plan.domain, plan.task_type)
        if not tools:
            tools = search_tools_by_keyword(plan.task_type)
        return tools

    def _run_tool_chain(self, tools: list[dict], query: str, file_paths: list[str]) -> list[dict]:
        """Call tools via SciToolAgent or SciAgentGYM depending on source."""
        gym_index = _build_gym_index()
        results = []
        for tool_info in tools:
            tool_name = tool_info["tool_id"]
            if tool_name in gym_index:
                result = _gym_call(tool_name, {})
            else:
                result = run_scitool.invoke({"tool_name": tool_name, "tool_input": query})
            results.append({
                "tool": tool_name,
                "description": tool_info.get("description", ""),
                "result": result,
            })
        return results

    def run(
        self,
        query: str,
        pdf_path: str = "",
        csv_path: str = "",
        verbose: bool = True,
    ) -> str:
        def log(msg):
            if verbose:
                print(f"[Harness] {msg}")

        file_paths = [p for p in [pdf_path, csv_path] if p]

        # 1. Route
        log("Routing query...")
        plan: RouterOutput = self.router.route(
            query,
            has_pdf=bool(pdf_path),
            has_csv=bool(csv_path),
        )
        log(f"Domain={plan.domain} | Task={plan.task_type} | Risk={plan.risk_level}")
        log(f"Agents={plan.required_agents}")

        # 2. KG-based tool selection
        log("Selecting tools from KG...")
        selected_tools = self._select_tools(plan)
        tool_names = [t["tool_id"] for t in selected_tools]
        log(f"Selected tools: {tool_names}")

        agent_outputs: dict[str, str] = {
            "query": query,
            "routing_plan": plan.model_dump_json(),
            "selected_tools": json.dumps([t["tool_id"] for t in selected_tools]),
        }

        for attempt in range(MAX_CRITIC_RETRIES + 1):
            # 3. Run specialist agents
            if "literature" in plan.required_agents:
                log("Running Literature Agent...")
                agent_outputs["literature"] = self.literature.analyze(
                    query, file_path=pdf_path
                )

            if "compute" in plan.required_agents:
                log("Running Compute Agent...")
                if csv_path:
                    agent_outputs["compute"] = self.compute.analyze_csv(query, csv_path)
                else:
                    agent_outputs["compute"] = self.compute.run(
                        f"Provide computational analysis for: {query}"
                    )

            if "experiment" in plan.required_agents:
                log("Running Experiment Agent...")
                agent_outputs["experiment"] = self.experiment.design(
                    query, agent_outputs.get("literature", "")
                )

            if "hypothesis" in plan.required_agents:
                log("Running Hypothesis Agent...")
                agent_outputs["hypothesis"] = self.hypothesis.generate(
                    query,
                    domain=plan.domain,
                    context_text=agent_outputs.get("literature", ""),
                )

            # 4. Execute tool chain via SciToolAgent
            if selected_tools:
                log(f"Executing {len(selected_tools)} tools via SciToolAgent service...")
                tool_results = self._run_tool_chain(selected_tools, query, file_paths)
                agent_outputs["tool_results"] = json.dumps(tool_results, ensure_ascii=False)

            # 5. Critic
            combined = "\n\n".join(
                f"[{k.upper()}]\n{v}"
                for k, v in agent_outputs.items()
                if k not in ("query", "routing_plan")
            )
            log(f"Running Critic Agent (attempt {attempt + 1}/{MAX_CRITIC_RETRIES + 1})...")
            critique = self.critic.critique(combined, query)
            agent_outputs["critique"] = critique

            if "CRITICAL" not in critique.upper() and "FATAL" not in critique.upper():
                break
            log("Critic flagged critical issues — replanning...")

        # 6. Synthesize
        log("Running Synthesizer Agent...")
        final_report = self.synthesizer.synthesize(query, agent_outputs)
        return final_report
