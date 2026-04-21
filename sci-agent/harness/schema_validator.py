from pydantic import BaseModel, field_validator
from typing import Literal


class RouterOutput(BaseModel):
    domain: Literal["biology", "chemistry", "physics", "materials", "medicine", "environmental", "general"]
    task_type: Literal["literature_review", "hypothesis_generation", "experiment_design", "data_analysis", "calculation", "critique"]
    needs_rag: bool
    needs_python: bool
    needs_safety_check: bool
    required_agents: list[str]
    required_tools: list[str]
    risk_level: Literal["low", "medium", "high"]
    output_format: str = "scientific_report"

    @field_validator("required_agents", "required_tools", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class AgentResult(BaseModel):
    agent_name: str
    status: Literal["success", "failed", "skipped"]
    output: str
    tool_calls: list[dict] = []
    citations: list[str] = []


class FinalReport(BaseModel):
    problem_type: str
    scientific_background: str
    evidence: str
    hypothesis_or_interpretation: str
    experiment_or_analysis_plan: str
    tool_results: str
    expected_results: str
    limitations: str
    confidence: str
    next_actions: str
