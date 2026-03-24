"""Standalone SkillRL-style fake news detection package."""

from .agent import (
    AnalyzerAgent,
    BaseFakeNewsAgent,
    DashScopeMultiModalAgent,
    HeuristicFakeNewsAgent,
    OpenAIResponsesAgent,
    QwenVLAgent,
    WorkerAgent,
    build_agent,
    build_agent_pair,
    select_inference_device,
)
from .env import FakeNewsEnv, FakeNewsEnvConfig
from .memory import SkillsOnlyMemory
from .metrics import compute_classification_metrics
from .web_search import DuckDuckGoSearcher, SearchResult

__all__ = [
    "BaseFakeNewsAgent",
    "FakeNewsEnv",
    "FakeNewsEnvConfig",
    "HeuristicFakeNewsAgent",
    "OpenAIResponsesAgent",
    "QwenVLAgent",
    "AnalyzerAgent",
    "DashScopeMultiModalAgent",
    "WorkerAgent",
    "SkillsOnlyMemory",
    "build_agent",
    "build_agent_pair",
    "compute_classification_metrics",
    "select_inference_device",
    "DuckDuckGoSearcher",
    "SearchResult",
]
