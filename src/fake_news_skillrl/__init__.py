"""Standalone SkillRL-style fake news detection package."""

from .agent import (
    AnalyzerAgent,
    BaseFakeNewsAgent,
    HeuristicFakeNewsAgent,
    QwenVLAgent,
    WorkerAgent,
    build_agent,
    build_agent_pair,
    select_inference_device,
)
from .env import FakeNewsEnv, FakeNewsEnvConfig
from .memory import SkillsOnlyMemory
from .metrics import compute_classification_metrics

__all__ = [
    "BaseFakeNewsAgent",
    "FakeNewsEnv",
    "FakeNewsEnvConfig",
    "HeuristicFakeNewsAgent",
    "QwenVLAgent",
    "AnalyzerAgent",
    "WorkerAgent",
    "SkillsOnlyMemory",
    "build_agent",
    "build_agent_pair",
    "compute_classification_metrics",
    "select_inference_device",
]
