"""Standalone SkillRL-style fake news detection package."""

from .agent import BaseFakeNewsAgent, HeuristicFakeNewsAgent, QwenVLAgent, build_agent
from .env import FakeNewsEnv, FakeNewsEnvConfig
from .memory import SkillsOnlyMemory
from .metrics import compute_classification_metrics

__all__ = [
    "BaseFakeNewsAgent",
    "FakeNewsEnv",
    "FakeNewsEnvConfig",
    "HeuristicFakeNewsAgent",
    "QwenVLAgent",
    "SkillsOnlyMemory",
    "build_agent",
    "compute_classification_metrics",
]
