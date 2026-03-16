from fake_news_skillrl.agent import HeuristicFakeNewsAgent, build_agent
from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv
from fake_news_skillrl.memory import SkillsOnlyMemory
from fake_news_skillrl.trainer import RolloutTrainer, SFTDataBuilder


def test_sft_builder_and_rollout_trainer_end_to_end():
    samples = load_normalized_samples("data/raw/smoke_samples.jsonl")
    sft_rows = SFTDataBuilder().build(samples)
    assert len(sft_rows) == len(samples)

    memory = SkillsOnlyMemory("memory_data/fake_news/claude_style_skills.json")
    trainer = RolloutTrainer(env=FakeNewsEnv(memory=memory), agent=HeuristicFakeNewsAgent())
    results = trainer.run(samples)

    assert "metrics" in results
    assert "label_accuracy" in results["metrics"]
    assert len(results["success"]) == len(samples)


def test_build_agent_heuristic_factory():
    agent = build_agent(agent_type="heuristic")
    assert isinstance(agent, HeuristicFakeNewsAgent)
