from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv, FakeNewsEnvConfig


def test_env_rollout_with_manual_actions():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=3))
    observations = env.reset([sample])
    assert "## Current Stage\nanalyzer_report" in observations[0]
    assert "## Current Role\nanalyzer" in observations[0]

    _, rewards, dones, infos = env.step(
        [
            "Visual: The frames show an exaggerated promotional montage.\n"
            "Claim: The post claims a miracle herb cures every virus in 24 hours.\n"
            "Need: Need a skill for judging unsupported extraordinary cure claims."
        ]
    )
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]
    assert infos[0]["role"] == "analyzer"

    _, rewards, dones, infos = env.step(
        [
            "Skill: Extraordinary cure claims need documentary support, not just dramatic promotional imagery."
        ]
    )
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]
    assert infos[0]["role"] == "worker"

    verdict = '<verdict>{"label":"fake","rationale":"unsupported absolute cure claim"}</verdict>'
    _, rewards, dones, infos = env.step([verdict])
    assert dones[0]
    assert rewards[0] > 0.0
    assert infos[0]["won"]


def test_intermediate_reasoning_action_is_recorded():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=3))
    env.reset([sample])
    env.step(
        [
            "Visual: Stylized imagery.\n"
            "Claim: A concrete cure claim is made.\n"
            "Need: Need a medical-claims verification skill."
        ]
    )
    _, rewards, dones, infos = env.step(
        ["Skill: Extraordinary medical claims need clear support from the provided inputs."]
    )
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["reasoning_action"] == "worker_skill"


def test_repeated_or_out_of_order_action_is_penalized():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=3))
    env.reset([sample])
    _, rewards, dones, infos = env.step(['<verdict>{"label":"fake","rationale":"too early"}</verdict>'])
    assert rewards[0] < 0.0
    assert not dones[0]
    assert not infos[0]["is_action_valid"]
