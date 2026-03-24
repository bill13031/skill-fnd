from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv, FakeNewsEnvConfig


def test_env_rollout_with_manual_actions():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=4))
    observations = env.reset([sample])
    assert "## Current Stage\nevent_extraction" in observations[0]
    assert "## Current Role\nanalyzer" in observations[0]

    _, rewards, dones, infos = env.step(
        [
            "Visual: The frames show an exaggerated promotional montage.\n"
            "Event: The post reports that a miracle herb cures every virus in 24 hours."
        ]
    )
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]
    assert infos[0]["role"] == "analyzer"

    _, rewards, dones, infos = env.step(
        [
            "Preliminary reasoning: The extracted miracle-cure event is doubtful because it makes an extreme medical claim without trustworthy support.\n"
            "Need: Need a skill for judging unsupported extraordinary cure claims."
        ]
    )
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]
    assert infos[0]["stage"] == "preliminary_analysis"

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
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=4))
    env.reset([sample])
    env.step(
        [
            "Visual: Stylized imagery.\n"
            "Event: The post reports a concrete cure claim."
        ]
    )
    _, _, _, infos = env.step(
        [
            "Preliminary reasoning: The extracted cure event is suspicious because the claim is sweeping and unsupported.\n"
            "Need: Need a medical-claims verification skill."
        ]
    )
    assert infos[0]["reasoning_action"] == "preliminary_analysis"
    _, rewards, dones, infos = env.step(
        ["Skill: Extraordinary medical claims need clear support from the provided inputs."]
    )
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["reasoning_action"] == "worker_skill"


def test_repeated_or_out_of_order_action_is_penalized():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=4))
    env.reset([sample])
    _, rewards, dones, infos = env.step(['<verdict>{"label":"fake","rationale":"too early"}</verdict>'])
    assert rewards[0] < 0.0
    assert not dones[0]
    assert not infos[0]["is_action_valid"]
