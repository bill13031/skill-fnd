from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv, FakeNewsEnvConfig


def test_env_rollout_with_manual_actions():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=5))
    observations = env.reset([sample])
    assert "## Current Stage\nvisual_understanding" in observations[0]

    _, rewards, dones, infos = env.step(["Describe what is visible in the frames."])
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]

    _, rewards, dones, infos = env.step(["State the main claim."])
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]

    _, rewards, dones, infos = env.step(["The visuals do not verify the cure claim."])
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]

    _, rewards, dones, infos = env.step(["Apply the principle that extraordinary medical claims need clear visual support."])
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]

    verdict = '<verdict>{"label":"fake","rationale":"unsupported absolute cure claim"}</verdict>'
    _, rewards, dones, infos = env.step([verdict])
    assert dones[0]
    assert rewards[0] > 0.0
    assert infos[0]["won"]

def test_intermediate_reasoning_action_is_recorded():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=5))
    env.reset([sample])
    env.step(["Describe what is visible in the frames."])
    env.step(["State the main claim."])
    env.step(["The visuals do not verify the claim."])
    _, rewards, dones, infos = env.step(["Apply source skepticism and cross-modal checking."])
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["reasoning_action"] == "skill_application"


def test_repeated_or_out_of_order_action_is_penalized():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=4))
    env.reset([sample])
    env.step(["Describe what is visible in the frames."])
    _, rewards, dones, infos = env.step(['<verdict>{"label":"fake","rationale":"too early"}</verdict>'])
    assert rewards[0] < 0.0
    assert not dones[0]
    assert not infos[0]["is_action_valid"]
