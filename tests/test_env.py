from fake_news_skillrl.dataset import load_normalized_samples
from fake_news_skillrl.env import FakeNewsEnv, FakeNewsEnvConfig


def test_env_rollout_with_manual_actions():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=4))
    observations = env.reset([sample])
    assert "Available Evidence" in observations[0]

    _, rewards, dones, infos = env.step(["<create>State the main claim.</create>"])
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["is_action_valid"]

    verdict = '<verdict>{"label":"fake","rationale":"unsupported absolute cure claim","evidence":["100% cure guaranteed"]}</verdict>'
    _, rewards, dones, infos = env.step([verdict])
    assert dones[0]
    assert rewards[0] > 0.0
    assert infos[0]["won"]

def test_intermediate_reasoning_action_is_recorded():
    sample = load_normalized_samples("data/raw/smoke_samples.jsonl")[0]
    env = FakeNewsEnv(FakeNewsEnvConfig(max_steps=4))
    env.reset([sample])
    _, rewards, dones, infos = env.step(["<use_skill>Apply source skepticism and cross-modal checking.</use_skill>"])
    assert rewards[0] == 0.0
    assert not dones[0]
    assert infos[0]["reasoning_action"] == "use_skill"
