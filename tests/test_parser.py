from fake_news_skillrl.parser import parse_action


def test_parse_valid_create():
    parsed = parse_action("<create>Check whether the claim is concrete.</create>", max_frame_index=2)
    assert parsed.is_valid
    assert parsed.action_type == "create"
    assert "claim" in parsed.payload["content"]


def test_parse_invalid_empty_create():
    parsed = parse_action("<create></create>", max_frame_index=1)
    assert not parsed.is_valid


def test_parse_valid_verdict():
    action = '<verdict>{"label":"fake","rationale":"bad claim"}</verdict>'
    parsed = parse_action(action, max_frame_index=0)
    assert parsed.is_valid
    assert parsed.payload["label"] == "fake"


def test_parse_unverified_verdict_invalid():
    action = '<verdict>{"label":"unverified","rationale":"not enough"}</verdict>'
    parsed = parse_action(action, max_frame_index=0)
    assert not parsed.is_valid
    assert parsed.error == "Unsupported verdict label."


def test_parse_mixed_action_invalid():
    action = "<check>compare metadata</check><verdict>{}</verdict>"
    parsed = parse_action(action, max_frame_index=0)
    assert not parsed.is_valid


def test_verdict_with_embedded_action_stays_verdict_invalid():
    action = '<verdict>{"label":"fake","rationale":"x","action":"<check>metadata</check>"}</verdict><|im_end|>'
    parsed = parse_action(action, max_frame_index=0)
    assert not parsed.is_valid
    assert parsed.action_type == "verdict"
