from fake_news_skillrl.parser import parse_action


def test_parse_valid_inspect():
    parsed = parse_action("<inspect>frame:1</inspect>", max_frame_index=2)
    assert parsed.is_valid
    assert parsed.action_type == "inspect"
    assert parsed.payload["frame_index"] == 1


def test_parse_invalid_frame():
    parsed = parse_action("<inspect>frame:4</inspect>", max_frame_index=1)
    assert not parsed.is_valid


def test_parse_valid_verdict():
    action = '<verdict>{"label":"fake","rationale":"bad claim","evidence":["proof"]}</verdict>'
    parsed = parse_action(action, max_frame_index=0)
    assert parsed.is_valid
    assert parsed.payload["label"] == "fake"


def test_parse_mixed_action_invalid():
    action = "<inspect>post_text</inspect><verdict>{}</verdict>"
    parsed = parse_action(action, max_frame_index=0)
    assert not parsed.is_valid


def test_verdict_with_embedded_inspect_stays_verdict_invalid():
    action = '<verdict>{"label":"fake","rationale":"x","evidence":[],"action":"<inspect>post_text</inspect>"}</verdict><|im_end|>'
    parsed = parse_action(action, max_frame_index=0)
    assert not parsed.is_valid
    assert parsed.action_type == "verdict"
