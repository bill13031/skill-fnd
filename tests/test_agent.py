from fake_news_skillrl.agent import QwenVLAgent, select_inference_device
from fake_news_skillrl.schema import FakeNewsSample, FrameRecord


def test_select_inference_device_prefers_cuda():
    assert select_inference_device(True) == "cuda"


def test_select_inference_device_falls_back_to_cpu():
    assert select_inference_device(False) == "cpu"


def test_build_messages_attaches_images_only_on_first_step():
    sample = FakeNewsSample(
        sample_id="sample-1",
        post_text="post",
        transcript="",
        ocr_text="",
        metadata={"task_type": "unknown"},
        frames=[FrameRecord(frame_id="0", path="https://example.com/frame.jpg", description="frame desc")],
        label="real",
        gold_evidence=[],
        split="train",
        data_source="test",
    )
    agent = object.__new__(QwenVLAgent)
    agent.attach_frames_first_step_only = True

    first_step_messages = agent._build_messages(sample, "obs", [])
    later_step_messages = agent._build_messages(sample, "obs", ["create: hypothesis"])

    first_types = [part["type"] for part in first_step_messages[0]["content"]]
    later_types = [part["type"] for part in later_step_messages[0]["content"]]

    assert "image" in first_types
    assert "image" not in later_types
