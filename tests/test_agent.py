from fake_news_skillrl.agent import select_inference_device


def test_select_inference_device_prefers_cuda():
    assert select_inference_device(True) == "cuda"


def test_select_inference_device_falls_back_to_cpu():
    assert select_inference_device(False) == "cpu"
