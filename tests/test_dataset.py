from fake_news_skillrl.dataset import infer_split, load_normalized_samples, normalize_records


def test_normalize_records_round_trip():
    rows = [
        {
            "sample_id": "id-1",
            "post_text": "post",
            "transcript": "transcript",
            "ocr_text": "ocr",
            "metadata": {"task_type": "misleading_caption"},
            "frames": [{"frame_id": "0", "path": "a.jpg", "description": "frame"}],
            "label": "fake",
            "gold_evidence": ["post"],
            "split": "train",
            "data_source": "unit",
        }
    ]
    normalized = normalize_records(rows)
    assert normalized[0]["label"] == "fake"
    assert normalized[0]["frames"][0]["path"] == "a.jpg"


def test_load_smoke_samples():
    samples = load_normalized_samples("data/raw/smoke_samples.jsonl")
    assert len(samples) == 3
    assert samples[0].sample_id == "smoke-001"


def test_infer_split_boundaries():
    assert infer_split(0, 10) == "train"
    assert infer_split(8, 10) == "val"
    assert infer_split(9, 10) == "test"


def test_gold_evidence_is_optional():
    rows = [
        {
            "sample_id": "id-2",
            "post_text": "post",
            "transcript": "",
            "ocr_text": "",
            "metadata": {"task_type": "unknown"},
            "frames": [],
            "label": "real",
            "split": "train",
            "data_source": "unit",
        }
    ]
    normalized = normalize_records(rows)
    assert normalized[0]["label"] == "real"
