from pathlib import Path

from fake_news_skillrl.dataset import (
    _relative_to_project_root,
    _select_frame_positions_from_fps,
    infer_split,
    load_normalized_samples,
    normalize_records,
)
from scripts.prepare_dataset import _print_frame_extraction_summary


def test_normalize_records_round_trip():
    rows = [
        {
            "sample_id": "id-1",
            "post_text": "post",
            "transcript": "transcript",
            "ocr_text": "ocr",
            "metadata": {},
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
            "metadata": {},
            "frames": [],
            "label": "real",
            "split": "train",
            "data_source": "unit",
        }
    ]
    normalized = normalize_records(rows)
    assert normalized[0]["label"] == "real"


def test_select_frame_positions_from_fps_respects_max_frames():
    positions = _select_frame_positions_from_fps(
        frame_count=300,
        video_fps=30.0,
        sampling_fps=2.0,
        max_frames=5,
    )
    assert len(positions) == 5
    assert positions[0] == 0
    assert positions[-1] < 300


def test_relative_to_project_root_returns_relative_path_inside_repo():
    repo_root = Path.cwd()
    inside_repo = repo_root / "data" / "frames" / "fakett" / "abc" / "0000.jpg"
    assert _relative_to_project_root(inside_repo) == "data/frames/fakett/abc/0000.jpg"


def test_prepare_dataset_prints_frame_extraction_summary(capsys):
    class Sample:
        def __init__(self, metadata):
            self.metadata = metadata

    samples = [
        Sample({"frame_extraction_status": "available", "frame_extraction_reason": "ok"}),
        Sample({"frame_extraction_status": "missing_or_unavailable", "frame_extraction_reason": "video_open_failed"}),
    ]

    _print_frame_extraction_summary(samples)
    output = capsys.readouterr().out
    assert "Frame extraction summary:" in output
    assert "status[available] = 1" in output
    assert "reason[video_open_failed] = 1" in output
