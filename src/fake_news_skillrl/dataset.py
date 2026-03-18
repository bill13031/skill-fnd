from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io_utils import dump_jsonl, load_jsonl
from .schema import FakeNewsSample, normalize_sample


def load_normalized_samples(path: str | Path) -> List[FakeNewsSample]:
    rows = load_jsonl(path)
    return [normalize_sample(row) for row in rows]


def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [normalize_sample(record) for record in records]
    serializable: List[Dict[str, Any]] = []
    for sample in normalized:
        payload = asdict(sample)
        serializable.append(payload)
    return serializable


def normalize_jsonl_file(input_path: str | Path, output_path: str | Path) -> List[FakeNewsSample]:
    raw_rows = load_jsonl(input_path)
    normalized_rows = normalize_records(raw_rows)
    dump_jsonl(output_path, normalized_rows)
    return [normalize_sample(row) for row in normalized_rows]


def infer_split(index: int, total: int) -> str:
    if total <= 1:
        return "train"
    ratio = index / total
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


def extract_video_frames(
    video_path: str | Path,
    frames_output_dir: str | Path,
    num_frames: int = 2,
) -> List[Dict[str, str]]:
    try:
        import cv2
    except ImportError:
        return []

    video_path = Path(video_path)
    frames_output_dir = Path(frames_output_dir)
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return []

    if num_frames <= 1:
        frame_positions = [frame_count // 2]
    else:
        step = max(frame_count // num_frames, 1)
        frame_positions = [min(i * step, frame_count - 1) for i in range(num_frames)]

    frames: List[Dict[str, str]] = []
    for output_index, frame_position in enumerate(frame_positions):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ok, frame = capture.read()
        if not ok:
            continue
        output_path = frames_output_dir / f"{video_path.stem}-{output_index}.jpg"
        cv2.imwrite(str(output_path), frame)
        frames.append(
            {
                "frame_id": str(output_index),
                "path": str(output_path),
                "description": f"Sampled frame {output_index} from video {video_path.stem}",
            }
        )

    capture.release()
    return frames


def normalize_fakett_file(
    input_path: str | Path,
    output_path: str | Path,
    video_dir: str | Path,
    frames_output_dir: str | Path | None = None,
    num_frames: int = 2,
) -> List[FakeNewsSample]:
    raw_rows = load_jsonl(input_path)
    video_dir = Path(video_dir).expanduser()
    total = len(raw_rows)
    normalized_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(raw_rows):
        video_id = str(row["video_id"])
        video_path = video_dir / f"{video_id}.mp4"
        frames: List[Dict[str, str]] = []
        if frames_output_dir is not None and video_path.exists():
            frames = extract_video_frames(
                video_path=video_path,
                frames_output_dir=frames_output_dir,
                num_frames=num_frames,
            )

        event = str(row.get("event", "")).strip()
        description = str(row.get("description", "")).strip()
        normalized_rows.append(
            {
                "sample_id": video_id,
                "post_text": description,
                "transcript": "",
                "ocr_text": "",
                "metadata": {
                    "platform": "fakett",
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "event": event,
                    "user_description": str(row.get("user_description", "")),
                    "user_certify": row.get("user_certify", 0),
                    "publish_time": row.get("publish_time"),
                    "task_type": "misleading_caption",
                },
                "frames": frames,
                "label": str(row.get("annotation", "unverified")).lower(),
                "gold_evidence": [event] if event else [],
                "split": infer_split(index, total),
                "data_source": "fakett",
            }
        )

    dump_jsonl(output_path, normalized_rows)
    return [normalize_sample(row) for row in normalized_rows]


def split_samples(samples: Iterable[FakeNewsSample]) -> Dict[str, List[FakeNewsSample]]:
    grouped: Dict[str, List[FakeNewsSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.split, []).append(sample)
    return grouped
