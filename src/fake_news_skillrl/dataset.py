from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io_utils import dump_jsonl, load_jsonl
from .schema import FakeNewsSample, normalize_sample


@dataclass(slots=True)
class FrameExtractionResult:
    frames: List[Dict[str, str]]
    status: str
    reason: str


def load_normalized_samples(path: str | Path) -> List[FakeNewsSample]:
    rows = load_jsonl(path)
    return [normalize_sample(row) for row in rows]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative_to_project_root(path: Path) -> str:
    project_root = _project_root()
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


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
    sample_id: str,
    fps: float = 2.0,
    max_frames: int = 16,
) -> FrameExtractionResult:
    try:
        import cv2
    except ImportError:
        return FrameExtractionResult([], "missing_or_unavailable", "opencv_not_installed")

    video_path = Path(video_path)
    frames_output_dir = Path(frames_output_dir)
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return FrameExtractionResult([], "missing_or_unavailable", "video_open_failed")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        capture.release()
        return FrameExtractionResult([], "missing_or_unavailable", "frame_count_zero")
    if video_fps <= 0:
        video_fps = 25.0

    frame_positions = _select_frame_positions_from_fps(
        frame_count=frame_count,
        video_fps=video_fps,
        sampling_fps=fps,
        max_frames=max_frames,
    )
    sample_output_dir = frames_output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    frames: List[Dict[str, str]] = []
    used_positions = set()
    candidate_offsets = [0, 3, -3, 6, -6, 9, -9]
    read_failures = 0
    rejected_low_variance = 0
    for output_index, frame_position in enumerate(frame_positions):
        selected_frame = None
        selected_position = None
        for offset in candidate_offsets:
            candidate_position = min(max(frame_position + offset, 0), frame_count - 1)
            if candidate_position in used_positions:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_position)
            ok, frame = capture.read()
            if not ok:
                read_failures += 1
                continue
            stddev = cv2.meanStdDev(frame)[1].mean()
            if stddev < 8.0:
                rejected_low_variance += 1
                continue
            selected_frame = frame
            selected_position = candidate_position
            break
        if selected_frame is None:
            continue
        used_positions.add(selected_position)
        output_path = sample_output_dir / f"{output_index:04d}.jpg"
        cv2.imwrite(str(output_path), selected_frame)
        frames.append(
            {
                "frame_id": str(output_index),
                "path": _relative_to_project_root(output_path),
                "description": (
                    f"Sampled frame {output_index} from video {video_path.stem} at frame {selected_position} "
                    f"({selected_position / video_fps:.2f}s)"
                ),
            }
        )

    capture.release()
    if frames:
        return FrameExtractionResult(frames, "available", "ok")
    if rejected_low_variance and not read_failures:
        return FrameExtractionResult([], "missing_or_unavailable", "all_sampled_frames_rejected_low_variance")
    if read_failures and not rejected_low_variance:
        return FrameExtractionResult([], "missing_or_unavailable", "all_sampled_frames_read_failed")
    if read_failures and rejected_low_variance:
        return FrameExtractionResult([], "missing_or_unavailable", "sampled_frames_failed_or_rejected")
    return FrameExtractionResult([], "missing_or_unavailable", "no_frames_selected")


def _select_frame_positions_from_fps(
    frame_count: int,
    video_fps: float,
    sampling_fps: float,
    max_frames: int,
) -> List[int]:
    if frame_count <= 0:
        return []
    if sampling_fps <= 0:
        sampling_fps = 2.0
    if max_frames <= 0:
        max_frames = 1

    frame_step = max(int(round(video_fps / sampling_fps)), 1)
    positions = list(range(0, frame_count, frame_step))
    if not positions:
        positions = [0]

    if len(positions) <= max_frames:
        return positions

    if max_frames == 1:
        return [positions[len(positions) // 2]]

    last_index = len(positions) - 1
    selected = []
    for i in range(max_frames):
        idx = int(round(i * last_index / (max_frames - 1)))
        selected.append(positions[idx])
    return selected


def normalize_fakett_file(
    input_path: str | Path,
    output_path: str | Path,
    video_dir: str | Path,
    frames_output_dir: str | Path | None = None,
    fps: float = 2.0,
    max_frames: int = 16,
) -> List[FakeNewsSample]:
    raw_rows = load_jsonl(input_path)
    video_dir = Path(video_dir).expanduser()
    total = len(raw_rows)
    normalized_rows: List[Dict[str, Any]] = []
    extraction_reasons: Dict[str, int] = {}

    for index, row in enumerate(raw_rows):
        video_id = str(row["video_id"])
        video_path = video_dir / f"{video_id}.mp4"
        frames: List[Dict[str, str]] = []
        extraction_status = "missing_or_unavailable"
        extraction_reason = "frames_dir_not_provided"
        if frames_output_dir is None:
            extraction_reason = "frames_dir_not_provided"
        elif not video_path.exists():
            extraction_reason = "video_file_missing"
        else:
            extraction_result = extract_video_frames(
                video_path=video_path,
                frames_output_dir=frames_output_dir,
                sample_id=video_id,
                fps=fps,
                max_frames=max_frames,
            )
            frames = extraction_result.frames
            extraction_status = extraction_result.status
            extraction_reason = extraction_result.reason
        extraction_reasons[extraction_reason] = extraction_reasons.get(extraction_reason, 0) + 1

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
                    "frame_extraction_status": extraction_status,
                    "frame_extraction_reason": extraction_reason,
                },
                "frames": frames,
                "label": str(row.get("annotation", "real")).lower(),
                "split": infer_split(index, total),
                "data_source": "fakett",
            }
        )

    dump_jsonl(output_path, normalized_rows)
    samples = [normalize_sample(row) for row in normalized_rows]
    setattr(normalize_fakett_file, "last_extraction_reasons", extraction_reasons)
    return samples


def split_samples(samples: Iterable[FakeNewsSample]) -> Dict[str, List[FakeNewsSample]]:
    grouped: Dict[str, List[FakeNewsSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.split, []).append(sample)
    return grouped
