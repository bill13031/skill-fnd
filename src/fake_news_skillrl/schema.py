from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


Label = Literal["fake", "real", "unverified"]


@dataclass(slots=True)
class FrameRecord:
    frame_id: str
    path: str
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], default_index: int) -> "FrameRecord":
        return cls(
            frame_id=str(payload.get("frame_id", default_index)),
            path=str(payload.get("path", "")),
            description=str(payload.get("description", "")),
        )


@dataclass(slots=True)
class FakeNewsSample:
    sample_id: str
    post_text: str
    transcript: str
    ocr_text: str
    metadata: Dict[str, Any]
    frames: List[FrameRecord]
    label: Label
    gold_evidence: List[str]
    split: str
    data_source: str

    @property
    def task_type(self) -> str:
        task_type = self.metadata.get("task_type", "misleading_caption")
        return str(task_type)

    @property
    def task_description(self) -> str:
        return (
            f"Verify whether the online post is factual. "
            f"Post text: {self.post_text.strip()} "
            f"Task type hint: {self.task_type.replace('_', ' ')}."
        )


@dataclass(slots=True)
class InspectionObservation:
    sample_id: str
    task_description: str
    visible_evidence: str
    inspected_items: List[str] = field(default_factory=list)
    available_frames: List[str] = field(default_factory=list)


def normalize_sample(raw: Dict[str, Any]) -> FakeNewsSample:
    required = [
        "sample_id",
        "post_text",
        "transcript",
        "ocr_text",
        "metadata",
        "frames",
        "label",
        "gold_evidence",
        "split",
        "data_source",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required sample keys: {missing}")

    frames = [
        FrameRecord.from_dict(item, default_index=index)
        for index, item in enumerate(raw["frames"])
    ]

    label = str(raw["label"]).lower()
    if label not in {"fake", "real", "unverified"}:
        raise ValueError(f"Unsupported label: {label}")

    return FakeNewsSample(
        sample_id=str(raw["sample_id"]),
        post_text=str(raw["post_text"]),
        transcript=str(raw["transcript"]),
        ocr_text=str(raw["ocr_text"]),
        metadata=dict(raw["metadata"]),
        frames=frames,
        label=label,  # type: ignore[arg-type]
        gold_evidence=[str(item) for item in raw["gold_evidence"]],
        split=str(raw["split"]),
        data_source=str(raw["data_source"]),
    )
