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


def split_samples(samples: Iterable[FakeNewsSample]) -> Dict[str, List[FakeNewsSample]]:
    grouped: Dict[str, List[FakeNewsSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.split, []).append(sample)
    return grouped
