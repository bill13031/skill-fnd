from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: Any) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_jsonl(path: str | Path) -> List[Any]:
    rows: List[Any] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
