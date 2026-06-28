from __future__ import annotations

import json
from pathlib import Path

from long_context_eval_lab.schemas import Probe


def load_probes(path: str | Path) -> list[Probe]:
    """Load probes from JSONL.

    JSONL is used because it is simple, diff-friendly, and easy to extend.
    Each line is one `Probe` object.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Probe file not found: {file_path}")

    probes: list[Probe] = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            probes.append(Probe.model_validate(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 - include line number for user-facing error.
            raise ValueError(f"Invalid probe at {file_path}:{line_number}: {exc}") from exc

    if not probes:
        raise ValueError(f"No probes found in: {file_path}")
    return probes
