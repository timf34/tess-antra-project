"""Generic JSONL manifest importer. Each line is a CorpusRecord JSON object plus an
``excerpt_candidates`` list (see corpus/manifest_template.jsonl)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..corpus import ExcerptCandidate
from ..schemas import CorpusRecord, Origin


def import_jsonl_manifest(
    path: Path, *, origin: Origin = Origin.real_target, options: dict[str, Any] | None = None
) -> Iterator[tuple[CorpusRecord, list[ExcerptCandidate]]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cands_raw = row.pop("excerpt_candidates", [])
            row.setdefault("origin", origin.value)
            rec = CorpusRecord.model_validate(row)
            cands = [ExcerptCandidate.model_validate({"record_id": rec.record_id, **c}) for c in cands_raw]
            yield rec, cands
