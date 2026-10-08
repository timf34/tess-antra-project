"""Append-only, hash-chained host-side evidence ledger.

Each entry records ``prev_hash`` and its own ``hash`` = sha256(prev_hash || canonical(entry body)).
``verify`` detects edits, reordering and truncation (a separately stored head file records the
expected final sequence number and hash). A hash chain alone cannot stop a privileged builder from
rewriting everything; the real boundary is that the evaluated assistant process never has access to
the ledger directory or to the head file (see ``runner/isolation.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..util import canonical_json, sha256_text, utc_now_iso

GENESIS = "0" * 64


@dataclass
class LedgerVerification:
    ok: bool
    entries: int
    problems: list[str]


class Ledger:
    def __init__(self, path: str | Path, head_path: str | Path | None = None):
        self.path = Path(path)
        self.head_path = Path(head_path) if head_path else self.path.with_suffix(".head.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._prev = self._load_tail()

    # -- internals -------------------------------------------------------------------------
    def _load_tail(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS
        last: dict[str, Any] | None = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
        if last is None:
            return 0, GENESIS
        return int(last["seq"]) + 1, str(last["hash"])

    @staticmethod
    def entry_hash(prev_hash: str, body: dict[str, Any]) -> str:
        return sha256_text(prev_hash + canonical_json(body))

    # -- public API ------------------------------------------------------------------------
    def append(self, kind: str, payload: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
        body = {
            "seq": self._seq,
            "ts": utc_now_iso(),
            "kind": kind,
            "run_id": run_id,
            "payload": payload,
            "prev_hash": self._prev,
        }
        h = self.entry_hash(self._prev, body)
        entry = {**body, "hash": h}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical_json(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq += 1
        self._prev = h
        self._write_head()
        return entry

    def _write_head(self) -> None:
        tmp = self.head_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": self._seq, "head_hash": self._prev}, f)
        os.replace(tmp, self.head_path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def verify(self) -> LedgerVerification:
        problems: list[str] = []
        prev = GENESIS
        n = 0
        for i, e in enumerate(self.entries()):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("seq") != i:
                problems.append(f"entry {i}: seq {e.get('seq')} != {i}")
            if body.get("prev_hash") != prev:
                problems.append(f"entry {i}: prev_hash mismatch")
            if self.entry_hash(prev, body) != e.get("hash"):
                problems.append(f"entry {i}: hash mismatch (edited)")
            prev = e.get("hash", "")
            n = i + 1
        if self.head_path.exists():
            with open(self.head_path, encoding="utf-8") as f:
                head = json.load(f)
            if head.get("entries") != n:
                problems.append(f"truncation: head says {head.get('entries')} entries, found {n}")
            elif head.get("head_hash") != prev:
                problems.append("head hash mismatch")
        return LedgerVerification(ok=not problems, entries=n, problems=problems)

    def find(self, kind: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        return [
            e
            for e in self.entries()
            if (kind is None or e["kind"] == kind) and (run_id is None or e.get("run_id") == run_id)
        ]
