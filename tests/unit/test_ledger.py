"""Hash-chain tests for the host-side evidence ledger: edits, truncation and reordering are detected."""

from __future__ import annotations

import json

import pytest

from npbench.runner.ledger import GENESIS, Ledger, LedgerVerification
from npbench.util import canonical_json, sha256_text


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("run_started", {"n": 0}, run_id="run_a")
    led.append("audit", {"n": 1, "note": "é"}, run_id="run_a")
    led.append("run_started", {"n": 2}, run_id="run_b")
    led.append("submission", {"n": 3, "files": ["a", "b"]}, run_id=None)
    return led


def _lines(path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_lines(path, lines) -> None:
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")


class TestChainConstruction:
    def test_entries_are_sequential_and_chained_from_genesis(self, ledger):
        entries = ledger.entries()
        assert [e["seq"] for e in entries] == [0, 1, 2, 3]
        assert entries[0]["prev_hash"] == GENESIS
        for prev, cur in zip(entries, entries[1:], strict=False):
            assert cur["prev_hash"] == prev["hash"]
        assert len({e["hash"] for e in entries}) == 4

    def test_hash_is_sha256_of_prev_hash_and_canonical_body(self, ledger):
        for e in ledger.entries():
            body = {k: v for k, v in e.items() if k != "hash"}
            expected = sha256_text(e["prev_hash"] + canonical_json(body))
            assert e["hash"] == expected
            assert Ledger.entry_hash(e["prev_hash"], body) == expected

    def test_append_returns_the_stored_entry(self, tmp_path):
        led = Ledger(tmp_path / "l.jsonl")
        returned = led.append("k", {"x": 1}, run_id="r")
        assert led.entries() == [returned]
        assert set(returned) == {"seq", "ts", "kind", "run_id", "payload", "prev_hash", "hash"}
        assert returned["seq"] == 0
        assert returned["kind"] == "k"
        assert returned["payload"] == {"x": 1}
        assert returned["run_id"] == "r"
        assert returned["prev_hash"] == GENESIS

    def test_head_file_records_count_and_last_hash(self, ledger, tmp_path):
        head_path = tmp_path / "ledger.head.json"  # default: path.with_suffix(".head.json")
        assert ledger.head_path == head_path
        head = json.loads(head_path.read_text(encoding="utf-8"))
        assert head == {"entries": 4, "head_hash": ledger.entries()[-1]["hash"]}
        assert not head_path.with_suffix(".tmp").exists()  # head is written atomically

    def test_custom_head_path(self, tmp_path):
        led = Ledger(tmp_path / "l.jsonl", head_path=tmp_path / "custom_head.json")
        led.append("k", {})
        assert json.loads((tmp_path / "custom_head.json").read_text(encoding="utf-8"))["entries"] == 1
        assert not (tmp_path / "l.head.json").exists()


class TestVerify:
    def test_intact_chain_verifies(self, ledger):
        v = ledger.verify()
        assert isinstance(v, LedgerVerification)
        assert v.ok is True
        assert v.entries == 4
        assert v.problems == []

    def test_empty_ledger_verifies_with_zero_entries(self, tmp_path):
        led = Ledger(tmp_path / "fresh.jsonl")
        assert led.verify() == LedgerVerification(ok=True, entries=0, problems=[])

    def test_edited_payload_is_detected_as_hash_mismatch(self, ledger):
        lines = _lines(ledger.path)
        e1 = json.loads(lines[1])
        e1["payload"]["n"] = 999  # edit the body, keep the recorded hash
        lines[1] = canonical_json(e1)
        _write_lines(ledger.path, lines)
        v = ledger.verify()
        assert v.ok is False
        assert v.entries == 4
        assert v.problems == ["entry 1: hash mismatch (edited)"]

    def test_edited_metadata_field_is_detected(self, ledger):
        lines = _lines(ledger.path)
        e2 = json.loads(lines[2])
        e2["run_id"] = "run_forged"
        lines[2] = canonical_json(e2)
        _write_lines(ledger.path, lines)
        v = ledger.verify()
        assert v.ok is False
        assert "entry 2: hash mismatch (edited)" in v.problems

    def test_truncation_is_detected_via_head_file(self, ledger):
        lines = _lines(ledger.path)
        _write_lines(ledger.path, lines[:-1])
        v = ledger.verify()
        assert v.ok is False
        assert v.entries == 3
        assert v.problems == ["truncation: head says 4 entries, found 3"]

    def test_truncation_is_only_detectable_with_the_head_file(self, ledger):
        # The chain itself is self-consistent after dropping its tail; the host-held head file is
        # what makes truncation visible, which is why the assistant must never reach it.
        lines = _lines(ledger.path)
        _write_lines(ledger.path, lines[:-1])
        ledger.head_path.unlink()
        v = ledger.verify()
        assert v.ok is True
        assert v.entries == 3

    def test_replaced_tail_with_well_formed_entry_is_a_head_hash_mismatch(self, ledger):
        lines = _lines(ledger.path)
        kept = json.loads(lines[2])
        forged_body = {
            "seq": 3,
            "ts": "2026-01-01T00:00:00.000+00:00",
            "kind": "submission",
            "run_id": None,
            "payload": {"n": 3, "files": ["a", "forged"]},
            "prev_hash": kept["hash"],
        }
        forged = {**forged_body, "hash": Ledger.entry_hash(kept["hash"], forged_body)}
        _write_lines(ledger.path, lines[:-1] + [canonical_json(forged)])
        v = ledger.verify()
        assert v.ok is False
        assert v.entries == 4
        assert v.problems == ["head hash mismatch"]

    def test_swapping_middle_lines_breaks_seq_and_prev_hash(self, ledger):
        lines = _lines(ledger.path)
        lines[1], lines[2] = lines[2], lines[1]
        _write_lines(ledger.path, lines)
        v = ledger.verify()
        assert v.ok is False
        assert v.entries == 4
        assert "entry 1: seq 2 != 1" in v.problems
        assert "entry 2: seq 1 != 2" in v.problems
        assert "entry 1: prev_hash mismatch" in v.problems
        assert "entry 2: prev_hash mismatch" in v.problems
        # the entry after the swap no longer chains either, while entry 0 is untouched
        assert "entry 3: prev_hash mismatch" in v.problems
        assert not any(p.startswith("entry 0") for p in v.problems)
        # the last line is still the true tail, so the head hash itself still agrees
        assert "head hash mismatch" not in v.problems

    def test_swapping_tail_lines_also_breaks_head_hash(self, ledger):
        lines = _lines(ledger.path)
        lines[2], lines[3] = lines[3], lines[2]
        _write_lines(ledger.path, lines)
        v = ledger.verify()
        assert v.ok is False
        assert "entry 2: seq 3 != 2" in v.problems
        assert "entry 3: seq 2 != 3" in v.problems
        assert "entry 2: prev_hash mismatch" in v.problems
        assert "entry 3: prev_hash mismatch" in v.problems
        assert "head hash mismatch" in v.problems


class TestReopen:
    def test_reopened_ledger_continues_chain(self, ledger):
        last = ledger.entries()[-1]
        reopened = Ledger(ledger.path)
        e = reopened.append("resumed", {"n": 4}, run_id="run_b")
        assert e["seq"] == 4
        assert e["prev_hash"] == last["hash"]
        assert e["hash"] == Ledger.entry_hash(last["hash"], {k: v for k, v in e.items() if k != "hash"})
        v = reopened.verify()
        assert v.ok is True
        assert v.entries == 5
        head = json.loads(reopened.head_path.read_text(encoding="utf-8"))
        assert head == {"entries": 5, "head_hash": e["hash"]}
        # a second reopen sees the full chain
        assert [x["seq"] for x in Ledger(ledger.path).entries()] == [0, 1, 2, 3, 4]

    def test_reopen_tolerates_blank_lines(self, ledger):
        lines = _lines(ledger.path)
        _write_lines(ledger.path, [lines[0], "", lines[1], "   ", lines[2], lines[3], ""])
        reopened = Ledger(ledger.path)
        e = reopened.append("resumed", {})
        assert e["seq"] == 4
        assert reopened.verify().ok is True


class TestFind:
    def test_find_filters_by_kind_and_run_id(self, ledger):
        assert [e["seq"] for e in ledger.find()] == [0, 1, 2, 3]
        assert [e["seq"] for e in ledger.find(kind="run_started")] == [0, 2]
        assert [e["seq"] for e in ledger.find(run_id="run_a")] == [0, 1]
        assert [e["seq"] for e in ledger.find(kind="run_started", run_id="run_a")] == [0]
        assert ledger.find(kind="nope") == []
        assert ledger.find(kind="submission", run_id="run_a") == []
        assert ledger.find(run_id="run_b")[0]["payload"] == {"n": 2}

    def test_find_on_empty_ledger(self, tmp_path):
        assert Ledger(tmp_path / "x.jsonl").find(kind="anything") == []
