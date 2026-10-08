"""Hand-checked tests for npbench.util: canonical JSON, hashing, atomic writes, JSONL, stable keys."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime

import pytest

from npbench.util import (
    append_jsonl,
    canonical_json,
    read_json,
    read_jsonl,
    repo_root,
    sha256_bytes,
    sha256_file,
    sha256_obj,
    sha256_text,
    stable_key,
    utc_now_iso,
    write_json,
)

# Well-known SHA-256 digests: the FIPS 180 test vectors for "" and "abc", and the empty JSON object.
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA256_EMPTY_OBJECT = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


class TestCanonicalJson:
    def test_sorted_keys_no_whitespace_nested(self):
        obj = {"b": 1, "a": [1, 2, {"z": 0, "y": None}], "c": True}
        assert canonical_json(obj) == '{"a":[1,2,{"y":null,"z":0}],"b":1,"c":true}'

    def test_insertion_order_does_not_matter(self):
        first = {"x": 1, "y": {"k2": 2, "k1": 1}}
        second = {"y": {"k1": 1, "k2": 2}, "x": 1}
        assert list(first) != list(second)  # genuinely different insertion order
        assert canonical_json(first) == canonical_json(second)

    def test_list_order_is_preserved(self):
        # Lists are ordered data; only mapping keys are canonicalized.
        assert canonical_json([2, 1]) == "[2,1]"
        assert canonical_json([2, 1]) != canonical_json([1, 2])

    def test_non_ascii_kept_verbatim(self):
        assert canonical_json({"s": "é"}) == '{"s":"é"}'

    def test_nan_and_inf_are_rejected(self):
        with pytest.raises(ValueError):
            canonical_json({"x": math.nan})
        with pytest.raises(ValueError):
            canonical_json([math.inf])

    def test_round_trips_through_json(self):
        obj = {"a": [1, 2.5, "s", None, False], "b": {"c": {}}}
        assert json.loads(canonical_json(obj)) == obj


class TestHashing:
    def test_sha256_text_and_bytes_known_vectors(self):
        assert sha256_text("") == SHA256_EMPTY
        assert sha256_bytes(b"") == SHA256_EMPTY
        assert sha256_text("abc") == SHA256_ABC
        assert sha256_bytes(b"abc") == SHA256_ABC

    def test_sha256_obj_is_hash_of_canonical_json(self):
        assert sha256_obj({}) == SHA256_EMPTY_OBJECT
        obj = {"k": [1, {"b": 2, "a": 1}]}
        assert sha256_obj(obj) == hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()

    def test_sha256_obj_stable_across_key_order_and_calls(self):
        a = {"x": 1, "y": [1, 2], "z": {"q": "r", "p": "s"}}
        b = {"z": {"p": "s", "q": "r"}, "y": [1, 2], "x": 1}
        assert sha256_obj(a) == sha256_obj(b) == sha256_obj(a)

    def test_sha256_obj_changes_with_content(self):
        assert sha256_obj({"x": 1}) != sha256_obj({"x": 2})
        assert sha256_obj({"x": 1}) != sha256_obj({"x": "1"})  # JSON type matters
        assert sha256_obj([1, 2]) != sha256_obj([2, 1])

    def test_sha256_file_matches_hashlib_with_small_chunks(self, tmp_path):
        p = tmp_path / "blob.bin"
        p.write_bytes(b"abc")
        assert sha256_file(p) == SHA256_ABC
        assert sha256_file(p, chunk=1) == SHA256_ABC  # exercises the chunk loop byte by byte
        data = bytes(range(256)) * 3
        p.write_bytes(data)
        assert sha256_file(p, chunk=7) == hashlib.sha256(data).hexdigest()


class TestWriteJson:
    def test_write_creates_parents_and_leaves_no_temp_file(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "data.json"
        obj = {"b": 2, "a": [1, 2, 3]}
        write_json(path, obj)
        assert path.exists()
        assert read_json(path) == obj
        # The temp-file-plus-rename leaves exactly the target behind: no ".data.json.*.tmp" residue.
        assert [p.name for p in path.parent.iterdir()] == ["data.json"]

    def test_default_indent_preserves_insertion_order(self, tmp_path):
        path = tmp_path / "pretty.json"
        write_json(path, {"b": 2, "a": 1})
        assert path.read_text(encoding="utf-8") == '{\n  "b": 2,\n  "a": 1\n}\n'

    def test_indent_none_sorts_keys_compactly(self, tmp_path):
        path = tmp_path / "compact.json"
        write_json(path, {"b": 2, "a": 1}, indent=None)
        assert path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'

    def test_overwrite_replaces_previous_content(self, tmp_path):
        path = tmp_path / "data.json"
        write_json(path, {"v": 1})
        write_json(path, {"v": 2})
        assert read_json(path) == {"v": 2}
        assert [p.name for p in tmp_path.iterdir()] == ["data.json"]

    def test_failed_write_keeps_old_file_and_removes_temp(self, tmp_path):
        path = tmp_path / "data.json"
        write_json(path, {"v": "original"})
        with pytest.raises(TypeError):
            write_json(path, {"v": {1, 2}})  # a set is not JSON serializable
        assert read_json(path) == {"v": "original"}
        assert [p.name for p in tmp_path.iterdir()] == ["data.json"]

    def test_non_ascii_written_verbatim(self, tmp_path):
        path = tmp_path / "u.json"
        write_json(path, {"s": "é"}, indent=None)
        assert path.read_text(encoding="utf-8") == '{"s": "é"}\n'
        assert read_json(path) == {"s": "é"}


class TestJsonl:
    def test_append_then_read_round_trip_preserves_order(self, tmp_path):
        path = tmp_path / "nested" / "log.jsonl"  # parent directory is created on demand
        rows = [
            {"seq": 0, "k": "a"},
            {"seq": 1, "k": "b", "nested": {"z": [1, 2], "y": None}},
            {"seq": 2, "k": "é"},
        ]
        for r in rows:
            append_jsonl(path, r)
        assert read_jsonl(path) == rows

    def test_lines_are_canonical_json(self, tmp_path):
        path = tmp_path / "log.jsonl"
        append_jsonl(path, {"b": 1, "a": 2})
        append_jsonl(path, {"z": [3, 2, 1]})
        assert path.read_text(encoding="utf-8") == '{"a":2,"b":1}\n{"z":[3,2,1]}\n'

    def test_read_skips_blank_lines(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text('{"a":1}\n\n   \n{"b":2}\n', encoding="utf-8")
        assert read_jsonl(path) == [{"a": 1}, {"b": 2}]

    def test_read_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert read_jsonl(path) == []


class TestStableKey:
    def test_deterministic_and_prefix_of_sha256_of_parts_list(self):
        k = stable_key("study", "bundle", 3)
        assert k == stable_key("study", "bundle", 3)
        assert k == sha256_obj(["study", "bundle", 3])[:24]
        assert len(k) == 24
        assert set(k) <= set("0123456789abcdef")

    def test_different_inputs_give_different_keys(self):
        base = stable_key("study", "bundle", 3)
        assert stable_key("study", "bundle", 4) != base
        assert stable_key("bundle", "study", 3) != base  # order of parts matters
        assert stable_key("study", "bundle") != base  # arity matters
        assert stable_key("study", "bundle", "3") != base  # int vs str

    def test_dict_parts_are_key_order_independent(self):
        assert stable_key({"a": 1, "b": 2}) == stable_key({"b": 2, "a": 1})

    def test_filesystem_safe(self):
        key = stable_key("a/b", "c d", {"x": "y:z"})
        assert "/" not in key and " " not in key and ":" not in key


def test_utc_now_iso_is_utc_with_millisecond_precision():
    s = utc_now_iso()
    assert s.endswith("+00:00")
    parsed = datetime.fromisoformat(s)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0
    # "YYYY-MM-DDTHH:MM:SS.mmm+00:00": the fractional part has exactly three digits
    assert len(s.split(".")[1].split("+")[0]) == 3


def test_repo_root_finds_nearest_pyproject_upward(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert repo_root() == tmp_path.resolve()
