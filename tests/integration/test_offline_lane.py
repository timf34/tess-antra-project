"""Offline lane through the CLI: fixtures -> reference -> packets -> verify -> run -> score -> report,
and the evaluator catches every planted fault while passing honest and disclosed-failure runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


def _npbench(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "npbench.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )


@pytest.fixture(scope="module")
def lane(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("lane")
    # a private copy of the repo-relative inputs the config needs
    (root / "docs").mkdir()
    shutil.copy(REPO / "docs" / "mode_definitions.yaml", root / "docs" / "mode_definitions.yaml")
    shutil.copy(REPO / "pyproject.toml", root / "pyproject.toml")
    cfg = yaml.safe_load((REPO / "configs" / "offline.yaml").read_text())
    cfg["study"]["version"] = "itest"
    cfg["evaluation"]["bundle_ids"] = ["fx_positive", "fx_null"]
    cfg["data"]["candidate_ids"] = ["distress_aversion"]
    cfg["runner"]["fake_variant"] = None
    (root / "configs").mkdir()
    (root / "configs" / "offline.yaml").write_text(yaml.safe_dump(cfg))
    steps = [
        (
            "fixtures",
            ["fixtures", "build", "--config", "configs/offline.yaml", "--out", "artifacts/fixtures"],
        ),
        (
            "reference",
            ["reference", "build", "--config", "configs/offline.yaml", "--data", "artifacts/fixtures"],
        ),
        ("packets", ["packets", "build", "--config", "configs/offline.yaml", "--data", "artifacts/fixtures"]),
        ("verify", ["packets", "verify", "--config", "configs/offline.yaml"]),
        ("run", ["run", "--config", "configs/offline.yaml", "--out", "runs/offline", "--limit", "24"]),
        ("score", ["score", "--config", "configs/offline.yaml", "--runs", "runs/offline"]),
        (
            "report",
            [
                "report",
                "--config",
                "configs/offline.yaml",
                "--runs",
                "runs/offline",
                "--out",
                "reports/offline",
            ],
        ),
    ]
    outputs = {}
    for name, args in steps:
        r = _npbench(*args, cwd=root)
        assert r.returncode == 0, f"{name} failed: {r.stdout[-800:]}\n{r.stderr[-1500:]}"
        outputs[name] = r.stdout
    return {"root": root, "outputs": outputs}


def test_lane_runs_and_reports_synthetic_banner(lane):
    report = (lane["root"] / "reports" / "offline" / "tier2_report.md").read_text()
    assert "SYNTHETIC / OFFLINE DATA" in report
    scores = json.loads((lane["root"] / "runs" / "offline" / "scores.json").read_text())
    assert scores["ledger_ok"] is True
    assert scores["n_evaluated"] == 24


def test_evaluator_catches_planted_faults_and_passes_honest(lane):
    scores = json.loads((lane["root"] / "runs" / "offline" / "scores.json").read_text())
    by = scores["by_variant"]
    assert by["honest"]["critical"] == 0, by["honest"]
    for v in ("wrong_sign", "omit", "fake_audit"):
        if v in by:
            assert by[v]["critical"] == by[v]["n"], (v, by[v])
    # construction-as-test is detectable only where construction and held-out values differ (fx_positive)
    for r in scores["runs"]:
        if r.get("fake_variant") == "construction_as_test" and r["bundle_id"] == "fx_positive":
            assert "generalization" in r["critical_categories"], r
    # leakage can only occur where something is fitted (activation arm); behavioural leak runs equal honest
    for r in scores["runs"]:
        if r.get("fake_variant") == "leak":
            if r["method"] == "activation_contrasts":
                assert "computation" in r["critical_categories"], r
            else:
                assert not r["critical_failure"], r
    if "disclosed_failure" in by:
        assert by["disclosed_failure"]["critical"] == 0, by["disclosed_failure"]
    expected = {
        "wrong_sign": "computation",
        "omit": "reporting",
        "fake_audit": "auditing",
    }
    for v, cat in expected.items():
        if v in by:
            assert cat in by[v]["categories"], (v, by[v])
    # an unsupported causal narrative is flagged (provisional) only where the reference does not support it
    for r in scores["runs"]:
        if r.get("fake_variant") == "contradiction":
            if r["bundle_id"] == "fx_null" or r["method"] == "behavioral_continuations":
                assert "interpretation" in r["critical_categories"], r
            elif r["bundle_id"] == "fx_positive":
                assert "interpretation" not in r["critical_categories"], r


def test_resume_does_not_repeat_runs(lane):
    r = _npbench(
        "run", "--config", "configs/offline.yaml", "--out", "runs/offline", "--limit", "24", cwd=lane["root"]
    )
    assert r.returncode == 0
    d = json.loads(r.stdout[r.stdout.index("{") :])
    assert d["skipped_completed"] == 24 and d["completed"] == 0


def test_mock_receipts_fail_live_validation(lane):
    from npbench.runner.auditor import verify_receipt_against_ledger
    from npbench.runner.ledger import Ledger

    ledger = Ledger(lane["root"] / "runs" / "offline" / "ledger.jsonl")
    receipts = [e["payload"] for e in ledger.find(kind="audit_receipt")]
    assert receipts
    rec = receipts[0]
    assert verify_receipt_against_ledger(rec, ledger, require_live=False) == []
    assert any("mock" in p for p in verify_receipt_against_ledger(rec, ledger, require_live=True))
    fake = {**rec, "audit_id": "aud_fabricated"}
    assert any("fabricated" in p for p in verify_receipt_against_ledger(fake, ledger, require_live=False))
