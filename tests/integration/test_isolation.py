"""Actual isolation restrictions (not configuration text): path confinement, read-only packet,
scrubbed environment, no network in executed code, timeouts, denied access logged."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from npbench.runner.isolation import AccessDenied, Workspace, run_confined
from npbench.runner.tools import ToolHost


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    packet = tmp_path / "packet_src"
    (packet / "data").mkdir(parents=True)
    (packet / "required_results.json").write_text(json.dumps({"required_results": []}))
    (packet / "data" / "rows.jsonl").write_text('{"row_id": "r1"}\n')
    # a "gold" file and a "ledger" live OUTSIDE the workspace
    (tmp_path / "gold.json").write_text('{"secret": 1}')
    (tmp_path / "ledger.jsonl").write_text("{}\n")
    return Workspace.create(tmp_path / "ws", packet)


def test_workspace_contains_only_packet_and_work(ws: Workspace):
    names = sorted(p.name for p in ws.root.iterdir())
    assert names == ["packet", "work"]
    assert not list(ws.root.rglob("gold.json"))
    assert not list(ws.root.rglob("ledger.jsonl"))


def test_tool_paths_cannot_escape_and_denials_are_logged(ws: Workspace):
    host = ToolHost(ws, auditor=None, run_id="r")
    text, rec = host.call("read_file", {"path": "../gold.json"})
    assert "access denied" in text and rec["error"] == "access_denied"
    text, rec = host.call("read_file", {"path": "/etc/passwd"})
    assert rec["error"] == "access_denied"
    text, rec = host.call("read_file", {"path": "../../ledger.jsonl"})
    assert rec["error"] == "access_denied"
    assert len(ws.denied) == 3
    with pytest.raises(AccessDenied):
        ws.resolve("packet/required_results.json", writable=True)


def test_packet_is_read_only_for_executed_code(ws: Workspace):
    host = ToolHost(ws, auditor=None, run_id="r")
    host.call(
        "write_file",
        {
            "path": "w.py",
            "content": "import os\ntry:\n    open(os.environ['NPBENCH_PACKET'] + '/required_results.json', 'a').write('x')\n    print('WROTE')\nexcept Exception as e:\n    print('denied', type(e).__name__)\n",
        },
    )
    text, rec = host.call("run_python", {"path": "w.py"})
    assert "denied" in text and "WROTE" not in text


def test_executed_code_sees_no_credentials(ws: Workspace, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-secret")
    host = ToolHost(ws, auditor=None, run_id="r")
    host.call(
        "write_file",
        {
            "path": "env.py",
            "content": "import os, json\nprint(json.dumps(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k or 'SECRET' in k)))\n",
        },
    )
    text, _ = host.call("run_python", {"path": "env.py"})
    assert "[]" in text and "sk-test-secret" not in text


def test_executed_code_has_no_network_when_netns(ws: Workspace):
    if not (ws.isolation_level.startswith("netns") or ws.isolation_level == "macos_seatbelt"):
        pytest.skip("network namespace not available on this host")
    host = ToolHost(ws, auditor=None, run_id="r")
    host.call(
        "write_file",
        {
            "path": "net.py",
            "content": "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=3); print('NET_OK')\nexcept Exception as e:\n    print('blocked', type(e).__name__)\n",
        },
    )
    text, _ = host.call("run_python", {"path": "net.py"})
    assert "blocked" in text and "NET_OK" not in text


def test_timeout_and_output_cap(ws: Workspace):
    r = run_confined(ws, ["python", "-c", "import time; time.sleep(5)"], timeout_s=1)
    assert r.timed_out
    r = run_confined(ws, ["python", "-c", "print('x' * 100000)"], timeout_s=10, output_cap=1000)
    assert r.truncated and len(r.stdout) == 1000


def test_malformed_tool_arguments_do_not_crash(ws: Workspace):
    host = ToolHost(ws, auditor=None, run_id="r")
    text, rec = host.call("read_file", {"nope": 1})
    assert rec["error"] == "malformed_arguments"
    text, rec = host.call("read_file", "not-a-dict")  # type: ignore[arg-type]
    assert rec["error"] == "malformed_arguments"
    text, rec = host.call("no_such_tool", {})
    assert rec["error"] == "unknown_tool"


def test_writes_rejected_after_submit(ws: Workspace):
    host = ToolHost(ws, auditor=None, run_id="r")
    host.call("submit", {})
    text, rec = host.call("write_file", {"path": "late.txt", "content": "x"})
    assert rec["error"] == "after_submit"
    assert not (ws.work / "late.txt").exists()
    assert os.environ.get("NPBENCH_PACKET") is None  # host env untouched


def test_executed_python_cannot_read_host_gold(ws):
    if ws.isolation_level != "macos_seatbelt":
        pytest.skip("filesystem sandbox unavailable; live runner must fail closed")
    gold = ws.root.parent / "gold.json"
    result = run_confined(
        ws, ["python", "-c", f"from pathlib import Path; print(Path({str(gold)!r}).read_text())"]
    )
    assert result.returncode != 0 and "Operation not permitted" in result.stderr
    assert "secret" not in result.stdout


def test_workspace_does_not_change_ancestor_permissions(tmp_path):
    tmp_path.chmod(0o700)
    packet = tmp_path / "source"
    packet.mkdir()
    Workspace.create(tmp_path / "workspace", packet)
    assert tmp_path.stat().st_mode & 0o777 == 0o700
