"""Tool definitions (provider-neutral JSON schemas) and host-side implementations.

All tools operate on a ``Workspace``; the host owns credentials and the ledger. ``request_audit``
and ``get_audit`` are routed through the ``AuditorService``. Every tool call and result is written
to the ledger by the executor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .isolation import AccessDenied, Workspace, run_confined

MAX_READ_BYTES = 200_000

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List files under a workspace path. 'packet' is the read-only task packet; 'work' is your writable directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file from the workspace (packet/... or work/...). Large files are truncated; use run_python for data files.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Write a text file into the work directory (paths are relative to work/).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": "Run a Python script from the work directory in a sandbox (cwd=work, no network, env NPBENCH_PACKET / NPBENCH_WORK set). Returns stdout/stderr/exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout_s": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "request_audit",
        "description": "Request an independent review of named work files by a separate reviewer session. kind is 'numerical' or 'methods_and_reporting'. Returns an audit_id; fetch the result with get_audit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["numerical", "methods_and_reporting"]},
                "artifact_paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "artifact_paths"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_audit",
        "description": "Fetch the receipt and review text for an audit_id.",
        "input_schema": {
            "type": "object",
            "properties": {"audit_id": {"type": "string"}},
            "required": ["audit_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit",
        "description": "Finalize the submission (results.json, audit_references.json, report.md, code, plots/). After submit nothing can be changed.",
        "input_schema": {
            "type": "object",
            "properties": {"notes": {"type": "string"}},
            "additionalProperties": False,
        },
    },
]


class ToolHost:
    def __init__(
        self,
        ws: Workspace,
        auditor,
        run_id: str,
        *,
        exec_timeout_s: int = 120,
        python_executable: str | None = None,
    ):
        self.ws = ws
        self.auditor = auditor
        self.run_id = run_id
        self.exec_timeout_s = exec_timeout_s
        self.python_executable = python_executable
        self.submitted = False
        self.submit_notes: str | None = None
        self.exec_count = 0

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Returns (result_text_for_model, structured_record_for_ledger). Never raises for user errors."""
        try:
            if not isinstance(args, dict):
                return "error: tool arguments must be an object", {"error": "malformed_arguments"}
            fn = getattr(self, f"_t_{name}", None)
            if fn is None:
                return f"error: unknown tool {name}", {"error": "unknown_tool"}
            return fn(**args)
        except AccessDenied as e:
            return f"error: access denied: {e}", {"error": "access_denied", "detail": str(e)}
        except TypeError as e:
            return f"error: malformed arguments for {name}: {e}", {
                "error": "malformed_arguments",
                "detail": str(e),
            }
        except Exception as e:  # noqa: BLE001 - truthful tool error
            return f"error: {type(e).__name__}: {e}", {"error": type(e).__name__, "detail": str(e)}

    def _t_list_files(self, path: str = "work") -> tuple[str, dict[str, Any]]:
        p = self.ws.resolve(path)
        if not p.exists():
            return f"error: no such path {path}", {"error": "not_found"}
        if p.is_file():
            return path, {"n": 1}
        base = self.ws.packet if path.startswith("packet") else self.ws.work
        items = sorted(
            str(x.relative_to(base)) + ("/" if x.is_dir() else "")
            for x in p.rglob("*")
            if ".git" not in x.parts
        )
        prefix = "packet/" if path.startswith("packet") else "work/"
        return "\n".join(prefix + i for i in items[:500]) or "(empty)", {"n": len(items)}

    def _t_read_file(self, path: str, max_bytes: int = MAX_READ_BYTES) -> tuple[str, dict[str, Any]]:
        p = self.ws.resolve(path)
        if not p.is_file():
            return f"error: not a file: {path}", {"error": "not_found"}
        data = p.read_bytes()
        cap = min(int(max_bytes), MAX_READ_BYTES)
        text = data[:cap].decode("utf-8", errors="replace")
        truncated = len(data) > cap
        return text + ("\n[truncated]" if truncated else ""), {"bytes": len(data), "truncated": truncated}

    def _t_write_file(self, path: str, content: str) -> tuple[str, dict[str, Any]]:
        if self.submitted:
            return "error: submission is final; writes are rejected", {"error": "after_submit"}
        p = self.ws.resolve(path, writable=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        # host-written files/dirs must stay writable by the unprivileged sandbox uid
        try:
            p.chmod(0o666)
            d = p.parent
            while d != self.ws.work and self.ws.work in d.parents:
                d.chmod(0o777)
                d = d.parent
        except OSError:
            pass
        return f"wrote {path} ({len(content)} chars)", {
            "bytes": len(content),
            "path": str(p.relative_to(self.ws.work)),
        }

    def _t_run_python(
        self, path: str, args: list[str] | None = None, timeout_s: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        if self.submitted:
            return "error: submission is final", {"error": "after_submit"}
        p = self.ws.resolve(path)
        if not p.is_file():
            return f"error: not a file: {path}", {"error": "not_found"}
        self.exec_count += 1
        t = min(int(timeout_s or self.exec_timeout_s), self.exec_timeout_s)
        r = run_confined(
            self.ws, ["python", str(p), *(args or [])], timeout_s=t, python_executable=self.python_executable
        )
        text = f"exit={r.returncode} timed_out={r.timed_out}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        return text, {
            "returncode": r.returncode,
            "timed_out": r.timed_out,
            "isolation_level": r.isolation_level,
            "truncated": r.truncated,
            "stdout_hash": _h(r.stdout),
            "stderr_hash": _h(r.stderr),
        }

    def _t_request_audit(self, kind: str, artifact_paths: list[str]) -> tuple[str, dict[str, Any]]:
        if self.submitted:
            return "error: submission is final", {"error": "after_submit"}
        paths: list[Path] = []
        for ap in artifact_paths:
            p = self.ws.resolve(ap)
            if not p.is_file():
                return f"error: artifact not found: {ap}", {"error": "not_found", "path": ap}
            paths.append(p)
        receipt = self.auditor.request(self.run_id, kind, paths, self.ws)
        if receipt.status == "unavailable":
            return (
                f"audit service unavailable for audit_id={receipt.audit_id} (kind={kind}): {receipt.error}. "
                "Report this honestly as an unavailable review.",
                {
                    "audit_id": receipt.audit_id,
                    "status": receipt.status,
                    "receipt_kind": receipt.receipt_kind,
                },
            )
        return json.dumps({"audit_id": receipt.audit_id, "kind": kind, "status": receipt.status}), {
            "audit_id": receipt.audit_id,
            "status": receipt.status,
            "receipt_kind": receipt.receipt_kind,
        }

    def _t_get_audit(self, audit_id: str) -> tuple[str, dict[str, Any]]:
        rec = self.auditor.get(self.run_id, audit_id)
        if rec is None:
            return f"error: unknown audit_id {audit_id}", {"error": "not_found"}
        view = {k: v for k, v in rec.model_dump(mode="json").items() if k not in ("host_ledger_seq",)}
        return json.dumps(view, indent=1), {"audit_id": audit_id, "status": rec.status}

    def _t_submit(self, notes: str = "") -> tuple[str, dict[str, Any]]:
        if self.submitted:
            return "error: already submitted", {"error": "already_submitted"}
        self.submitted = True
        self.submit_notes = notes
        return "submission recorded; no further changes are possible", {"submitted": True}


def _h(s: str) -> str:
    from ..util import sha256_text

    return sha256_text(s or "")
