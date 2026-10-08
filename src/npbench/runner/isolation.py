"""Workspace isolation for evaluated assistant processes.

A workspace holds exactly two things: a read-only copy of the task packet and a writable work
directory. No evaluator source, gold files, other runs, repository checkout, host ledger, or API keys
are inside it. Tool paths are confined to the workspace, the environment of executed code is
scrubbed, and code runs in a fresh network namespace when ``unshare`` supports it (recorded as the
achieved ``isolation_level``). Container-level isolation (docker --network none) is the production
path and is documented in docs/runbook.md; this module is the host-side fallback and is tested for
its actual restrictions, not its configuration text."""

from __future__ import annotations

import os
import resource
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SECRET_PREFIXES = (
    "ANTHROPIC",
    "OPENAI",
    "OPENROUTER",
    "HF_",
    "HUGGING",
    "RUNPOD",
    "AWS_",
    "GH_",
    "GITHUB",
    "CLAUDE",
    "GOOGLE",
    "GCP",
    "CLOUDSDK",
    "SESSION",
    "NPBENCH_HOST",
)


class AccessDenied(PermissionError):
    pass


@dataclass
class Workspace:
    root: Path
    packet: Path  # read-only copy
    work: Path  # writable
    isolation_level: str = "process_only"
    denied: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, root: Path, packet_src: Path) -> Workspace:
        root = Path(root)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        packet = root / "packet"
        shutil.copytree(packet_src, packet)
        for p in packet.rglob("*"):
            if p.is_dir():
                p.chmod(
                    stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
                )
            else:
                p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        work = root / "work"
        work.mkdir()
        work.chmod(0o777)  # outputs are written by an unprivileged uid when the host runs as root
        root.chmod(0o755)
        _ensure_traversable(root)
        ws = cls(root=root, packet=packet, work=work)
        ws.isolation_level = detect_isolation_level()
        return ws

    # -- path confinement ---------------------------------------------------------------------
    def resolve(self, rel: str, *, writable: bool = False) -> Path:
        """Resolve a tool path inside the workspace. ``packet/...`` is read-only; everything else is
        relative to ``work``. Any escape (.., symlinks, absolute paths outside) is denied and logged."""
        rel = str(rel)
        base = self.work
        sub = rel
        if rel.startswith("packet/") or rel == "packet":
            base = self.packet
            sub = rel[len("packet") :].lstrip("/")
            if writable:
                self.denied.append({"path": rel, "reason": "packet is read-only"})
                raise AccessDenied(f"packet is read-only: {rel}")
        elif rel.startswith("work/"):
            sub = rel[len("work/") :]
        candidate = (base / sub).resolve() if not os.path.isabs(sub) else Path(sub).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            self.denied.append({"path": rel, "reason": "outside workspace"})
            raise AccessDenied(f"path outside the workspace: {rel}") from None
        return candidate

    def teardown(self) -> None:
        for p in self.packet.rglob("*"):
            try:
                p.chmod(p.stat().st_mode | stat.S_IWUSR)
            except OSError:
                pass


UNPRIV_UID = 65534  # nobody
UNPRIV_GID = 65534


def _ensure_traversable(path: Path) -> None:
    """Make every parent we own traversable (o+rx) so an unprivileged sandbox uid can reach the workspace."""
    for p in [path, *path.parents]:
        try:
            st_ = p.stat()
            if st_.st_uid == os.getuid() and (st_.st_mode & 0o005) != 0o005:
                p.chmod(st_.st_mode | 0o055)
        except OSError:
            pass


def _unshare_ok(prefix: list[str]) -> bool:
    try:
        r = subprocess.run([*prefix, "true"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def sandbox_prefix() -> tuple[list[str], str]:
    """Command prefix and achieved isolation level.

    * root host with unshare + setpriv: new network namespace, then drop to an unprivileged uid so
      file permissions (read-only packet) are actually enforced -> "netns_unprivileged"
    * non-root host with unprivileged user namespaces: unshare -r -n -> "netns"
    * otherwise: plain subprocess with scrubbed env and rlimits -> "process_only"
    """
    if shutil.which("unshare") is None:
        return [], "process_only"
    if os.geteuid() == 0 and shutil.which("setpriv") is not None:
        prefix = [
            "unshare",
            "-n",
            "--",
            "setpriv",
            f"--reuid={UNPRIV_UID}",
            f"--regid={UNPRIV_GID}",
            "--clear-groups",
        ]
        if _unshare_ok(prefix):
            return prefix, "netns_unprivileged"
    if _unshare_ok(["unshare", "-r", "-n"]):
        return ["unshare", "-r", "-n"], "netns"
    return [], "process_only"


def detect_isolation_level() -> str:
    return sandbox_prefix()[1]


def scrubbed_env(ws: Workspace) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(ws.work),
        "TMPDIR": str(ws.work),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "NPBENCH_PACKET": str(ws.packet),
        "NPBENCH_WORK": str(ws.work),
        "LANG": "C.UTF-8",
    }
    for k, v in os.environ.items():
        if k in env or any(k.upper().startswith(p) for p in SECRET_PREFIXES):
            continue
        if k.lower().endswith(("_key", "_token", "_secret", "password")):
            continue
        if k in ("LD_LIBRARY_PATH", "VIRTUAL_ENV", "PYTHONPATH"):
            continue
        if k.startswith(("LC_", "TZ")):
            env[k] = v
    # keep the proxy variables out: executed code must not reach the network even without netns
    return env


def _limits(mem_mb: int, cpu_s: int):
    def pre() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        except (ValueError, OSError):
            pass

    return pre


@dataclass
class ExecResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    isolation_level: str
    truncated: bool


def run_confined(
    ws: Workspace,
    argv: list[str],
    *,
    timeout_s: int = 120,
    mem_mb: int = 2048,
    output_cap: int = 20_000,
    python_executable: str | None = None,
) -> ExecResult:
    """Run ``argv`` with cwd=work, scrubbed env, rlimits, and a network namespace when available.
    ``argv`` entries equal to "python" are replaced by the configured interpreter in isolated mode."""
    py = python_executable or sys.executable
    cmd = [py if a == "python" else a for a in argv]
    if cmd and cmd[0] == py and "-I" not in cmd[1:2]:
        cmd = [py, "-I", *cmd[1:]]
    prefix, level = sandbox_prefix()
    cmd = [*prefix, *cmd]
    try:
        r = subprocess.run(
            cmd,
            cwd=ws.work,
            env=scrubbed_env(ws),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_limits(mem_mb, timeout_s + 5),
        )
        out, err, code, to = r.stdout, r.stderr, r.returncode, False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        code, to = None, True
    truncated = len(out) > output_cap or len(err) > output_cap
    return ExecResult(code, out[:output_cap], err[:output_cap], to, level, truncated)
