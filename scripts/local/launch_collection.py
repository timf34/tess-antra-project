"""Supervise a single experiment-owned pod via Tim's rp workflow.

Run with `uv run --no-sync --env-file .env python scripts/local/launch_collection.py`.
The pushed branch, on-pod watchdog, local deadline and fresh pod are all mandatory.
Downloads and verifies results, publishes target-only data, then TERMINATES only this pod.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RP = Path("/Users/timf34/Documents/VSCode/runpod-runner/.venv/bin/rp")
NAME = os.environ.get("NPBENCH_RESUME_NAME") or "npbench-affect-" + time.strftime("%m%d-%H%M%S")
ART = ROOT / "artifacts/launch"
ART.mkdir(parents=True, exist_ok=True)
DEADLINE = time.time() + 10 * 3600
if os.environ.get("NPBENCH_RESUME_NAME"):
    active = json.loads((ART / "active.json").read_text())
    if active["name"] != NAME:
        raise ValueError("Can only resume this experiment's recorded pod")
    DEADLINE = active["deadline_epoch"]


def rp(*args, capture=False, timeout=1200):
    return subprocess.run(
        [str(RP), *map(str, args)],
        check=True,
        cwd=ROOT,
        env=os.environ,
        text=True,
        capture_output=capture,
        timeout=min(timeout, max(1, DEADLINE - time.time())),
    )


def main():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=ROOT, text=True
    ).split()[0]
    if sha != remote:
        raise RuntimeError("Push branch before provisioning")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit source before provisioning")
    envfile = ART / "pod.local.env"
    envfile.write_text(
        "\n".join(k + "=" + shlex.quote(os.environ[k]) for k in ("HF_TOKEN", "RUNPOD_API_KEY")) + "\n"
    )
    envfile.chmod(0o600)
    statefile = Path.home() / ".runpod-runner/pods" / f"{NAME}.json"
    success = False
    try:
        print("PROVISION", NAME, flush=True)
        if not os.environ.get("NPBENCH_RESUME_NAME"):
            rp(
                "up",
                "--name",
                NAME,
                "--gpu",
                "h200",
                "--gpus",
                "1",
                "--volume",
                "none",
                "--disk",
                "160",
                "--volume-size",
                "40",
            )
        state = json.loads(statefile.read_text())
        podid = state["id"]
        (ART / "active.json").write_text(
            json.dumps({"name": NAME, "id": podid, "deadline_epoch": DEADLINE, "source_commit": sha})
        )
        watchdog = ART / "npbench_watchdog.json"
        watchdog.write_text(json.dumps({"pod_id": podid, "deadline_epoch": DEADLINE}))
        for source, target in [
            (envfile, "npbench.local.env"),
            (watchdog, "npbench_watchdog.json"),
            (ROOT / "scripts/pod/watchdog.py", "npbench_watchdog.py"),
        ]:
            rp("scp", NAME, str(source), "pod:/workspace/" + target)
        if os.environ.get("NPBENCH_RESUME_NAME"):
            rp("ssh", NAME, "--", 'kill -0 "$(cat /workspace/npbench_watchdog_armed)"')
        else:
            rp(
                "ssh",
                NAME,
                "--",
                "chmod 600 /workspace/npbench.local.env; set -a; . /workspace/npbench.local.env; set +a; nohup python3 /workspace/npbench_watchdog.py > /workspace/watchdog.log 2>&1 < /dev/null &",
            )
            rp("ssh", NAME, "--", "sleep 2; test -s /workspace/npbench_watchdog_armed")
        print("WATCHDOG_ARMED", flush=True)
        rp("ssh", NAME, "--", "python3 -m venv --system-site-packages /workspace/tess-antra-project_venv")
        rp(
            "bootstrap",
            NAME,
            "--repo",
            "https://github.com/timf34/tess-antra-project.git",
            "--branch",
            branch,
            "--env",
            str(envfile),
            "--req",
            "scripts/pod/requirements-capture.txt",
            timeout=2400,
        )
        rp(
            "ssh",
            NAME,
            "--",
            "/workspace/tess-antra-project_venv/bin/pip install -e /workspace/tess-antra-project",
        )
        rp("run", NAME, "--job", "collection", "--dotenv", "--", "bash scripts/pod/collection_worker.sh")
        while time.time() < DEADLINE:
            time.sleep(30)
            try:
                tail = rp("logs", NAME, "--job", "collection", "-n", "8", capture=True, timeout=90).stdout
            except subprocess.CalledProcessError:
                time.sleep(10)
                continue
            print(tail, flush=True)
            if "EXIT=" in tail:
                if "EXIT=0" not in tail:
                    raise RuntimeError("Pod collection failed; see saved logs")
                break
        else:
            raise TimeoutError("Collection wall deadline reached")
        dest = ROOT / "artifacts/matched_affect_v1"
        if dest.exists():
            raise RuntimeError("Local result directory exists; refusing overwrite")
        rp(
            "scp",
            NAME,
            "pod:/workspace/tess-antra-project/artifacts/matched_affect_v1",
            str(dest),
            "-r",
            timeout=900,
        )
        from npbench.util import sha256_file

        for name, digest in json.loads((dest / "download_manifest.json").read_text()).items():
            if sha256_file(dest / name) != digest:
                raise RuntimeError("Download hash mismatch: " + name)
        # Raw target data only. Evaluator oracle, scoring, prompts and private API logs stay local.
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        repo = "timf34/npbench-matched-affect-v1"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        if api.repo_info(repo, repo_type="dataset").private:
            raise RuntimeError("Expected public target-data repo")
        result = api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=dest / "bundles",
            path_in_repo="bundles",
            commit_message="Exploratory Gemma4 target data; provisional modes, ratings pending",
        )
        print("PUBLIC_TARGET_DATA", result, flush=True)
        (ART / "success.json").write_text(
            json.dumps({"name": NAME, "id": podid, "source_commit": sha, "dataset": repo})
        )
        success = True
    finally:
        if statefile.exists():
            try:
                rp("logs", NAME, "--job", "collection", "-n", "100", capture=True, timeout=30)
            except Exception:
                pass
            # Stop on failure to preserve persistent diagnostics; delete only after verified downloads/upload.
            args = [str(RP), "down", NAME, "-y"] + (["--terminate"] if success else [])
            subprocess.run(args, check=True, cwd=ROOT, env=os.environ, timeout=120)
        envfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
