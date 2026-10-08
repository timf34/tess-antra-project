"""Attach to an already-running collection job and finish it: wait, download, verify, publish, terminate.

Mirrors the post-collection logic of launch_collection.py without provisioning or launching anything.
Run with `uv run --no-sync --env-file .env python scripts/local/finalize_collection.py`.
Refuses to run if the recorded pod or job does not match artifacts/launch/active.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RP = Path("/Users/timf34/Documents/VSCode/runpod-runner/.venv/bin/rp")
ART = ROOT / "artifacts/launch"
REMOTE_RESULTS = "pod:/workspace/tess-antra-project/artifacts/matched_affect_v1"
DATASET = "timf34/npbench-matched-affect-v1"
JOB = os.environ.get("NPBENCH_JOB", "collection")


def rp(*args, capture=False, timeout=1200, check=True):
    return subprocess.run(
        [str(RP), *map(str, args)], check=check, cwd=ROOT, env=os.environ, text=True,
        capture_output=capture, timeout=timeout,
    )


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def main():
    active = json.loads((ART / "active.json").read_text())
    name, podid, deadline, sha = active["name"], active["id"], active["deadline_epoch"], active["source_commit"]
    state = json.loads((Path.home() / ".runpod-runner/pods" / f"{name}.json").read_text())
    if state["id"] != podid:
        raise ValueError("Local rp state does not match the recorded experiment pod")
    if (ART / "success.json").exists():
        raise RuntimeError("success.json already exists; nothing to finalize")
    dest = ROOT / "artifacts/matched_affect_v1"
    if dest.exists():
        raise RuntimeError("Local result directory exists; refusing overwrite")

    # 1. Wait for the existing job to exit. Never launch.
    log("ATTACH", name, podid, "deadline", time.strftime("%FT%TZ", time.gmtime(deadline)))
    while True:
        if time.time() > deadline - 600:
            raise TimeoutError("Within 10 min of the watchdog deadline; not safe to continue")
        try:
            tail = rp("logs", name, "--job", JOB, "-n", "6", capture=True, timeout=90).stdout
        except subprocess.CalledProcessError:
            time.sleep(15)
            continue
        if "EXIT=" in tail:
            log(tail.strip().splitlines()[-1])
            if "EXIT=0" not in tail:
                raise RuntimeError("Pod collection failed; see saved logs")
            break
        time.sleep(30)

    # 2. Preserve the full remote log before anything else.
    rp("scp", name, "pod:/workspace/collection.log", str(ART / f"collection_{sha[:12]}_{int(time.time())}.log"), timeout=90)
    rp("scp", name, "pod:/workspace/watchdog.log", str(ART / "watchdog_remote.log"), timeout=60, check=False)

    # 3. Download and verify.
    log("DOWNLOAD")
    rp("scp", name, REMOTE_RESULTS, str(dest), "-r", timeout=900)
    sys.path.insert(0, str(ROOT / "src"))
    from npbench.util import sha256_file

    manifest = json.loads((dest / "download_manifest.json").read_text())
    for fname, digest in manifest.items():
        if sha256_file(dest / fname) != digest:
            raise RuntimeError("Download hash mismatch: " + fname)
    if not (dest / "COLLECTION_COMPLETE").exists():
        raise RuntimeError("COLLECTION_COMPLETE marker missing")
    log("VERIFIED", len(manifest), "files")

    # 4. Publish target bundles only.
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(DATASET, repo_type="dataset", private=False, exist_ok=True)
    if api.repo_info(DATASET, repo_type="dataset").private:
        raise RuntimeError("Expected public target-data repo")
    result = api.upload_folder(
        repo_id=DATASET, repo_type="dataset", folder_path=dest / "bundles", path_in_repo="bundles",
        commit_message="Exploratory Gemma4 target data; provisional modes, ratings pending",
    )
    log("PUBLIC_TARGET_DATA", result)

    # 5. Record success in the launcher's schema, then terminate only this pod.
    (ART / "success.json").write_text(json.dumps({"name": name, "id": podid, "source_commit": sha, "dataset": DATASET}))
    log("SUCCESS_WRITTEN")
    rp("down", name, "-y", "--terminate", timeout=120)
    log("POD_TERMINATED", podid)


if __name__ == "__main__":
    main()
