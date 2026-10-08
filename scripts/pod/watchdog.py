"""Pod-independent-of-laptop wall-clock stop. Uses stdlib only; armed before bootstrap."""

import json
import os
import time
import urllib.request
from pathlib import Path

state = json.loads(Path("/workspace/npbench_watchdog.json").read_text())
key = os.environ["RUNPOD_API_KEY"]
Path("/workspace/npbench_watchdog_armed").write_text(str(os.getpid()))
while time.time() < state["deadline_epoch"]:
    time.sleep(min(15, max(0, state["deadline_epoch"] - time.time())))
for _attempt in range(20):
    try:
        req = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{state['pod_id']}/stop",
            method="POST",
            headers={"Authorization": "Bearer " + key, "User-Agent": "npbench-watchdog/1"},
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            print("Deadline reached; stop requested", response.status, flush=True)
        break
    except Exception as error:
        print("Stop request failed:", type(error).__name__, flush=True)
        time.sleep(15)
