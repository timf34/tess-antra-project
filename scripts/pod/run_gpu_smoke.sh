#!/usr/bin/env bash
# GPU smoke of the npbench open-weight target lane. Runs ON the pod, launched by `rp run`.
#
# Laptop side (timf34/runpod-runner convention: rp up -> rp bootstrap -> rp run -> rp logs -> rp down).
# The whole pipeline is CPU-tested offline first (pytest -m target); never debug on a live pod.
#
#   export RUNPOD_API_KEY=...                        # or ~/.config/runpod-runner/config.toml
#   rp up --name npbench-smoke --gpu h100 --volume none --disk 40 --volume-size 160
#   rp bootstrap npbench-smoke --repo https://github.com/timf34/tess-antra-project.git --env ~/secrets/npbench.env
#   rp run npbench-smoke --job smoke --dotenv -- bash scripts/pod/run_gpu_smoke.sh
#   rp logs npbench-smoke --job smoke -n 50          # poll; the log ends with EXIT=<code>
#   rp down npbench-smoke                            # stop billing; results are already on the HF dataset
#
# .env (installed by `rp bootstrap --env`, exported by `rp run --dotenv`) must provide
#   HF_TOKEN        Hub token of an account that accepted the Gemma licence (weights + results upload)
#   RUNPOD_API_KEY  only needed for SHUTDOWN=stop|terminate (self-stop through the REST API)
# Before launching: set budget.max_gpu_hours and target.revision in configs/gpu_smoke.yaml.
#
# Knobs: CONFIG (configs/gpu_smoke.yaml)  HF_RESULTS_REPO (timf34/npbench-results)  HF_PRIVATE (1)
#        SHUTDOWN=stop|terminate (unset = leave the pod running)  MAX_MINUTES (180, hard-stop backstop)
#        NPBENCH_HF_HOME (/hf_cache: local container disk; rp run presets /workspace/hf, a FUSE mount
#        with a 50 GB quota when no network volume is attached, too small for 27B bf16 weights)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
export HF_HOME=${NPBENCH_HF_HOME:-/hf_cache}
CONFIG=${CONFIG:-configs/gpu_smoke.yaml}
HF_RESULTS_REPO=${HF_RESULTS_REPO:-timf34/npbench-results}
HF_PREFIX=${HF_PREFIX:-gpu_smoke}
COLLECTION_ONLY=${COLLECTION_ONLY:-0}
HF_PRIVATE=${HF_PRIVATE:-1}            # bundles hold evaluator data (never mounted for assistants): private by default
MAX_MINUTES=${MAX_MINUTES:-180}
STARTED=$(date -u +%s)
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)

step() { echo; echo "== $* ($(( ($(date -u +%s) - STARTED) / 60 )) min elapsed) =="; }
die()  { echo; echo "!!! gpu smoke FAILED: $1"; exit "${2:-1}"; }

self_stop() {  # SHUTDOWN=stop|terminate: stop billing through the RunPod REST API (same as the user's self_stop.sh)
  [ -n "${SHUTDOWN:-}" ] || return 0
  local pod_id="${RUNPOD_POD_ID:-}"
  if [ -z "$pod_id" ] && [ -r /proc/1/environ ]; then
    pod_id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)
  fi
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "$pod_id" ]; then
    echo "!! self-stop skipped: RUNPOD_API_KEY or RUNPOD_POD_ID missing -- run: rp down npbench-smoke"; return 0
  fi
  if [ "$SHUTDOWN" = "terminate" ]; then
    echo "== SHUTDOWN=terminate -> DELETE pod $pod_id =="
    curl -s -X DELETE "https://rest.runpod.io/v1/pods/$pod_id" -H "Authorization: Bearer $RUNPOD_API_KEY"
  else
    echo "== SHUTDOWN=stop -> stop pod $pod_id =="
    curl -s -X POST "https://rest.runpod.io/v1/pods/$pod_id/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
  fi
  echo
}
# An idle pod is worse than a stopped one: stop on failure too (the log says what failed), and arm a
# hard-stop timer as a backstop against hangs. `terminate` is downgraded to `stop` on failure so the
# container disk (logs, artifacts) survives for inspection.
on_exit() {
  local code=$?
  [ -n "${WATCHDOG_PID:-}" ] && kill "$WATCHDOG_PID" 2>/dev/null || true
  if [ "$code" -ne 0 ] && [ "${SHUTDOWN:-}" = "terminate" ]; then SHUTDOWN=stop; fi
  echo "== finished with exit $code after $(( ($(date -u +%s) - STARTED) / 60 )) min =="
  self_stop
}
trap on_exit EXIT
if [ -n "${SHUTDOWN:-}" ]; then
  ( sleep "$(( MAX_MINUTES * 60 ))"; echo "!! hard stop after $MAX_MINUTES min"; self_stop ) &
  WATCHDOG_PID=$!
fi

step "0/8 preconditions"
[ -f "$CONFIG" ] || die "config $CONFIG not found" 2
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set (rp bootstrap --env ... and rp run --dotenv)" 2
if grep -Eq '^\s*max_gpu_hours:\s*null' "$CONFIG"; then
  die "budget.max_gpu_hours is null in $CONFIG; set the authorized GPU-hour cap before renting a pod" 2
fi
MODEL_ID=$(sed -nE 's/^\s*model_id:\s*"?([^"#[:space:]]+)"?.*/\1/p' "$CONFIG" | head -1)
REVISION=$(sed -nE 's/^\s*revision:\s*"?([^"#[:space:]]+)"?.*/\1/p' "$CONFIG" | head -1)
if [ -z "$REVISION" ] || [ "$REVISION" = "null" ]; then
  echo "!! target.revision is unpinned in $CONFIG (allowed for the smoke stage; every bundle records 'unpinned')"
  REVISION=""
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "no GPU visible" 3
mkdir -p "$HF_HOME"
df -h "$HF_HOME" | tail -1

step "1/8 uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# Weight download starts now, in parallel with the install (the biggest idle block on a volume-less pod).
step "2/8 weights: $MODEL_ID ${REVISION:+@$REVISION} -> $HF_HOME (background)"
export HF_HUB_ENABLE_HF_TRANSFER=1
uvx --from 'huggingface_hub[hf_transfer]' hf download "$MODEL_ID" ${REVISION:+--revision "$REVISION"} \
  --include '*.json' '*.safetensors' '*.model' '*.txt' '*.jinja' > /workspace/weights.log 2>&1 &
DL_PID=$!

step "3/8 uv sync --extra dev --extra target --extra cuda"
uv sync --locked --extra dev --extra target --extra cuda || die "uv sync failed" 5

step "4/8 npbench doctor --gpu"
uv run --no-sync npbench doctor --config "$CONFIG" --gpu --out "artifacts/doctor_gpu_smoke_$RUN_TAG.json" || die "doctor reported problems" 6

step "5/8 pytest tests/gpu -m gpu"
if [ -d tests/gpu ]; then
  NPBENCH_GPU_CONFIG="$CONFIG" NPBENCH_REQUIRE_GPU=1 uv run --no-sync pytest tests/gpu -m gpu || die "GPU tests failed" 7
else
  echo "!! tests/gpu does not exist in this checkout; skipping (the target-lane CPU suite ran before launch)"
fi

step "waiting for the weight download"
if ! wait "$DL_PID"; then
  tail -20 /workspace/weights.log
  die "weight download failed (licence accepted for this HF_TOKEN? revision exists?)" 8
fi
tail -3 /workspace/weights.log

step "6/8 target lane: generate -> collect -> derive -> intervene"
uv run --no-sync npbench target generate  --config "$CONFIG" || die "target generate failed" 10
uv run --no-sync npbench target collect   --config "$CONFIG" || die "target collect failed" 11
uv run --no-sync npbench target derive    --config "$CONFIG" || die "target derive failed" 12
uv run --no-sync npbench target intervene --config "$CONFIG" || die "target intervene failed" 13

step "7/8 reference build, packets build/verify"
if [ "$COLLECTION_ONLY" = "1" ]; then
  echo "Collection only: ratings remain pending; no research-assistant packets are released."
else
uv run --no-sync npbench reference build --config "$CONFIG" || die "reference build failed" 14
uv run --no-sync npbench packets build   --config "$CONFIG" || die "packets build failed" 15
uv run --no-sync npbench packets verify  --config "$CONFIG" || die "packets verify failed" 16

fi

step "8/8 sync artifacts to the HF dataset $HF_RESULTS_REPO (private=$HF_PRIVATE)"
VERSION=$(uv run --no-sync python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["study"]["version"])' "$CONFIG")
ARTIFACTS="artifacts/${VERSION:-gpu_smoke_v1}"
[ -d "$ARTIFACTS" ] || die "no artifacts directory $ARTIFACTS to upload" 17
uv run --no-sync python - "$ARTIFACTS" "$HF_RESULTS_REPO" "$RUN_TAG" "$HF_PRIVATE" "$HF_PREFIX" <<'PY' || die "HF upload failed; the pod is kept so nothing is lost (rp scp npbench-smoke pod:/workspace/tess-antra-project/artifacts ./artifacts -r)" 18
import os, sys
from huggingface_hub import HfApi

folder, repo, tag, private = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
prefix = sys.argv[5]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
if private and not api.repo_info(repo, repo_type="dataset").private:
    raise RuntimeError("Refusing evaluator-data upload to an existing public dataset")
api.upload_folder(
    folder_path=folder, repo_id=repo, repo_type="dataset", path_in_repo=f"{prefix}/{tag}",
    ignore_patterns=["*.tmp", "*.tmp.npz"], commit_message=f"gpu smoke {tag}",
)
print(f"uploaded {folder} -> {repo}/{prefix}/{tag}")
PY

echo
echo "== GPU SMOKE OK: artifacts in $ARTIFACTS and on $HF_RESULTS_REPO/$HF_PREFIX/$RUN_TAG =="
echo "   pod age $(( ($(date -u +%s) - STARTED) / 60 )) min; now: rp down npbench-smoke (unless SHUTDOWN was set)"
