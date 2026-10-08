#!/usr/bin/env bash
# ON-POD entry point. Laptop provisioning/bootstrap needs its own bounded supervisor.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required for the shutdown watchdog}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID must identify this pod}"
: "${MAX_MINUTES:?Set this from the approved remaining GPU time budget}"
[[ "$MAX_MINUTES" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_MINUTES must be a positive integer'; exit 2; }
export CONFIG=configs/matched_affect_collection.yaml
export SHUTDOWN=stop
export HF_PRIVATE=1
export HF_PREFIX=matched_affect_v1
export COLLECTION_ONLY=1
exec bash scripts/pod/run_gpu_smoke.sh
