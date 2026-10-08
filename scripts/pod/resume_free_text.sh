#!/usr/bin/env bash
# Resume ONLY the free-text stage and completion marker after the generation environment
# (/es_venv, EasySteer overlay, CUDA 12.9 compat) was already built by collection_worker.sh.
# Capture, derive and intervene outputs are untouched. Independent watchdog must be alive.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
test -s /workspace/npbench_watchdog_armed && kill -0 "$(cat /workspace/npbench_watchdog_armed)"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/hf_cache TOKENIZERS_PARALLELISM=false
unset HF_HUB_ENABLE_HF_TRANSFER
CONFIG=configs/matched_affect_collection.yaml
CAPTURE_PY=$(command -v python)
echo "source $(git rev-parse HEAD)"
export PATH=/es_venv/bin:$PATH
ninja --version
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if (( driver_major >= 570 && driver_major < 575 )); then
  export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
fi
/es_venv/bin/pip install --quiet -e .   # pick up the source fix in the generation venv
/es_venv/bin/python -c "import torch; assert torch.cuda.is_available() and torch.version.cuda=='12.9'"
echo "STAGE free-text $(date -u +%FT%TZ)"
/es_venv/bin/python -m npbench.target.free_text --config "$CONFIG"
"$CAPTURE_PY" - <<'PY'
import json
from pathlib import Path
from npbench.util import sha256_file
root=Path('artifacts/matched_affect_v1')
manifest={str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file() and p.name!='download_manifest.json'}
(root/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'COLLECTION_COMPLETE').write_text('Target capture, interventions and free text complete; ratings pending.\n')
PY
echo 'COLLECTION_COMPLETE'
