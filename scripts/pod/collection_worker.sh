#!/usr/bin/env bash
# Executed by rp run in the bootstrap capture venv. Independent watchdog must already be armed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/hf_cache TOKENIZERS_PARALLELISM=false
unset HF_HUB_ENABLE_HF_TRANSFER
CONFIG=configs/matched_affect_collection.yaml
CAPTURE_PY=$(command -v python)
mkdir -p artifacts/matched_affect_v1
python - <<'PY'
import json,subprocess,re
from pathlib import Path
from npbench.util import sha256_file,sha256_obj
lock=json.loads(Path('configs/matched_affect_design.lock.json').read_text())
assert sha256_obj(lock['design'])==lock['sha256']
assert sha256_file('configs/matched_affect_collection.yaml')==lock['design']['config_sha256']
assert sha256_file('configs/stimuli/matched_affect_v1.yaml')==lock['design']['stimulus_sha256']
text=subprocess.check_output(['nvidia-smi'],text=True)
v=re.search(r'CUDA Version: (\d+)\.(\d+)',text)
assert v and tuple(map(int,v.groups()))>=(12,8),'Driver CUDA must be >=12.8'
print('source',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'design',lock['sha256'])
PY
python - <<'PY'
import os,yaml
from huggingface_hub import snapshot_download
c=yaml.safe_load(open('configs/matched_affect_collection.yaml'))
snapshot_download(c['target']['model_id'],revision=c['target']['revision'],token=os.environ['HF_TOKEN'],
    allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'])
PY
export NPBENCH_GPU_CONFIG="$CONFIG" NPBENCH_REQUIRE_GPU=1
python -m pytest tests/gpu -m gpu
for step in generate collect derive intervene; do
  echo "STAGE $step $(date -u +%FT%TZ)"
  python -m npbench.cli target "$step" --config "$CONFIG"
done
# Generation uses an explicit CUDA wheel; a torch index alone does not select vLLM's CUDA build.
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
python3 -m venv /es_venv
/es_venv/bin/pip install --quiet --upgrade pip
/es_venv/bin/pip install --quiet -r scripts/pod/requirements-generation.txt
# H200 / driver 570 supports NVIDIA's CUDA 12.9 forward-compatibility package,
# including PTX JIT. Apply only to generation, leaving capture on its CUDA 12.8 runtime.
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if (( driver_major >= 570 && driver_major < 575 )); then
  curl -fsSL --retry 3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb -o /tmp/npbench-cuda-compat.deb
  dpkg -i /tmp/npbench-cuda-compat.deb
  export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
fi
if [ ! -d /es_source/.git ]; then
  git clone --quiet https://github.com/ZJU-REAL/EasySteer-vllm-v1.git /es_source
fi
git -C /es_source checkout --quiet 6267ca0cfc9c6e93b1427d36b1d655821d6d6f9b
VLLM_DIR=$(/es_venv/bin/python -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')
cp -a /es_source/vllm/. "$VLLM_DIR/"
/es_venv/bin/pip install --quiet -e .
/es_venv/bin/python - <<'PY'
import torch
from vllm import LLM, SamplingParams  # Exercise media/dependency imports before loading weights.
from vllm.steer_vectors import ApplySpec
assert torch.cuda.is_available()
assert torch.version.cuda == '12.9'
x = torch.ones(16, device='cuda')
assert (x @ x).item() == 16
torch.cuda.synchronize()
assert ApplySpec(prompt_positions=[-1]).generation is None
print('generation runtime',torch.__version__,torch.version.cuda)
PY
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
