# Runbook

All commands run from the repository root. `uv sync --extra dev` creates `.venv`; the examples use
`uv run`. Secrets go in `.env` (never committed); the host broker reads them, evaluated processes never
see them. Live stages require `budget.max_total_usd` (and `max_gpu_hours` for the GPU lane): with the
budget unresolved every live command stops at preflight (`unresolved_live_budget_policy: preflight_only`).

## 1. Offline lane (no keys, no GPU) — executed in the build session

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit tests/integration tests/scientific -m "not gpu and not live"
uv run npbench doctor --config configs/offline.yaml --offline
uv run npbench fixtures build --config configs/offline.yaml --out artifacts/fixtures
uv run npbench reference build --config configs/offline.yaml --data artifacts/fixtures
uv run npbench packets build --config configs/offline.yaml --data artifacts/fixtures
uv run npbench packets verify --config configs/offline.yaml
uv run npbench run --config configs/offline.yaml --out runs/offline            # add --limit N for a smoke
uv run npbench score --config configs/offline.yaml --runs runs/offline
uv run npbench report --config configs/offline.yaml --runs runs/offline --out reports/offline
```

Tier-zero offline (synthetic manifest + mock judges, exercises the judge pipeline):

```bash
uv run npbench corpus import --config configs/offline.yaml
uv run npbench corpus validate --config configs/offline.yaml
uv run npbench plan --config configs/offline.yaml --tier 0 --out artifacts/offline/tier0_plan.json
uv run npbench judge run --config configs/offline.yaml --out runs/offline_tier0 --plan artifacts/offline/tier0_plan.json
uv run npbench score --config configs/offline.yaml --runs runs/offline_tier0
uv run npbench report --config configs/offline.yaml --runs runs/offline_tier0 --out reports/offline_tier0
```

## 2. Tier zero on the real corpus — import/validate/plan executed; judge calls blocked on budget

Inputs: read-only clone of `timf34/AttractorStatePrefillAttack` and the HF dataset `timf34/dprobe-results`
(spiral transcripts + stories). Paths are set in `configs/tier0.local.yaml`; adjust them on another machine
(e.g. clone the repo next to this one and `huggingface-cli download timf34/dprobe-results --repo-type dataset
--include "spiral/*/extended/*" "stories/gemma3_27b/*" --local-dir ../hf_dprobe_results`).

```bash
uv run npbench corpus import --config configs/tier0.local.yaml
uv run npbench corpus validate --config configs/tier0.local.yaml          # freezes artifacts/corpus/tier0_panel.json
uv run npbench plan --config configs/tier0.local.yaml --out artifacts/tier0_plan.json
# set ANTHROPIC_API_KEY / OPENROUTER_API_KEY in .env, fill the non-Claude judge price, set budget.max_total_usd, then:
uv run npbench doctor --config configs/tier0.local.yaml --live             # re-resolves model ids and OpenRouter pricing
uv run npbench judge run --config configs/tier0.local.yaml --out runs/tier0 --plan artifacts/tier0_plan.json --smoke 3 --live
uv run npbench judge run --config configs/tier0.local.yaml --out runs/tier0 --plan artifacts/tier0_plan.json --live
uv run npbench score --config configs/tier0.local.yaml --runs runs/tier0
uv run npbench report --config configs/tier0.local.yaml --runs runs/tier0 --out reports/tier0
```

Preflight (200 items × 3 judges × 2 repetitions = 1,200 ratings): about 12 USD for the two Claude judges at
the current list prices; the OpenRouter judge price is filled by `doctor --live`.

## 3. Target lane (GPU) — scripted, not executed in the build session

Development on CPU with a tiny random model (exercises every command):

```bash
uv sync --extra dev --extra target --extra cpu
uv run pytest tests/target -m target
uv run npbench target generate --config configs/development_target_tiny.yaml
uv run npbench target collect  --config configs/development_target_tiny.yaml
uv run npbench target derive   --config configs/development_target_tiny.yaml
uv run npbench target intervene --config configs/development_target_tiny.yaml
```

Real target on RunPod via the user's `rp` CLI (`timf34/runpod-runner`): pin `target.revision` and set
`budget.max_gpu_hours` in `configs/gpu_smoke.yaml`, export `RUNPOD_API_KEY` and `HF_TOKEN`, then run
`scripts/pod/run_gpu_smoke.sh` (rp up → bootstrap → run → logs → down). On the pod:

```bash
uv sync --extra dev --extra target --extra cuda
uv run npbench doctor --config configs/gpu_smoke.yaml --gpu
uv run pytest tests/gpu -m gpu
uv run npbench target generate --config configs/gpu_smoke.yaml
uv run npbench target collect --config configs/gpu_smoke.yaml
uv run npbench target derive --config configs/gpu_smoke.yaml
uv run npbench target intervene --config configs/gpu_smoke.yaml
uv run npbench reference build --config configs/gpu_smoke.yaml
uv run npbench packets build --config configs/gpu_smoke.yaml
uv run npbench packets verify --config configs/gpu_smoke.yaml
```

Smoke bundles are marked `smoke_scope` and `exploratory` (mode registry provisional) and cannot feed the pilot.

## 4. Live development smoke (assistants + real auditor) — blocked on credentials and budget

```bash
uv run npbench doctor --config configs/development.yaml --live
uv run pytest tests/live -m live
uv run npbench plan --config configs/development.yaml --out artifacts/development_plan.json
uv run npbench run --config configs/development.yaml --out runs/development --limit 4 --live   # small declared smoke first
uv run npbench score --config configs/development.yaml --runs runs/development
uv run npbench report --config configs/development.yaml --runs runs/development --out reports/development
```

## 5. Pilot — blocked until mode definitions are confirmed and a budget exists

```bash
uv run npbench doctor --config configs/pilot.local.yaml --live --gpu
uv run npbench target generate/collect/derive/intervene --config configs/pilot.local.yaml
uv run npbench reference build --config configs/pilot.local.yaml
uv run npbench packets build --config configs/pilot.local.yaml && uv run npbench packets verify --config configs/pilot.local.yaml
uv run npbench freeze --config configs/pilot.local.yaml --out artifacts/pilot_lock.json
uv run npbench plan --config configs/pilot.local.yaml --lock artifacts/pilot_lock.json --out artifacts/pilot_plan.json
uv run npbench run --config configs/pilot.local.yaml --lock artifacts/pilot_lock.json --out runs/pilot --live
uv run npbench score --config configs/pilot.local.yaml --lock artifacts/pilot_lock.json --runs runs/pilot
uv run npbench report --config configs/pilot.local.yaml --lock artifacts/pilot_lock.json --runs runs/pilot --out reports/pilot
```

`freeze` refuses a pilot stage while the mode registry is provisional or any packet origin is not
`real_target`; `plan --lock` refuses if anything changed since the freeze.

## Isolation in production

The host-side sandbox (`runner/isolation.py`) gives a network namespace plus an unprivileged uid on a root
host. For the pilot, run the executor inside a container with `--network none` for the sandboxed step and
mount only the packet (read-only) and the work directory; keep the ledger and audits directory outside the
container. Record the achieved `isolation_level` (it is written to every run's status).
