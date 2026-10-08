# Matched affect collection: RunPod handoff

## Objective and status

Collect one real Gemma 3 activation/intervention bundle for an exploratory research-assistant
framing experiment. Tim explicitly accepted provisional mode definitions for exploration.
Do not call this an Antra replication or change the confirmed-study gate.

Prepared config: `configs/matched_affect_collection.yaml`.
Pinned model: `google/gemma-3-27b-it`, revision
`005ad3404e59d6023443cb575daa05336842228a` (tokenizer pinned identically).
Stimuli: `configs/stimuli/matched_affect_v1.yaml`, 12 hand-authored distress/aversion contexts,
3 modes × 3 polarities = 108 prompts. Context splits are 6 construction / 2 validation /
4 test; no context crosses a split. These prompts are adapted unchanged from the existing
engineering spec. Their scientific adequacy remains exploratory; inspect renderings before launch.
Roleplay uses the chat template, simulation raw continuation, and enactment own-voice chat.
Transport, token position semantics and speaker/content differences remain possible confounds.

The future researcher experiment is 18 runs: 2 framings × 3 models × 3 repetitions,
with two real independent review sessions per run. This collection DOES NOT launch those runs.

## Before spending

1. Read the user's reply about the GPU budget. It was requested separately; config caps remain
   null until supplied. Resolve a current offered GPU price and include download/setup time,
   disk costs and a shutdown margin. Do not treat `max_gpu_hours` as an enforced billing cap.
2. The RunPod key and HF token in the repository `.env` were verified read-only. Never print them.
   Two existing dprobe pods were stopped when checked. Do not reuse/restart/delete unrelated pods.
3. Prefer one H100 80GB (or an appropriately priced larger GPU), at least 120GB container disk
   for `/hf_cache` weights/dependencies, and a 40GB persistent `/workspace` pod volume for code/results.
   Check actual space and memory rather than assuming those sizes guarantee success.
4. Use `/Users/timf34/Documents/VSCode/runpod-runner` and its `.venv/bin/rp`
   (or `.venv/bin/python -m rp` from that directory). `rp` is not currently on PATH.
5. Set `budget.max_gpu_hours` from the agreed total limit, then package **local** source:

```sh
uv run --extra dev --extra target --extra cpu --locked pytest tests/target -m target
uv run python scripts/local/package_collection.py
```

This writes `artifacts/collection_source.tar.gz` and a SHA-256 file manifest. It includes
uncommitted source fixes and excludes `.env`, environments, runs and generated artifacts.
Do not simply clone the default GitHub branch: it does not contain these changes.

## Provision and transfer

Provision a fresh pod named `npbench-matched-v1`. Start a LAPTOP-SIDE deadline supervisor at
creation time, covering provisioning, SSH waits, bootstrap, collection and upload. It must
stop this pod on timeout/error/disconnection and verify `EXITED` through the API. The on-pod
watchdog below is a second backstop, not coverage for bootstrap failures. Do not provision
until this supervisor is in place and the current price fits the cap.

Use the existing rp up/bootstrap/scp/run workflow, then overwrite the checkout with the
prepared source archive BEFORE executing anything from it. Keep code under
`/workspace/tess-antra-project` and record the archive hash in the job log. Transfer a
chmod-600 minimal environment containing only HF_TOKEN and RUNPOD_API_KEY, never the full
local `.env` or any OpenRouter/GitHub key. RUNPOD_POD_ID must be available to the job.

The CUDA extra is pinned to torch 2.8.* for CUDA 12.8 hosts. Use the updated lockfile.
Check the host driver before dependency install. Do not silently upgrade torch.
Do not install dependencies into the image's system Python.

On the pod, launch through rp with `--dotenv`, set MAX_MINUTES to the REMAINING time under
the approved GPU cap, and execute:

```sh
bash scripts/pod/run_matched_collection.sh
```

The entry point requires shutdown credentials/pod identity and a positive deadline. It uses
`SHUTDOWN=stop`, never automatic termination, to preserve persistent workspace outputs.
Stopping ends GPU compute billing but persistent storage can still accrue charges.

## Outputs and checks

- Collect -> derive -> intervene; raw provenance/renderings/splits retained.
- All arrays finite where expected; failed rows and elicitation failures retained.
- Verify the raw contrast identity, signs, split integrity, projection scales and random controls.
- Resume must retain checkpoint/config/vector hashes and not mix incompatible partial runs.
- Download results to the laptop and verify hashes. The script uploads the final folder to
  private `timf34/npbench-results` under `matched_affect_v1/<timestamp>`.
- On failure, recover partial files/logs from persistent `/workspace`; do not restart automatically.
- Verify the pod is stopped; report GPU runtime, price, status and artifact locations.

## Important: what is NOT complete after collection

The current target lane scores fixed response alternatives; it does NOT sample free-text
continuations. `rating_scale1` remains null. Never invent ratings, pass alternative text off
as a generated continuation, or mark a pending required readout genuinely not applicable.
`COLLECTION_ONLY=1` deliberately prevents creating research-assistant packets at this stage.

Before the 18 researcher runs:
- Implement matched free-continuation generation under the same single-position intervention
  convention, with seeds/prefix/token provenance, then a blinded rating pass with stored responses.
- Confirm the intended elicitation modes from samples; retain ambiguous/failed cases.
- Finish agent-loop cost reservations, response retention, reviewers and resume identity checks
  (current durable budget changes cover tier-zero judges only).
- Make researcher prompts identical except the interpretation paragraph; replace the current
  inaccurate 'source-defined' wording. Preserve all methodological information in both arms.
- Rebuild independent references/packets, verify framing equivalence and known-failure detection.
- Run six live shakedown tasks on the Mac sandbox; then freeze a separate 18-run exploratory batch.

No numerical effect or framing difference is claimed by this setup.
