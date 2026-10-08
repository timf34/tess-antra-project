"""npbench command-line interface.

Every command validates its inputs, writes a machine-readable summary, and exits non-zero on failure.
Study paths resolve relative to the repository root (the directory containing pyproject.toml)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .config import StudyConfig, load_config
from .util import repo_root, utc_now_iso, write_json

app = typer.Typer(
    help="npbench: research-assistant evaluation on non-persona motivation research.", no_args_is_help=True
)
corpus_app = typer.Typer(help="Corpus intake (tier zero).", no_args_is_help=True)
judge_app = typer.Typer(help="Tier-zero judge panel.", no_args_is_help=True)
fixtures_app = typer.Typer(help="Synthetic fixtures (origin=synthetic_fixture).", no_args_is_help=True)
reference_app = typer.Typer(help="Evaluator-only reference calculations.", no_args_is_help=True)
packets_app = typer.Typer(help="Frozen research packets.", no_args_is_help=True)
target_app = typer.Typer(help="Open-weight target lane (GPU).", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(judge_app, name="judge")
app.add_typer(fixtures_app, name="fixtures")
app.add_typer(reference_app, name="reference")
app.add_typer(packets_app, name="packets")
app.add_typer(target_app, name="target")

ConfigOpt = typer.Option(..., "--config", "-c", help="Study configuration YAML.", exists=True, dir_okay=False)


def _cfg(path: Path) -> StudyConfig:
    try:
        return load_config(path)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"error: invalid config {path}: {e}", err=True)
        raise typer.Exit(2) from e


def _emit(summary: dict[str, Any], out: Path | None = None) -> None:
    text = json.dumps(summary, indent=2, default=str)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)


def _fail(msg: str, code: int = 1) -> None:
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code)


@app.callback()
def _main(version: bool = typer.Option(False, "--version", help="Print version and exit.")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


# ----------------------------------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------------------------------


@app.command()
def doctor(
    config: Path = ConfigOpt,
    offline: bool = typer.Option(False, "--offline", help="No network checks."),
    live: bool = typer.Option(
        False, "--live", help="List provider models and record credentials presence (names only)."
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Check torch/CUDA and the target extra."),
    out: Path | None = typer.Option(None, "--out", help="Write the doctor summary JSON here."),
) -> None:
    """Environment and configuration preflight."""
    cfg = _cfg(config)
    from .mode_registry import load_mode_registry

    checks: dict[str, Any] = {
        "kind": "doctor",
        "created_at": utc_now_iso(),
        "config": str(config),
        "config_hash": cfg.config_hash(),
        "python": sys.version.split()[0],
        "repo_root": str(repo_root()),
    }
    problems: list[str] = []
    try:
        reg = load_mode_registry(cfg.resolve(cfg.mode_definitions.registry_path))
        allowed, reason = reg.pilot_allowed()
        checks["mode_registry"] = {
            "version": reg.registry_version,
            "status": reg.status,
            "pilot_allowed": allowed,
            "reason": reason,
            "external_action_as_enactment": reg.external_action_used_as_enactment(),
        }
        if reg.external_action_used_as_enactment():
            problems.append("mode registry operationalizes enactment as an external action")
    except Exception as e:  # noqa: BLE001
        problems.append(f"mode registry: {e}")
    srcs = []
    for s in cfg.tier0.corpus_sources:
        p = cfg.resolve(s.path)
        srcs.append({"kind": s.kind, "path": str(p), "exists": p.exists()})
        if not p.exists():
            problems.append(f"corpus source missing: {p}")
    checks["tier0_sources"] = srcs
    checks["budget"] = {
        "max_total_usd": cfg.budget.max_total_usd,
        "policy": cfg.budget.unresolved_live_budget_policy,
        "live_budget_resolved": cfg.live_budget_resolved(),
    }
    cred_names = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "RUNPOD_API_KEY",
    ]
    checks["credentials_present"] = {n: bool(os.environ.get(n)) for n in cred_names}
    if live and not offline:
        from .providers import ProviderError, build_provider

        live_models: dict[str, Any] = {}
        for j in cfg.tier0.judges + [
            type(
                "J",
                (),
                {
                    "slot_id": m.slot,
                    "provider": m.provider,
                    "model_id": m.model_id,
                    "settings": {},
                    "base_url": None,
                    "api_key_env": None,
                },
            )()
            for m in cfg.providers.models
            if m.provider and m.provider != "fake"
        ]:
            if j.provider == "fake":
                continue
            try:
                prov = build_provider(
                    j.provider,
                    j.model_id,
                    getattr(j, "settings", {}),
                    base_url=getattr(j, "base_url", None),
                    api_key_env=getattr(j, "api_key_env", None),
                )
                models = prov.list_models()
                ids = [m.get("id") for m in models]
                entry = {
                    "provider": prov.name,
                    "n_models": len(models),
                    "configured_model_id": j.model_id,
                    "configured_model_listed": j.model_id in ids,
                    "sample_ids": ids[:40],
                }
                pr = next((m.get("pricing") for m in models if m.get("id") == j.model_id), None)
                if pr:
                    entry["pricing"] = pr
                live_models[j.slot_id] = entry
                if j.model_id not in ids:
                    problems.append(f"{j.slot_id}: configured model {j.model_id} not in provider model list")
            except ProviderError as e:
                live_models[j.slot_id] = {"error": str(e), "kind": e.kind}
                problems.append(f"{j.slot_id}: {e}")
            except Exception as e:  # noqa: BLE001
                live_models[j.slot_id] = {"error": repr(e)}
                problems.append(f"{j.slot_id}: {e!r}")
        checks["live_models"] = live_models
    if gpu:
        try:
            import torch

            checks["gpu"] = {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else [],
            }
            try:
                import transformers

                checks["gpu"]["transformers"] = transformers.__version__
            except ImportError:
                problems.append("transformers not installed (uv sync --extra target)")
            if not torch.cuda.is_available():
                problems.append("no CUDA device available")
        except ImportError:
            problems.append("torch not installed (uv sync --extra target --extra cpu|cuda)")
    checks["problems"] = problems
    checks["ok"] = not problems
    _emit(checks, out)
    if problems:
        raise typer.Exit(1)


# ----------------------------------------------------------------------------------------------
# tier zero: corpus / plan / judge / score / report
# ----------------------------------------------------------------------------------------------


def _origin_for(cfg: StudyConfig):
    return cfg.data.origin_required


@corpus_app.command("import")
def corpus_import(config: Path = ConfigOpt) -> None:
    """Import configured corpus sources into the provenance-preserving corpus JSONL."""
    cfg = _cfg(config)
    from .corpus import import_corpus

    s = import_corpus(cfg, origin=_origin_for(cfg))
    summary = {"kind": "corpus_import", **s.__dict__}
    _emit(summary, cfg.resolve(cfg.tier0.corpus_out).with_suffix(".import_summary.json"))
    if s.records == 0:
        _fail("no records imported")


@corpus_app.command("validate")
def corpus_validate(config: Path = ConfigOpt) -> None:
    """Validate the imported corpus and freeze the stratified panel manifest (hash-pinned)."""
    cfg = _cfg(config)
    from .corpus import sample_panel, validate_corpus, write_panel

    rep, records, cands = validate_corpus(cfg)
    summary: dict[str, Any] = {"kind": "corpus_validate", **rep.__dict__}
    if not rep.ok or not records:
        _emit(summary)
        _fail("corpus validation failed")
    panel = sample_panel(cfg, records, cands, origin=_origin_for(cfg))
    out = cfg.resolve(cfg.tier0.panel_out)
    write_panel(panel, out)
    summary["panel"] = {
        "path": str(out),
        "manifest_id": panel.manifest_id,
        "manifest_hash": panel.manifest_hash,
        "n_items": len(panel.items),
        "strata_counts": panel.strata_counts,
    }
    _emit(summary, out.with_suffix(".validate_summary.json"))


@app.command()
def plan(
    config: Path = ConfigOpt,
    out: Path = typer.Option(..., "--out", help="Plan JSON output path."),
    lock: Path | None = typer.Option(None, "--lock", help="Freeze lock JSON (tier two)."),
    tier: int | None = typer.Option(
        None, "--tier", help="0 = judge panel plan, 2 = assistant-run plan (default: from stage)."
    ),
) -> None:
    """Enumerate scheduled calls with stable keys and preflight cost/budget."""
    cfg = _cfg(config)
    if tier is None:
        tier = 0 if cfg.study.stage.startswith("tier0") else 2
    if tier == 0:
        from .corpus import load_panel
        from .tier0.plan import build_plan

        panel = load_panel(cfg.resolve(cfg.tier0.panel_out))
        p = build_plan(cfg, panel)
        write_json(cfg.resolve(out), p)
        _emit({k: v for k, v in p.items() if k != "requests"})
        return
    from .runner.schedule import build_tier2_plan

    p = build_tier2_plan(cfg, lock_path=cfg.resolve(lock) if lock else None)
    write_json(cfg.resolve(out), p)
    _emit({k: v for k, v in p.items() if k != "runs"})


@judge_app.command("run")
def judge_run(
    config: Path = ConfigOpt,
    out: Path = typer.Option(..., "--out", help="Run directory (ledger, judgments)."),
    plan_path: Path | None = typer.Option(
        None, "--plan", help="Plan JSON (default: <out>/plan.json or artifacts/tier0_plan.json)."
    ),
    smoke: int | None = typer.Option(None, "--smoke", help="Only the first N panel items, one repetition."),
    live: bool = typer.Option(False, "--live", help="Permit live provider calls (budget must be resolved)."),
) -> None:
    """Run the three-judge panel (resumable; fresh session per rating)."""
    cfg = _cfg(config)
    from .corpus import load_panel
    from .tier0.plan import build_plan
    from .tier0.run import run_judges

    panel = load_panel(cfg.resolve(cfg.tier0.panel_out))
    if plan_path and plan_path.exists():
        p = json.loads(plan_path.read_text())
    else:
        p = build_plan(cfg, panel)
        write_json(cfg.resolve(out) / "plan.json", p)
    if p.get("panel_manifest_hash") != panel.manifest_hash:
        _fail("plan does not match the frozen panel manifest (re-run plan)")
    if p.get("rubric_hash") != panel.rubric_hash:
        _fail("rubric hash changed since the panel was frozen")
    try:
        s = run_judges(cfg, panel, p, cfg.resolve(out), smoke_items=smoke, allow_live=live)
    except PermissionError as e:
        _fail(str(e), 3)
    _emit(s)


@app.command()
def score(
    config: Path = ConfigOpt,
    runs: Path = typer.Option(..., "--runs", help="Run directory."),
    lock: Path | None = typer.Option(None, "--lock"),
) -> None:
    """Score a run directory (tier zero: paired judge differences; tier two: evaluator)."""
    cfg = _cfg(config)
    runs = cfg.resolve(runs)
    if (runs / "judgments.jsonl").exists():  # tier-zero run directory
        from .corpus import load_panel
        from .tier0.score import load_judgments, score_tier0, write_scores

        panel = load_panel(cfg.resolve(cfg.tier0.panel_out))
        jp = runs / "judgments.jsonl"
        if not jp.exists():
            _fail(f"no judgments at {jp}")
        js = load_judgments(jp)
        sc = score_tier0(cfg, panel, js)
        paths = write_scores(sc, panel, runs / "scores")
        _emit(
            {
                "kind": "tier0_score_summary",
                "n_items": sc["n_items"],
                "n_judgments": sc["n_judgments"],
                "n_flagged": sc["n_flagged"],
                "origin": sc["origin"],
                "any_mock": sc["any_mock"],
                **paths,
            }
        )
        return
    from .scoring.evaluate import score_runs

    summary = score_runs(cfg, runs, lock_path=cfg.resolve(lock) if lock else None)
    _emit(summary)
    if summary.get("errors"):
        raise typer.Exit(1)


@app.command()
def report(
    config: Path = ConfigOpt,
    runs: Path = typer.Option(..., "--runs"),
    out: Path = typer.Option(..., "--out"),
    lock: Path | None = typer.Option(None, "--lock"),
) -> None:
    """Render the report for a scored run directory."""
    cfg = _cfg(config)
    runs = cfg.resolve(runs)
    out = cfg.resolve(out)
    if (runs / "judgments.jsonl").exists():  # tier-zero run directory
        from .corpus import load_panel
        from .tier0.report import render_report, write_report

        panel = load_panel(cfg.resolve(cfg.tier0.panel_out))
        sp = runs / "scores" / "scores.json"
        if not sp.exists():
            _fail(f"no scores at {sp}; run `npbench score` first")
        scores = json.loads(sp.read_text())
        rs = runs / "run_summary.json"
        run_summary = json.loads(rs.read_text()) if rs.exists() else None
        p = write_report(render_report(scores, panel, run_summary), out)
        _emit({"kind": "tier0_report", "path": p, "origin": scores["origin"], "any_mock": scores["any_mock"]})
        return
    from .analysis.report import write_tier2_report

    p = write_tier2_report(cfg, runs, out)
    _emit({"kind": "tier2_report", "path": str(p)})


# ----------------------------------------------------------------------------------------------
# fixtures / reference / packets / run / freeze / target
# ----------------------------------------------------------------------------------------------


@fixtures_app.command("build")
def fixtures_build(
    config: Path = ConfigOpt, out: Path = typer.Option(Path("artifacts/fixtures"), "--out")
) -> None:
    """Build synthetic fixtures: tier-zero manifest and tier-two bundles with known effects."""
    cfg = _cfg(config)
    from .fixtures import build_tier0_synthetic_manifest

    out = cfg.resolve(out)
    m = build_tier0_synthetic_manifest(out / "tier0_synthetic_manifest.jsonl", seed=cfg.study.seed)
    summary: dict[str, Any] = {"kind": "fixtures_build", "tier0_manifest": str(m)}
    try:
        from .fixtures_tier2 import build_tier2_fixtures

        summary["tier2"] = build_tier2_fixtures(cfg, out)
    except ImportError:
        summary["tier2"] = "not available in this build"
    _emit(summary, out / "fixtures_summary.json")


@reference_app.command("build")
def reference_build(config: Path = ConfigOpt, data: Path | None = typer.Option(None, "--data")) -> None:
    """Compute evaluator-only reference results for every bundle in the data directory."""
    cfg = _cfg(config)
    from .reference.build import build_reference

    _emit(build_reference(cfg, cfg.resolve(data) if data else None))


@packets_app.command("build")
def packets_build(config: Path = ConfigOpt, data: Path | None = typer.Option(None, "--data")) -> None:
    """Build frozen anonymized/revealed packet pairs from bundle data."""
    cfg = _cfg(config)
    from .packets.build import build_packets

    _emit(build_packets(cfg, cfg.resolve(data) if data else None))


@packets_app.command("verify")
def packets_verify(config: Path = ConfigOpt) -> None:
    """Verify manifests, byte-identical data within framing pairs, blinding, and no mounted oracle."""
    cfg = _cfg(config)
    from .packets.verify import verify_packets

    s = verify_packets(cfg)
    _emit(s)
    if not s.get("ok"):
        raise typer.Exit(1)


@app.command()
def run(
    config: Path = ConfigOpt,
    out: Path = typer.Option(..., "--out"),
    lock: Path | None = typer.Option(None, "--lock"),
    plan_path: Path | None = typer.Option(None, "--plan"),
    limit: int | None = typer.Option(None, "--limit", help="Run at most N scheduled runs (smoke)."),
    live: bool = typer.Option(False, "--live"),
) -> None:
    """Execute scheduled assistant runs (tier one/two) with isolation, audits, and the host ledger."""
    cfg = _cfg(config)
    from .runner.execute import execute_runs

    try:
        s = execute_runs(
            cfg,
            cfg.resolve(out),
            lock_path=cfg.resolve(lock) if lock else None,
            plan_path=cfg.resolve(plan_path) if plan_path else None,
            limit=limit,
            allow_live=live,
        )
    except PermissionError as e:
        _fail(str(e), 3)
    _emit(s)


@app.command()
def freeze(config: Path = ConfigOpt, out: Path = typer.Option(..., "--out")) -> None:
    """Freeze data hashes, gold results, prompts, scoring, model configs, pricing and the schedule."""
    cfg = _cfg(config)
    from .runner.schedule import freeze_study

    try:
        s = freeze_study(cfg, cfg.resolve(out))
    except PermissionError as e:
        _fail(str(e), 3)
    _emit(s)


@target_app.command("generate")
def target_generate(config: Path = ConfigOpt) -> None:
    """Render registered stimuli (prompts per protocol) for the target lane."""
    cfg = _cfg(config)
    from .target.pipeline import target_generate as _g

    _emit(_g(cfg))


@target_app.command("collect")
def target_collect(config: Path = ConfigOpt) -> None:
    """Collect activations at the registered site (requires the target extra and weights)."""
    cfg = _cfg(config)
    from .target.pipeline import target_collect as _c

    _emit(_c(cfg))


@target_app.command("derive")
def target_derive(config: Path = ConfigOpt) -> None:
    """Derive contrast vectors, controls and projection scales from construction data."""
    cfg = _cfg(config)
    from .target.pipeline import target_derive as _d

    _emit(_d(cfg))


@target_app.command("intervene")
def target_intervene(config: Path = ConfigOpt) -> None:
    """Run the registered single-position interventions and readouts."""
    cfg = _cfg(config)
    from .target.pipeline import target_intervene as _i

    _emit(_i(cfg))


if __name__ == "__main__":
    app()
