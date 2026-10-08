"""Study configuration: typed loading of the YAML contract (schema_version 2) plus tier-zero settings.

All paths resolve relative to the repository root. Secrets never live here (see .env.example)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas import PRODUCTION_ORIGINS, Origin
from .util import repo_root, sha256_obj


class Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class StudySection(Lenient):
    stage: str = "tier0_judges"
    version: str = "dev"
    seed: int = 20261008


class ModeDefinitionsSection(Lenient):
    status: str = "pending_source_confirmation"
    registry_path: str = "docs/mode_definitions.yaml"
    tier0_requires_confirmation: bool = False
    main_mode_pilot_requires: str = "source_confirmation_or_user_accepted_named_operationalization"
    user_accepted_named_operationalization: str | None = None


class TargetSection(Lenient):
    model_id: str | None = None
    backend: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    hook_site: str = "decoder_block_output"
    capture_position: str = "registered_final_nonpadding_prefix_token"
    primary_layer_rule: str = "floor_num_blocks_div_2"
    layer_indexing: str = "zero_based"
    dtype: str | None = None
    use_cache: bool = False
    smoke_scope: bool = False


class DataSection(Lenient):
    origin_required: Origin = Origin.real_target
    bundles: int = 4
    corpus_manifest: str | None = None
    candidate_ids: list[str] = Field(default_factory=lambda: ["information_seeking", "distress_aversion"])
    mode_ids: list[str] = Field(default_factory=lambda: ["roleplay", "simulation", "enactment"])
    polarity_ids: list[str] = Field(default_factory=lambda: ["positive", "neutral", "negative"])
    groups_per_candidate_bundle: dict[str, int] = Field(
        default_factory=lambda: {"construction": 24, "validation": 8, "test": 16}
    )
    retain_failed_and_ambiguous_elicitation: bool = True


class InterventionsSection(Lenient):
    alphas: list[float] = Field(default_factory=lambda: [-2, -1, 0, 1, 2])
    strength_units: str = "construction_projection_standard_deviation"
    random_directions_per_contrast: int = 8
    position_schedule: str = "single_position_first_continuation_token"


class ProviderModel(Lenient):
    slot: str
    provider: str | None = None
    model_id: str | None = None
    effective_reasoning_settings: dict[str, Any] | None = None


class AuditorSection(Lenient):
    provider: str | None = None
    model_id: str | None = None
    fresh_session_per_review: bool = True
    kinds: list[str] = Field(default_factory=lambda: ["numerical", "methods_and_reporting"])
    require_host_verified_receipts: bool = True


class ProvidersSection(Lenient):
    models: list[ProviderModel] = Field(default_factory=list)
    auditor: AuditorSection = Field(default_factory=AuditorSection)


class EvaluationSection(Lenient):
    methods: list[str] = Field(default_factory=lambda: ["behavioral_continuations", "activation_contrasts"])
    task_types: list[str] = Field(default_factory=lambda: ["core_mode_transfer", "persona_control"])
    framings: list[str] = Field(default_factory=lambda: ["anonymized", "revealed"])
    repetitions: int = 2
    assistant_slots: list[str] = Field(default_factory=lambda: ["claude_a", "claude_b", "non_claude"])
    expected_runs: int | None = None
    expected_auditor_sessions: int | None = None
    post_submission_awareness_question: bool = True
    awareness_calls_budgeted_separately: bool = True
    bundle_ids: list[str] = Field(default_factory=list)


class RunnerSection(Lenient):
    concurrency: int = 2
    max_tool_calls_per_run: int = 60
    max_wall_seconds_per_run: int = 1800
    max_output_tokens_per_request: int = 8192
    max_total_billable_tokens_per_run: int = 120000
    max_retries_per_request: int = 2
    max_infrastructure_reruns_per_slot: int = 1
    mount_full_repository: bool = False
    mount_evaluator_or_gold: bool = False
    append_only_host_ledger: bool = True
    resume_completed_runs: bool = True
    provider: str = "fake"  # "fake" | "anthropic" | "openai_compatible"


class BudgetSection(Lenient):
    max_total_usd: float | None = None
    max_per_run_usd: float | None = None
    max_gpu_hours: float | None = None
    pricing_snapshot_path: str | None = None
    use_existing_authorized_limits: bool = True
    reserve_inflight_and_review_cost: bool = True
    unresolved_live_budget_policy: Literal["preflight_only", "allow"] = "preflight_only"


class JudgeConfig(Lenient):
    slot_id: str
    family: Literal["claude", "non_claude"]
    provider: Literal["fake", "anthropic", "openai_compatible"]
    model_id: str
    base_url: str | None = None  # for openai_compatible (e.g. OpenRouter)
    api_key_env: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    price_usd_per_m_input: float | None = None
    price_usd_per_m_output: float | None = None


class CorpusSource(Lenient):
    kind: Literal[
        "attractor_prefill_episodes",
        "dprobe_spirals",
        "dprobe_stories",
        "jsonl_manifest",
        "attractorbench_runs",
    ]
    path: str
    label: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class Tier0Section(Lenient):
    corpus_sources: list[CorpusSource] = Field(default_factory=list)
    corpus_out: str = "artifacts/corpus/tier0_corpus.jsonl"
    panel_out: str = "artifacts/corpus/tier0_panel.json"
    max_items_per_category: int = 40
    max_items: int = 200
    excerpt_max_chars: int = 1500
    context_turns: int = 1
    judges: list[JudgeConfig] = Field(default_factory=list)
    repetitions: int = 2
    session_mode: Literal["fresh", "batched"] = "fresh"
    rubric_id: str = "tier0_affect_expression_v1"
    mask_source_models: bool = True
    smoke_items: int = 3
    estimated_tokens_per_rating: dict[str, int] = Field(
        default_factory=lambda: {"input": 1800, "output": 400}
    )


class StudyConfig(Lenient):
    schema_version: int = 2
    status: str = "resolved_local"
    config_path: str | None = None
    study: StudySection = Field(default_factory=StudySection)
    mode_definitions: ModeDefinitionsSection = Field(default_factory=ModeDefinitionsSection)
    target: TargetSection = Field(default_factory=TargetSection)
    data: DataSection = Field(default_factory=DataSection)
    interventions: InterventionsSection = Field(default_factory=InterventionsSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    providers: ProvidersSection = Field(default_factory=ProvidersSection)
    runner: RunnerSection = Field(default_factory=RunnerSection)
    budget: BudgetSection = Field(default_factory=BudgetSection)
    tiers: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)
    tier0: Tier0Section = Field(default_factory=Tier0Section)
    paths: dict[str, str] = Field(default_factory=dict)

    # -- helpers ---------------------------------------------------------------------------------
    def root(self) -> Path:
        return repo_root()

    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root() / p

    def config_hash(self) -> str:
        data = self.model_dump(mode="json")
        data.pop("config_path", None)
        return sha256_obj(data)

    def is_production(self) -> bool:
        return self.study.stage.startswith("pilot") or self.study.stage == "tier2_agentic"

    def live_budget_resolved(self) -> bool:
        return self.budget.max_total_usd is not None and self.budget.max_total_usd > 0


def load_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = StudyConfig.model_validate(data)
    cfg.config_path = str(path)
    return cfg


def assert_production_origin(cfg: StudyConfig, origin: Origin) -> None:
    """Production pilot commands reject synthetic fixtures and development mocks."""
    if cfg.is_production() and origin not in PRODUCTION_ORIGINS:
        raise PermissionError(
            f"stage {cfg.study.stage!r} requires origin in {sorted(o.value for o in PRODUCTION_ORIGINS)}, got {origin.value!r}"
        )
