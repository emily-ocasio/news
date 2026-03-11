"""
Controller [K]: Orphan adjudication replacement flow with deterministic staging,
LLM-assisted anchor extraction/ranking, and idempotent caching.
"""
# pylint: disable=too-many-lines,too-few-public-methods,too-many-instance-attributes,too-many-return-statements,too-many-locals,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, RootModel

from article import Article
from menuprompts import NextStep
from pymonad import (SQL, DbBackend, EnvKey, ErrorPayload, InputPrompt, Left,
                     Namespace, PromptKey, Right, Run, SQLParams, String, ask,
                     input_with_prompt, local, pure, put_line,
                     resolve_prompt_template, response_with_gpt_prompt,
                     run_except, splink_dedupe_job, sql_exec, sql_query, throw,
                     to_gpt_tuple, to_json, to_prompts, with_duckdb,
                     with_models, with_namespace)
from pymonad.array import Array
from pymonad.openai import GPTModel, GPTResponseTuple
from pymonad.run import fold_run
from pymonad.runsplink import PairsTableName, PredictionInputTableNames
from pymonad.traverse import array_traverse_run
from splink_types import SplinkType

PASS1_MODEL = "gpt-5-mini"
RANK_MODEL = "gpt-5-mini"
PASS1_MODEL_KEY = EnvKey("k_pass1_model")
RANK_MODEL_KEY = EnvKey("k_rank_model")
PASS1_PROMPT_VERSION = "k_pass1_v1"
RANK_PROMPT_VERSION = "k_rank_v2"
PASS1_PROMPT_KEY = "k_pass1_anchor_prompt"
RANK_PROMPT_KEY = "k_rank_candidate_prompt"
E2E_CACHE_STAGE = "adjudication_e2e"

HIGH_FREQ_ANCHOR_DOC_THRESHOLD = 120
MAX_FULLTEXT_CHARS = 24000
MAX_MERGED_CANDIDATES_FOR_API2 = 20

ORPHAN_ADJ_PROMPTS: dict[str, str | tuple[str, str] | tuple[str]] = {
    "k_limit": "Enter number of records requiring new API calls [0]: ",
    "k_start_after": "Starting after orphan_id (blank for none): ",
    "k_group_same_incident": "Group same incident? [Y/n]: ",
    "k_dry_run": "Dry run only (no overrides write)? [y/N]: ",
    "k_full_backfill": "Full backfill (ignore limit)? [y/N]: ",
    PASS1_PROMPT_KEY: ("pmpt_69ab817541588196935eb8137e392df80334ac0c71d05cf6",),
    RANK_PROMPT_KEY: ("pmpt_69aca43ee474819681d1bc3980e5d87c05903d6db9bfe4e1",),
}


class WordSet(RootModel[list[str]]):
    """A list of related search tokens for one retrieval variant."""


class Anchor(BaseModel):
    """A validated anchor theme and its retrieval variants from API 1."""

    model_config = ConfigDict(extra="forbid")
    anchor_theme: str
    variants: list[WordSet]


class Pass1Response(BaseModel):
    """Structured API 1 response containing candidate retrieval anchors."""

    model_config = ConfigDict(extra="forbid")
    anchors: list[Anchor]


class RankResponse(BaseModel):
    """Structured API 2 response containing the final candidate choice."""

    model_config = ConfigDict(extra="forbid")
    match_result: Literal["match", "no_match", "insufficient_information"]
    matched_entity_uid: str | None


@dataclass(frozen=True)
class CacheReadinessRow:
    """Run-scoped readiness snapshot for one adjudication target."""

    orphan_id: str
    article_id: int | None
    city_id: str
    year: int | None
    month: int | None
    midpoint_day: float | None
    weapon: str
    circumstance: str
    match_id: str
    queue_pos: int
    readiness_status: str
    readiness_reason: str
    pass1_idempotency_key: str
    rank_idempotency_key: str
    group_id: str = ""

    def with_readiness(
        self,
        *,
        readiness_status: str,
        readiness_reason: str,
        pass1_idempotency_key: str,
        rank_idempotency_key: str,
    ) -> "CacheReadinessRow":
        """Return a copy updated with the computed cache-readiness state."""
        return CacheReadinessRow(
            orphan_id=self.orphan_id,
            article_id=self.article_id,
            city_id=self.city_id,
            year=self.year,
            month=self.month,
            midpoint_day=self.midpoint_day,
            weapon=self.weapon,
            circumstance=self.circumstance,
            match_id=self.match_id,
            queue_pos=self.queue_pos,
            readiness_status=readiness_status,
            readiness_reason=readiness_reason,
            pass1_idempotency_key=pass1_idempotency_key,
            rank_idempotency_key=rank_idempotency_key,
            group_id=self.group_id,
        )

    def with_group(self, group_id: str) -> "CacheReadinessRow":
        """Return a copy assigned to a run-local processing group."""
        return CacheReadinessRow(
            orphan_id=self.orphan_id,
            article_id=self.article_id,
            city_id=self.city_id,
            year=self.year,
            month=self.month,
            midpoint_day=self.midpoint_day,
            weapon=self.weapon,
            circumstance=self.circumstance,
            match_id=self.match_id,
            queue_pos=self.queue_pos,
            readiness_status=self.readiness_status,
            readiness_reason=self.readiness_reason,
            pass1_idempotency_key=self.pass1_idempotency_key,
            rank_idempotency_key=self.rank_idempotency_key,
            group_id=group_id,
        )


@dataclass(frozen=True)
class Dossier:
    """Normalized source facts for one adjudication target."""

    orphan_id: str
    article_id: int | None
    city_id: str
    year: int | None
    month: int | None
    midpoint_day: float | None
    date_precision: str
    incident_date: str
    victim_count: str
    weapon: str
    circumstance: str
    geo_address_norm: str
    geo_address_short: str
    geo_address_short_2: str
    relationship: str
    incident_summary_gpt: str
    article_title: str
    article_text: str
    article_pub_date: str


@dataclass(frozen=True)
class GroupDossier:
    """Shared article/incident context for one same-incident orphan group."""

    group_key: str
    orphan_ids: tuple[str, ...]
    article_id: int | None
    incident_idx: int | None
    city_id: str
    year: int | None
    month: int | None
    midpoint_day: float | None
    incident_date: str
    victim_count: str
    incident_location: str
    incident_summary: str
    article_title: str
    article_text: str
    article_pub_date: str


@dataclass(frozen=True)
class Pass1Request:
    """Prompt payload for API 1 anchor generation."""

    article_title: str
    article_text: str
    article_date: str
    incident_date: str
    incident_location: str
    victim_count: str
    incident_summary: str

    def to_prompt_variables(self) -> dict[str, str]:
        """Serialize the request into stored-prompt variables."""
        return {
            "article_title": self.article_title,
            "article_text": self.article_text,
            "article_date": self.article_date,
            "incident_date": self.incident_date,
            "incident_location": self.incident_location,
            "victim_count": self.victim_count,
            "incident_summary": self.incident_summary,
        }

    def to_cache_payload(self) -> dict[str, Any]:
        """Serialize the request for diagnostic cache storage."""
        return self.to_prompt_variables()

    def to_display_payload(self) -> dict[str, Any]:
        """Serialize the request for display-only logging."""
        return {
            "article_title": self.article_title,
            "article_text": _truncate_article_text_for_display(self.article_text),
            "article_date": self.article_date,
            "incident_date": self.incident_date,
            "incident_location": self.incident_location,
            "victim_count": self.victim_count,
            "incident_summary": self.incident_summary,
        }


@dataclass(frozen=True)
class ValidatedAnchor:
    """Anchor that passed deterministic validation and FTS checks."""

    anchor_text: str
    variants: tuple[str, ...]
    variant_word_sets: tuple[tuple[str, ...], ...]
    doc_freq: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the anchor for evidence and table storage."""
        return {
            "anchor_text": self.anchor_text,
            "anchor_type": "phrase",
            "source_span": "",
            "incident_link_reason": "",
            "variants": list(self.variants),
            "variant_word_sets": [list(words) for words in self.variant_word_sets],
            "doc_freq": self.doc_freq,
        }


@dataclass(frozen=True)
class C2QuerySpec:
    """Display text plus safe SQLite FTS expression for one token bundle."""

    display_text: str
    fts_query: str


@dataclass(frozen=True)
class C2Candidate:
    """Entity candidate retrieved from the C2 full-text stage."""

    entity_uid: str
    source_article_id: int | None
    source_incident_idx: int | None
    midpoint_day: float | None
    day_gap: float
    compat_count: int
    summary_cosine: float
    det_score: float
    query_variant: str
    splink_match_weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate for persistence and logging."""
        return {
            "entity_uid": self.entity_uid,
            "source_article_id": self.source_article_id,
            "article_id": self.source_article_id,
            "source_incident_idx": self.source_incident_idx,
            "midpoint_day": self.midpoint_day,
            "day_gap": self.day_gap,
            "compat_count": self.compat_count,
            "summary_cosine": self.summary_cosine,
            "det_score": self.det_score,
            "query_variant": self.query_variant,
            "stage_name": "C2",
            "splink_match_weight": self.splink_match_weight,
        }


@dataclass(frozen=True)
class MergedCandidate:
    """Entity-level candidate merged across retrieval hits."""

    entity_uid: str
    det_score: float
    midpoint_day: float | None
    day_gap: float
    compat_count: int
    summary_cosine: float
    splink_match_weight: float | None
    anchor_hit_count: int
    source_stages: tuple[str, ...]
    query_variant: str
    query_variants: tuple[str, ...]
    source_article_id: int | None
    source_article_ids: tuple[int, ...]
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the merged candidate for logging and evidence."""
        return {
            "entity_uid": self.entity_uid,
            "det_score": self.det_score,
            "midpoint_day": self.midpoint_day,
            "day_gap": self.day_gap,
            "compat_count": self.compat_count,
            "summary_cosine": self.summary_cosine,
            "splink_match_weight": self.splink_match_weight,
            "anchor_hit_count": self.anchor_hit_count,
            "source_stages": list(self.source_stages),
            "query_variant": self.query_variant,
            "query_variants": list(self.query_variants),
            "source_article_id": self.source_article_id,
            "source_article_ids": list(self.source_article_ids),
            "rows": list(self.rows),
        }


@dataclass(frozen=True)
class IncidentRepresentativeCandidate:
    """One API-2 candidate representing a single candidate incident."""

    entity_uid: str
    source_article_id: int | None
    source_incident_idx: int | None
    midpoint_day: float | None
    day_gap: float
    compat_count: int
    summary_cosine: float
    det_score: float
    splink_match_weight: float | None
    anchor_hit_count: int
    query_variant: str
    query_variants: tuple[str, ...]
    eligible_entity_uids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the incident representative for logging and evidence."""
        return {
            "entity_uid": self.entity_uid,
            "source_article_id": self.source_article_id,
            "article_id": self.source_article_id,
            "source_incident_idx": self.source_incident_idx,
            "midpoint_day": self.midpoint_day,
            "day_gap": self.day_gap,
            "compat_count": self.compat_count,
            "summary_cosine": self.summary_cosine,
            "det_score": self.det_score,
            "splink_match_weight": self.splink_match_weight,
            "anchor_hit_count": self.anchor_hit_count,
            "query_variant": self.query_variant,
            "query_variants": list(self.query_variants),
            "eligible_entity_uids": list(self.eligible_entity_uids),
        }


@dataclass(frozen=True)
class RankCandidateContext:
    """Candidate article context presented to API 2."""

    entity_uid: str
    article_title: str
    article_text: str
    article_date: str
    incident_date: str
    incident_location: str
    victim_count: str
    incident_summary: str

    def to_prompt_payload(self) -> dict[str, Any]:
        """Serialize one candidate context for stored-prompt input."""
        return {
            "entity_uid": self.entity_uid,
            "article_title": self.article_title,
            "article_text": self.article_text,
            "article_date": self.article_date,
            "incident_date": self.incident_date,
            "incident_location": self.incident_location,
            "victim_count": self.victim_count,
            "incident_summary": self.incident_summary,
        }

    def to_display_payload(self) -> dict[str, Any]:
        """Serialize one candidate context for display-only logging."""
        return {
            "entity_uid": self.entity_uid,
            "article_title": self.article_title,
            "article_text": _truncate_article_text_for_display(self.article_text),
            "article_date": self.article_date,
            "incident_date": self.incident_date,
            "incident_location": self.incident_location,
            "victim_count": self.victim_count,
            "incident_summary": self.incident_summary,
        }


@dataclass(frozen=True)
class RankRequest:
    """Prompt payload for API 2 direct candidate ranking."""

    article_title: str
    article_text: str
    article_date: str
    incident_date: str
    incident_location: str
    victim_count: str
    incident_summary: str
    candidates: tuple[RankCandidateContext, ...]

    def to_prompt_variables(self) -> dict[str, str]:
        """Serialize the request into stored-prompt variables."""
        return _stringify_prompt_vars(
            {
                "target_article_title": self.article_title,
                "target_article_text": self.article_text,
                "target_article_date": self.article_date,
                "target_incident_date": self.incident_date,
                "target_incident_location": self.incident_location,
                "target_victim_count": self.victim_count,
                "target_incident_summary": self.incident_summary,
                "candidates": [
                    candidate.to_prompt_payload() for candidate in self.candidates
                ],
            }
        )

    def to_display_payload(self) -> dict[str, Any]:
        """Serialize the request into a log-friendly structure."""
        return {
            "article_title": self.article_title,
            "article_text": _truncate_article_text_for_display(self.article_text),
            "article_date": self.article_date,
            "incident_date": self.incident_date,
            "incident_location": self.incident_location,
            "victim_count": self.victim_count,
            "incident_summary": self.incident_summary,
            "candidates": [
                candidate.to_display_payload() for candidate in self.candidates
            ],
        }


@dataclass(frozen=True)
class RunSummary:
    """Aggregate controller outcomes for one [K] run."""

    needs_api_total: int
    selected_needs_api: int
    decision_ready_from_cache: int
    processed: int
    grouped: int
    matched: int
    not_same_person: int
    insufficient_information: int
    analysis_incomplete: int
    rebuilt_terminal_rows: int
    dry_run: bool


@dataclass(frozen=True)
class KParams:
    """Interactive execution parameters for controller [K]."""

    limit: int
    starting_after_orphan_id: str
    group_same_incident: bool
    dry_run: bool
    full_backfill: bool


@dataclass
class CaseDecision:
    """Final adjudication decision for one target article-victim row."""

    orphan_id: str
    article_id: int | None
    label: str
    resolved_entity_id: str | None
    confidence: float | None
    reason_summary: str
    evidence_json: dict[str, Any]


@dataclass
class OrphanWork:
    """Intermediate per-target work product before final group assignment."""

    orphan_id: str
    article_id: int | None
    insufficient: bool
    insufficient_reason: str | None
    stage_trace: list[dict[str, Any]]
    anchors: list[dict[str, Any]]
    valid_variants: list[str]
    merged_candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    provisional_reason: str



def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _display_safe_id(raw: str | None) -> str:
    """Render colon-delimited IDs with zero-width breaks for readable logs."""
    return _safe_text(raw).replace(":", ":\u200b")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sanitize_sql_identifier(raw: str, *, prefix: str = "k") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", _safe_text(raw))
    cleaned = cleaned.strip("_")
    if cleaned == "":
        cleaned = "tmp"
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return f"{prefix}_{cleaned[:48]}"


def _dedupe_model_path() -> Path:
    key_str = str(SplinkType.DEDUP).replace("/", "_")
    return Path("splink_models") / f"splink_model_{key_str}.json"


def _load_dedupe_model_settings() -> dict[str, Any]:
    model_path = _dedupe_model_path()
    if not model_path.exists():
        raise RuntimeError(
            "dedupe_splink_model_missing: run [D] before C2 Splink scoring"
        )
    try:
        with open(model_path, "r", encoding="utf-8") as handle:
            settings = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"dedupe_splink_model_load_failed: {exc}") from exc
    if not isinstance(settings, dict) or "comparisons" not in settings:
        raise RuntimeError("dedupe_splink_model_invalid: comparisons missing")
    return settings


def _settings_for_c2_weight_scoring(dedupe_settings: dict[str, Any]) -> dict[str, Any]:
    settings = dict(dedupe_settings)
    settings.update(
        {
            "link_type": "link_only",
            "unique_id_column_name": "unique_id",
            "blocking_rules_to_generate_predictions": ["1=1"],
        }
    )
    return settings


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _as_float(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default






def _pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


def _format_pass1_response_for_display(response: Pass1Response) -> str:
    lines = ['{', '  "anchors": [']
    anchors = response.anchors
    for index, anchor in enumerate(anchors):
        lines.append("    {")
        lines.append(
            '      "anchor_theme": '
            + json.dumps(anchor.anchor_theme, ensure_ascii=True)
            + ","
        )
        lines.append('      "variants": [')
        variants = anchor.variants
        for variant_index, word_set in enumerate(variants):
            suffix = "," if variant_index < len(variants) - 1 else ""
            lines.append(
                "        "
                + json.dumps(word_set.root, ensure_ascii=True)
                + suffix
            )
        lines.append("      ]")
        anchor_suffix = "," if index < len(anchors) - 1 else ""
        lines.append("    }" + anchor_suffix)
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


def _truncate_article_text_for_display(text: str, *, limit: int = 100) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_fts_bundle_token(token: str) -> str:
    cleaned = re.sub(r"\s+", " ", _safe_text(token)).strip()
    if cleaned == "":
        return ""
    return cleaned.replace('"', '""')


def _build_fts_query_from_word_set(word_set: tuple[str, ...]) -> str:
    tokens = [
        normalized
        for normalized in (_normalize_fts_bundle_token(token) for token in word_set)
        if normalized != ""
    ]
    if len(tokens) == 0:
        return ""
    return " AND ".join(f'"{token}"' for token in tokens)


def _query_specs_from_validated_anchors(
    valid_anchors: Array[ValidatedAnchor],
) -> Array[C2QuerySpec]:
    seen: dict[str, C2QuerySpec] = {}
    for anchor in valid_anchors:
        for word_set in anchor.variant_word_sets:
            display_text = " ".join(word_set).strip()
            fts_query = _build_fts_query_from_word_set(word_set)
            if display_text == "" or fts_query == "":
                continue
            if display_text not in seen:
                seen[display_text] = C2QuerySpec(
                    display_text=display_text,
                    fts_query=fts_query,
                )
    return Array.make(tuple(seen.values()))


def _norm(s: str) -> str:
    s2 = re.sub(r"\s+", " ", _safe_text(s)).strip().lower()
    return s2


def _parse_bool(raw: str, default: bool) -> bool:
    t = raw.strip().lower()
    if t == "":
        return default
    if t in {"y", "yes", "1", "true", "t"}:
        return True
    if t in {"n", "no", "0", "false", "f"}:
        return False
    return default


def _parse_limit(raw: str, *, default: int = 20) -> int:
    t = raw.strip()
    if t == "":
        return max(0, default)
    try:
        n = int(t)
    except ValueError:
        return max(0, default)
    return max(0, n)


def _with_sqlite(subprog: Run[Any]) -> Run[Any]:
    return local(lambda env: {**env, "current_backend": DbBackend.SQLITE}, subprog)


def _rows_to_dicts(rows: Array[Any]) -> list[dict[str, Any]]:
    return [dict(cast(dict[str, Any], row)) for row in rows]


def _sql_param(value: Any) -> String | int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return String(value)
    if isinstance(value, (int, float)):
        return value
    return String(str(value))


def _sql_params(values: tuple[Any, ...]) -> SQLParams:
    return SQLParams(tuple(_sql_param(value) for value in values))


def _parse_full_date(raw: Any) -> tuple[int, int, int] | None:
    text = _safe_text(raw).strip()
    if text == "":
        return None

    token = text.split("T", 1)[0].split(" ", 1)[0].strip()

    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", token)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (year, month, day)

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", token)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (year, month, day)

    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", token)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (year, month, day)

    return None


def _incident_idx_from_orphan_uid(uid: str) -> int | None:
    parts = uid.split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _parse_month_year(raw: Any) -> tuple[int, int] | None:
    text = _safe_text(raw).strip()
    if text == "":
        return None
    token = text.split("T", 1)[0].split(" ", 1)[0].strip()

    m = re.match(r"^(\d{4})[-/](\d{1,2})$", token)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    m = re.match(r"^(\d{1,2})/(\d{4})$", token)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    m = re.match(r"^(\d{4})(\d{2})$", token)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    return None


def _parse_year_only(raw: Any) -> int | None:
    text = _safe_text(raw).strip()
    if text == "":
        return None
    token = text.split("T", 1)[0].split(" ", 1)[0].strip()

    m = re.match(r"^(\d{4})$", token)
    if m:
        return int(m.group(1))
    return None


def _int_or_none(raw: Any) -> int | None:
    text = _safe_text(raw).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _format_article_date(raw: Any) -> str:
    ymd = _parse_full_date(raw)
    if ymd is None:
        return ""
    y, m, d = ymd
    return f"{m:02d}/{d:02d}/{y:04d}"


def _format_incident_date(raw: Any, *, year: Any = None, month: Any = None) -> str:
    ymd = _parse_full_date(raw)
    if ymd is not None:
        y, m, d = ymd
        return f"{m:02d}/{d:02d}/{y:04d}"

    year_i = _int_or_none(year)
    month_i = _int_or_none(month)

    if year_i is None or month_i is None:
        ym = _parse_month_year(raw)
        if ym is not None:
            year_i, month_i = ym

    if year_i is not None and month_i is not None and 1 <= month_i <= 12:
        return f"{month_i:02d}/{year_i:04d}"

    if year_i is None:
        year_i = _parse_year_only(raw)
    if year_i is not None:
        return f"{year_i:04d}"

    return ""
















def _count_by_year(
    rows: list[dict[str, Any]], *, status: str | None = None
) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in rows:
        if status is not None and _safe_text(r.get("readiness_status")) != status:
            continue
        year_val = r.get("year")
        year_key = "unknown" if year_val is None else str(year_val)
        counts[year_key] = counts.get(year_key, 0) + 1
    return sorted(counts.items(), key=lambda t: (t[0] == "unknown", t[0]))


def _format_cache_readiness_summary(rows: list[dict[str, Any]]) -> str:
    needs_total = sum(
        1 for r in rows if _safe_text(r.get("readiness_status")) == "needs_api"
    )

    lines: list[str] = ["[K] Orphans requiring new API calls by year:"]
    for year_key, n in _count_by_year(rows, status="needs_api"):
        lines.append(f"  {year_key}: {n}")
    lines.append(f"  total: {needs_total}")
    return "\n".join(lines)


def _select_needs_api_rows(
    readiness_rows: list[dict[str, Any]], params: KParams
) -> list[dict[str, Any]]:
    needs_rows = [
        r
        for r in readiness_rows
        if _safe_text(r.get("readiness_status")) == "needs_api"
    ]
    needs_rows.sort(
        key=lambda r: (int(r.get("queue_pos") or 0), _safe_text(r.get("orphan_id")))
    )

    if params.starting_after_orphan_id.strip() != "":
        start_pos = None
        for r in needs_rows:
            if _safe_text(r.get("orphan_id")) == params.starting_after_orphan_id:
                start_pos = int(r.get("queue_pos") or 0)
                break
        if start_pos is not None:
            needs_rows = [
                r for r in needs_rows if int(r.get("queue_pos") or 0) > start_pos
            ]

    if params.full_backfill:
        return needs_rows
    selected = needs_rows[: params.limit]
    if len(selected) == 0:
        return selected

    boundary_group_key = _incident_group_key_from_orphan_id(
        _safe_text(selected[-1].get("orphan_id"))
    )
    boundary_queue_pos = int(selected[-1].get("queue_pos") or 0)
    trailing_group_rows = [
        row
        for row in needs_rows
        if int(row.get("queue_pos") or 0) > boundary_queue_pos
        and _incident_group_key_from_orphan_id(_safe_text(row.get("orphan_id")))
        == boundary_group_key
    ]
    return selected + trailing_group_rows




def _incident_group_key_from_orphan_id(orphan_id: str) -> str:
    parts = _safe_text(orphan_id).split(":")
    if len(parts) >= 2 and parts[0] != "" and parts[1] != "":
        return f"{parts[0]}:{parts[1]}"
    return _safe_text(orphan_id)


def _build_groups(
    rows: list[dict[str, Any]], group_same_incident: bool, run_id: str
) -> list[list[dict[str, Any]]]:
    if not group_same_incident:
        return [[r] for r in rows]
    if len(rows) == 0:
        return []

    grouped_by_key: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for row in rows:
        group_key = _incident_group_key_from_orphan_id(
            _safe_text(row.get("orphan_id"))
        )
        if group_key not in grouped_by_key:
            grouped_by_key[group_key] = []
            group_order.append(group_key)
        grouped_by_key[group_key].append(row)

    grouped = [grouped_by_key[group_key] for group_key in group_order]

    for i, grp in enumerate(grouped, start=1):
        gid = f"{run_id}_G{i:04d}"
        for r in grp:
            r["group_id"] = gid
    return grouped












def _case_fingerprint(
    payload: dict[str, Any], *, model: str, prompt_version: str
) -> str:
    return _sha256(
        _canonical_json(
            {"model": model, "prompt_version": prompt_version, "payload": payload}
        )
    )




def _decision_to_cache_payload(decision: CaseDecision) -> dict[str, Any]:
    return {
        "orphan_id": decision.orphan_id,
        "article_id": decision.article_id,
        "label": decision.label,
        "resolved_entity_id": decision.resolved_entity_id,
        "confidence": decision.confidence,
        "reason_summary": decision.reason_summary,
        "evidence_json": decision.evidence_json,
    }


def _candidate_source_article_id(row: dict[str, Any]) -> int | None:
    source_article_id = row.get("source_article_id")
    if source_article_id is not None:
        try:
            return int(source_article_id)
        except (TypeError, ValueError):
            pass

    article_ids = row.get("source_article_ids")
    if isinstance(article_ids, list):
        for value in article_ids:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue

    rows = row.get("rows")
    if isinstance(rows, list):
        for child in rows:
            if not isinstance(child, dict):
                continue
            value = child.get("source_article_id") or child.get("article_id")
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                continue

    return None


def _incident_key_from_candidate(
    source_article_id: int | None, source_incident_idx: int | None
) -> str:
    article_part = "" if source_article_id is None else str(source_article_id)
    incident_part = "" if source_incident_idx is None else str(source_incident_idx)
    return f"{article_part}:{incident_part}"


def _group_dossier_from_dossier(
    group_key: str, orphan_ids: tuple[str, ...], dossier: Dossier
) -> GroupDossier:
    return GroupDossier(
        group_key=group_key,
        orphan_ids=orphan_ids,
        article_id=dossier.article_id,
        incident_idx=_incident_idx_from_orphan_uid(dossier.orphan_id),
        city_id=dossier.city_id,
        year=dossier.year,
        month=dossier.month,
        midpoint_day=dossier.midpoint_day,
        incident_date=_format_incident_date(
            dossier.incident_date, year=dossier.year, month=dossier.month
        ),
        victim_count=dossier.victim_count,
        incident_location=dossier.geo_address_norm,
        incident_summary=dossier.incident_summary_gpt,
        article_title=dossier.article_title,
        article_text=dossier.article_text,
        article_pub_date=dossier.article_pub_date,
    )


def _build_pass1_request_for_group_v2(group: GroupDossier) -> Pass1Request:
    return Pass1Request(
        article_title=group.article_title,
        article_text=group.article_text,
        article_date=_format_article_date(group.article_pub_date),
        incident_date=group.incident_date,
        incident_location=group.incident_location,
        victim_count=group.victim_count,
        incident_summary=group.incident_summary,
    )


def _decision_from_cache_payload(
    payload: dict[str, Any], fallback_orphan_id: str
) -> CaseDecision:
    raw_evidence = payload.get("evidence_json")
    evidence_json = raw_evidence if isinstance(raw_evidence, dict) else {}
    confidence_raw = payload.get("confidence")
    confidence = float(confidence_raw) if confidence_raw is not None else None
    article_raw = payload.get("article_id")
    try:
        article_id = int(article_raw) if article_raw is not None else None
    except (TypeError, ValueError):
        article_id = None
    return CaseDecision(
        orphan_id=fallback_orphan_id,
        article_id=article_id,
        label=_safe_text(payload.get("label")),
        resolved_entity_id=_safe_text(payload.get("resolved_entity_id")) or None,
        confidence=confidence,
        reason_summary=_safe_text(payload.get("reason_summary")),
        evidence_json=evidence_json,
    )




def _stringify_prompt_vars(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in payload.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, (int, float, bool)):
            out[k] = str(v)
        else:
            out[k] = json.dumps(v, ensure_ascii=True)
    return out






def _structured_echo_terms(dossier: dict[str, Any]) -> list[str]:
    terms = [
        _safe_text(dossier.get("weapon")),
        _safe_text(dossier.get("circumstance")),
        _safe_text(dossier.get("geo_address_norm")),
        _safe_text(dossier.get("geo_address_short")),
        _safe_text(dossier.get("geo_address_short_2")),
    ]
    cleaned: list[str] = []
    for t in terms:
        n = _norm(t)
        if n != "" and n not in {"unknown", "undetermined", "none", "null"}:
            cleaned.append(n)
    return cleaned






def _pass1_gate_failed(
    valid_anchors: list[dict[str, Any]], anchor_failures: list[str]
) -> bool:
    if len(valid_anchors) < 1:
        return True
    return any(
        failure
        in {"no_valid_anchors", "insufficient_anchor_diversity", "no_query_variants"}
        for failure in anchor_failures
    )












def _merge_candidates(
    bc_rows: list[dict[str, Any]],
    c2_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_entity: dict[str, dict[str, Any]] = {}
    for r in bc_rows + c2_rows:
        entity_uid = _safe_text(r.get("entity_uid"))
        if entity_uid == "":
            continue
        existing = by_entity.get(entity_uid)
        source_stage = _safe_text(r.get("stage_name"))
        if existing is None:
            by_entity[entity_uid] = {
                "entity_uid": entity_uid,
                "det_score": _as_float(r.get("det_score"), 0.0),
                "day_gap": _as_float(r.get("day_gap"), 9999.0),
                "compat_count": _as_int(r.get("compat_count"), 0),
                "summary_cosine": _as_float(r.get("summary_cosine"), 0.0),
                "splink_match_weight": (
                    _as_float(r.get("splink_match_weight"), 0.0)
                    if r.get("splink_match_weight") is not None
                    else None
                ),
                "source_stages": {source_stage},
                "source_article_hits": (
                    set([r.get("source_article_id")])
                    if r.get("source_article_id") is not None
                    else set()
                ),
                "rows": [r],
            }
            continue

        existing["det_score"] = max(
            _as_float(existing["det_score"], 0.0), _as_float(r.get("det_score"), 0.0)
        )
        existing["day_gap"] = min(
            _as_float(existing["day_gap"], 9999.0), _as_float(r.get("day_gap"), 9999.0)
        )
        existing["compat_count"] = max(
            _as_int(existing["compat_count"], 0), _as_int(r.get("compat_count"), 0)
        )
        existing["summary_cosine"] = max(
            _as_float(existing["summary_cosine"], 0.0),
            _as_float(r.get("summary_cosine"), 0.0),
        )
        row_weight = (
            _as_float(r.get("splink_match_weight"), 0.0)
            if r.get("splink_match_weight") is not None
            else None
        )
        existing_weight = existing.get("splink_match_weight")
        if row_weight is not None and (
            existing_weight is None
            or _as_float(row_weight, 0.0) > _as_float(existing_weight, 0.0)
        ):
            existing["splink_match_weight"] = row_weight
        existing["source_stages"].add(source_stage)
        if r.get("source_article_id") is not None:
            existing["source_article_hits"].add(r.get("source_article_id"))
        existing["rows"].append(r)

    merged: list[dict[str, Any]] = []
    for entity_uid, rec in by_entity.items():
        stages = sorted(s for s in rec["source_stages"] if s)
        anchor_hits = len(rec["source_article_hits"])
        merged_score = float(rec["det_score"]) + 0.5 * anchor_hits
        row_entries = [row for row in rec["rows"] if isinstance(row, dict)]
        query_variants = list(
            dict.fromkeys(
                _safe_text(row.get("query_variant")).strip()
                for row in row_entries
                if _safe_text(row.get("query_variant")).strip() != ""
            )
        )
        midpoint_vals = [
            _as_float(row.get("midpoint_day"), 0.0)
            for row in row_entries
            if row.get("midpoint_day") is not None
        ]
        source_article_ids = sorted(
            int(v)
            for v in rec["source_article_hits"]
            if v is not None and _safe_text(v).strip().isdigit()
        )
        merged.append(
            {
                "entity_uid": entity_uid,
                "det_score": merged_score,
                "midpoint_day": midpoint_vals[0] if len(midpoint_vals) > 0 else None,
                "day_gap": float(rec["day_gap"]),
                "compat_count": int(rec["compat_count"]),
                "summary_cosine": float(rec["summary_cosine"]),
                "splink_match_weight": rec.get("splink_match_weight"),
                "anchor_hit_count": anchor_hits,
                "source_stages": stages,
                "query_variant": query_variants[0] if len(query_variants) > 0 else "",
                "query_variants": query_variants,
                "source_article_id": (
                    source_article_ids[0] if len(source_article_ids) > 0 else None
                ),
                "source_article_ids": source_article_ids,
                "rows": rec["rows"],
            }
        )

    merged.sort(
        key=lambda r: (
            -float(r["det_score"]),
            -int(r["anchor_hit_count"]),
            -float(r["summary_cosine"]),
            -int(r["compat_count"]),
            float(r["day_gap"]),
            _safe_text(r["entity_uid"]),
        )
    )
    return merged[:MAX_MERGED_CANDIDATES_FOR_API2]




def _validate_rank_output(
    ranked: RankResponse,
    merged: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if ranked.match_result != "match":
        return []

    allowed = {_safe_text(r.get("entity_uid")) for r in merged}
    merged_by_uid = {_safe_text(r.get("entity_uid")): r for r in merged}

    uid = _safe_text(ranked.matched_entity_uid)
    if uid == "" or uid not in allowed:
        return []
    base = merged_by_uid[uid]
    return [
        {
            "entity_uid": uid,
            "rank_score": 1.0,
            "det_score": float(base.get("det_score") or 0.0),
            "splink_match_weight": (
                _as_float(base.get("splink_match_weight"), 0.0)
                if base.get("splink_match_weight") is not None
                else None
            ),
            "anchor_hit_count": int(base.get("anchor_hit_count") or 0),
            "summary_cosine": float(base.get("summary_cosine") or 0.0),
            "compat_count": int(base.get("compat_count") or 0),
            "day_gap": _as_float(base.get("day_gap"), 9999.0),
            "source_stages": base.get("source_stages") or [],
        }
    ]




























def _compact_candidate_view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        raw_query_variants = r.get("query_variants")
        query_variants: list[str] = (
            [_safe_text(v) for v in raw_query_variants]
            if isinstance(raw_query_variants, list)
            else []
        )
        query_variant = _safe_text(r.get("query_variant"))
        if query_variant == "" and len(query_variants) > 0:
            query_variant = " | ".join(v for v in query_variants[:3] if v != "")
        out.append(
            {
                "entity_uid": _safe_text(r.get("entity_uid")),
                "article_id": r.get("source_article_id"),
                "stage_name": _safe_text(r.get("stage_name")),
                "query_variant": query_variant,
                "query_variants": query_variants,
                "source_article_id": r.get("source_article_id"),
                "source_article_ids": r.get("source_article_ids") or [],
                "det_score": float(r.get("det_score") or 0.0),
                "splink_match_weight": (
                    _as_float(r.get("splink_match_weight"), 0.0)
                    if r.get("splink_match_weight") is not None
                    else None
                ),
                "rank_score": float(r.get("rank_score") or 0.0),
                "compat_count": int(r.get("compat_count") or 0),
                "summary_cosine": float(r.get("summary_cosine") or 0.0),
                "day_gap": float(r.get("day_gap") or 0.0),
                "anchor_hit_count": int(r.get("anchor_hit_count") or 0),
                "source_stages": r.get("source_stages") or [],
            }
        )
    return out


def _format_c2_candidates_table(
    rows: list[dict[str, Any]], *, include_anchor_hit_count: bool = False
) -> str:
    headers = [
        "entity_uid",
        "article_id",
        "query_variant",
        "midpoint_day",
        "det_score",
        "splink_match_weight",
        "compat_count",
        "summary_cosine",
        "day_gap",
    ]
    if include_anchor_hit_count:
        headers.append("anchor_hit_count")
    if len(rows) == 0:
        return "  (none)"

    table_rows: list[list[str]] = []
    for r in rows:
        entity_uid = _safe_text(r.get("entity_uid"))
        article_id = _safe_text(
            r.get("article_id")
            if r.get("article_id") is not None
            else r.get("source_article_id")
        )
        query_variant = _safe_text(r.get("query_variant"))
        midpoint_day = (
            f"{_as_float(r.get('midpoint_day'), 0.0):.0f}"
            if r.get("midpoint_day") is not None
            else ""
        )
        det_score = f"{_as_float(r.get('det_score'), 0.0):.4f}"
        splink_weight = (
            f"{_as_float(r.get('splink_match_weight'), 0.0):.4f}"
            if r.get("splink_match_weight") is not None
            else ""
        )
        compat_count = str(_as_int(r.get("compat_count"), 0))
        summary_cosine = f"{_as_float(r.get('summary_cosine'), 0.0):.4f}"
        day_gap = f"{_as_float(r.get('day_gap'), 0.0):.0f}"
        row_vals = [
            entity_uid,
            article_id,
            query_variant,
            midpoint_day,
            det_score,
            splink_weight,
            compat_count,
            summary_cosine,
            day_gap,
        ]
        if include_anchor_hit_count:
            row_vals.append(str(_as_int(r.get("anchor_hit_count"), 0)))
        table_rows.append(row_vals)

    widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(vals: list[str]) -> str:
        return " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals)))

    lines = [_fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(_fmt_row(row) for row in table_rows)
    return "\n".join(lines)








def _assign_group(work_items: list[OrphanWork]) -> dict[str, dict[str, Any]]:
    proposals: list[tuple[str, int | None, dict[str, Any]]] = []
    for w in work_items:
        if w.insufficient:
            continue
        if len(w.ranked_candidates) == 0:
            continue
        proposals.append((w.orphan_id, w.article_id, w.ranked_candidates[0]))

    proposals.sort(key=lambda t: (-float(t[2].get("rank_score") or 0.0), t[0]))

    used_entities: set[str] = set()
    used_article_entity: set[tuple[int | None, str]] = set()
    assigned: dict[str, dict[str, Any]] = {}

    for orphan_id, article_id, cand in proposals:
        uid = _safe_text(cand.get("entity_uid"))
        if uid == "":
            continue
        if uid in used_entities:
            continue
        if (article_id, uid) in used_article_entity:
            continue

        assigned[orphan_id] = cand
        used_entities.add(uid)
        used_article_entity.add((article_id, uid))

    return assigned


def _decision_from_work(
    work: OrphanWork,
    assigned_candidate: dict[str, Any] | None,
) -> CaseDecision:
    if work.insufficient:
        reason_code = "insufficient_information_pass1_gate"
        reason_summary = (
            "Pass-1 anchor extraction did not yield incident-specific, valid anchors"
            " after deterministic validation. Because no reliable anchor set was"
            " available, candidate expansion and ranking were not sufficient for"
            " identity adjudication."
        )
        conflict_analysis = "insufficient anchor detail"
        if work.insufficient_reason == "rank":
            reason_code = "insufficient_information_rank_api"
            reason_summary = (
                "Candidate retrieval completed, but the ranking API determined the"
                " article lacks enough distinguishing information to make a defensible"
                " one-candidate-vs-none decision."
            )
            conflict_analysis = "insufficient differentiating detail in target article"
        evidence = {
            "stage_trace": work.stage_trace,
            "narrative_evidence": {
                "orphan_anchors": work.anchors,
                "candidate_anchors": [],
                "conflict_analysis": conflict_analysis,
            },
            "top_candidates": [],
            "reason_code": reason_code,
            "execution_audit": {
                "query_mode": "interactive_sql",
                "fts_used": False,
                "fts_query_list": work.valid_variants,
            },
            "policy_enforcement": {
                "stage_a_removed": True,
                "terminal_non_apply": True,
            },
        }
        return CaseDecision(
            orphan_id=work.orphan_id,
            article_id=work.article_id,
            label="insufficient_information",
            resolved_entity_id=None,
            confidence=None,
            reason_summary=reason_summary,
            evidence_json=evidence,
        )

    top = work.ranked_candidates[:3]
    if assigned_candidate is None:
        reason_code = "no_valid_winner_after_rank_and_constraints"
        reason_summary = (
            "Candidate generation and ranking were completed, but no entity passed"
            " the deterministic one-to-one and conflict gates. Reviewed candidates"
            " lacked sufficient identity-level correspondence to support a"
            " defensible match."
        )
        conflict_analysis = "no candidate passed deterministic assignment gates"
        if work.provisional_reason == "no matching incident from rank api":
            reason_code = "no_matching_incident_after_group_rank"
            reason_summary = (
                "The group-level ranking step reviewed incident-deduplicated"
                " candidates and did not identify any candidate incident as the"
                " same homicide incident."
            )
            conflict_analysis = "no candidate incident matched the target incident"
        elif (
            work.provisional_reason
            == "matched incident selected but orphan unassigned"
        ):
            reason_code = "matched_incident_but_orphan_unassigned_after_matching"
            reason_summary = (
                "A candidate incident was selected, but this orphan was not part of"
                " the best one-to-one assignment after same-article exclusion and"
                " Splink-based within-incident matching."
            )
            conflict_analysis = (
                "matched incident selected, but this orphan was not in the optimal"
                " allowed assignment"
            )
        evidence = {
            "stage_trace": work.stage_trace,
            "narrative_evidence": {
                "orphan_anchors": work.anchors,
                "candidate_anchors": top,
                "conflict_analysis": conflict_analysis,
            },
            "top_candidates": [
                _safe_text(c.get("entity_uid"))
                for c in top
                if _safe_text(c.get("entity_uid")) != ""
            ],
            "reason_code": reason_code,
            "execution_audit": {
                "query_mode": "interactive_sql",
                "fts_used": len(work.valid_variants) > 0,
                "fts_query_list": work.valid_variants,
            },
            "policy_enforcement": {
                "one_to_one_mapping": True,
                "same_article_entity_uniqueness": True,
            },
        }
        return CaseDecision(
            orphan_id=work.orphan_id,
            article_id=work.article_id,
            label="not_same_person",
            resolved_entity_id=None,
            confidence=None,
            reason_summary=reason_summary,
            evidence_json=evidence,
        )

    uid = _safe_text(assigned_candidate.get("entity_uid"))
    evidence = {
        "stage_trace": work.stage_trace,
        "narrative_evidence": {
            "orphan_anchors": work.anchors,
            "candidate_anchors": [assigned_candidate],
            "conflict_analysis": (
                "no blocking contradiction after post-rank constraints"
            ),
        },
        "top_candidates": [
            _safe_text(c.get("entity_uid"))
            for c in work.ranked_candidates[:3]
            if _safe_text(c.get("entity_uid")) != ""
        ],
        "reason_code": "matched_via_ranked_candidate_after_constraints",
        "execution_audit": {
            "query_mode": "interactive_sql",
            "fts_used": len(work.valid_variants) > 0,
            "fts_query_list": work.valid_variants,
        },
        "policy_enforcement": {
            "one_to_one_mapping": True,
            "same_article_entity_uniqueness": True,
            "assignment_mode": "maximum_weight_matching",
        },
    }
    return CaseDecision(
        orphan_id=work.orphan_id,
        article_id=work.article_id,
        label="matched",
        resolved_entity_id=uid,
        confidence=None,
        reason_summary=(
            f"The highest-ranked candidate {uid} is the only option that passed"
            " deterministic assignment constraints after anchor-led retrieval and"
            " ranking. Incident-specific anchor evidence and candidate-level"
            " compatibility supported same-person plausibility without a disqualifying"
            " conflict."
        ),
        evidence_json=evidence,
    )




def _decision_hash_for_case(decision: CaseDecision) -> str:
    payload = {
        "label": decision.label,
        "resolved_entity_id": decision.resolved_entity_id,
        "reason_summary": decision.reason_summary,
        "evidence_digest": {
            "top_candidates": decision.evidence_json.get("top_candidates"),
            "reason_code": decision.evidence_json.get("reason_code"),
        },
    }
    return _sha256(_canonical_json(payload))






def _prior_decision_hash(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    ev = row.get("evidence_json")
    if isinstance(ev, dict):
        h = _safe_text(ev.get("decision_hash"))
        return h if h != "" else None
    try:
        obj = json.loads(_safe_text(ev))
        if isinstance(obj, dict):
            h = _safe_text(obj.get("decision_hash"))
            return h if h != "" else None
    except json.JSONDecodeError:
        return None
    return None




















# === Monadic [K] controller overrides ===


def _cache_readiness_row_from_dict_v2(row: dict[str, Any]) -> CacheReadinessRow:
    return CacheReadinessRow(
        orphan_id=_safe_text(row.get("orphan_id")),
        article_id=(
            _as_int(row.get("article_id"), 0)
            if row.get("article_id") is not None
            else None
        ),
        city_id=_safe_text(row.get("city_id")),
        year=_as_int(row.get("year"), 0) if row.get("year") is not None else None,
        month=_as_int(row.get("month"), 0) if row.get("month") is not None else None,
        midpoint_day=(
            _as_float(row.get("midpoint_day"), 0.0)
            if row.get("midpoint_day") is not None
            else None
        ),
        weapon=_safe_text(row.get("weapon")),
        circumstance=_safe_text(row.get("circumstance")),
        match_id=_safe_text(row.get("match_id")),
        queue_pos=_as_int(row.get("queue_pos"), 0),
        readiness_status=_safe_text(row.get("readiness_status")),
        readiness_reason=_safe_text(row.get("readiness_reason")),
        pass1_idempotency_key=_safe_text(row.get("pass1_idempotency_key")),
        rank_idempotency_key=_safe_text(row.get("rank_idempotency_key")),
        group_id=_safe_text(row.get("group_id")),
    )


def _cache_readiness_row_to_dict_v2(row: CacheReadinessRow) -> dict[str, Any]:
    return {
        "orphan_id": row.orphan_id,
        "article_id": row.article_id,
        "city_id": row.city_id,
        "year": row.year,
        "month": row.month,
        "midpoint_day": row.midpoint_day,
        "weapon": row.weapon,
        "circumstance": row.circumstance,
        "match_id": row.match_id,
        "queue_pos": row.queue_pos,
        "readiness_status": row.readiness_status,
        "readiness_reason": row.readiness_reason,
        "pass1_idempotency_key": row.pass1_idempotency_key,
        "rank_idempotency_key": row.rank_idempotency_key,
        "group_id": row.group_id,
    }


def _dossier_from_dict_v2(row: dict[str, Any]) -> Dossier:
    return Dossier(
        orphan_id=_safe_text(row.get("unique_id") or row.get("orphan_id")),
        article_id=(
            _as_int(row.get("article_id"), 0)
            if row.get("article_id") is not None
            else None
        ),
        city_id=_safe_text(row.get("city_id")),
        year=_as_int(row.get("year"), 0) if row.get("year") is not None else None,
        month=_as_int(row.get("month"), 0) if row.get("month") is not None else None,
        midpoint_day=(
            _as_float(row.get("midpoint_day"), 0.0)
            if row.get("midpoint_day") is not None
            else None
        ),
        date_precision=_safe_text(row.get("date_precision")),
        incident_date=_safe_text(row.get("incident_date")),
        victim_count=_safe_text(row.get("victim_count")),
        weapon=_safe_text(row.get("weapon")),
        circumstance=_safe_text(row.get("circumstance")),
        geo_address_norm=_safe_text(row.get("geo_address_norm")),
        geo_address_short=_safe_text(row.get("geo_address_short")),
        geo_address_short_2=_safe_text(row.get("geo_address_short_2")),
        relationship=_safe_text(row.get("relationship")),
        incident_summary_gpt=_safe_text(row.get("incident_summary_gpt")),
        article_title=_safe_text(row.get("article_title")),
        article_text=_safe_text(row.get("article_text")),
        article_pub_date=_safe_text(row.get("article_pub_date")),
    )


def _dossier_to_dict_v2(dossier: Dossier) -> dict[str, Any]:
    return {
        "unique_id": dossier.orphan_id,
        "orphan_id": dossier.orphan_id,
        "article_id": dossier.article_id,
        "city_id": dossier.city_id,
        "year": dossier.year,
        "month": dossier.month,
        "midpoint_day": dossier.midpoint_day,
        "date_precision": dossier.date_precision,
        "incident_date": dossier.incident_date,
        "victim_count": dossier.victim_count,
        "weapon": dossier.weapon,
        "circumstance": dossier.circumstance,
        "geo_address_norm": dossier.geo_address_norm,
        "geo_address_short": dossier.geo_address_short,
        "geo_address_short_2": dossier.geo_address_short_2,
        "relationship": dossier.relationship,
        "incident_summary_gpt": dossier.incident_summary_gpt,
        "article_title": dossier.article_title,
        "article_text": dossier.article_text,
        "article_pub_date": dossier.article_pub_date,
    }


def _validated_anchor_to_dict_v2(anchor: ValidatedAnchor) -> dict[str, Any]:
    return anchor.to_dict()


def _c2_candidate_to_dict_v2(candidate: C2Candidate) -> dict[str, Any]:
    return candidate.to_dict()


def _merged_candidate_from_dict_v2(row: dict[str, Any]) -> MergedCandidate:
    raw_rows = row.get("rows")
    row_entries = (
        tuple(r for r in raw_rows if isinstance(r, dict))
        if isinstance(raw_rows, list)
        else tuple()
    )
    return MergedCandidate(
        entity_uid=_safe_text(row.get("entity_uid")),
        det_score=_as_float(row.get("det_score"), 0.0),
        midpoint_day=(
            _as_float(row.get("midpoint_day"), 0.0)
            if row.get("midpoint_day") is not None
            else None
        ),
        day_gap=_as_float(row.get("day_gap"), 9999.0),
        compat_count=_as_int(row.get("compat_count"), 0),
        summary_cosine=_as_float(row.get("summary_cosine"), 0.0),
        splink_match_weight=(
            _as_float(row.get("splink_match_weight"), 0.0)
            if row.get("splink_match_weight") is not None
            else None
        ),
        anchor_hit_count=_as_int(row.get("anchor_hit_count"), 0),
        source_stages=tuple(_safe_text(v) for v in (row.get("source_stages") or [])),
        query_variant=_safe_text(row.get("query_variant")),
        query_variants=tuple(_safe_text(v) for v in (row.get("query_variants") or [])),
        source_article_id=(
            _as_int(row.get("source_article_id"), 0)
            if row.get("source_article_id") is not None
            else None
        ),
        source_article_ids=tuple(
            int(v)
            for v in (row.get("source_article_ids") or [])
            if _safe_text(v).strip().isdigit()
        ),
        rows=row_entries,
    )


def _ensure_tables_run_v2() -> Run[Any]:
    statements = Array.make(
        (
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_run (
              run_id VARCHAR PRIMARY KEY,
              started_at TIMESTAMP,
              finished_at TIMESTAMP,
              requested_limit INTEGER,
              processed_count INTEGER,
              grouped_count INTEGER,
              matched_count INTEGER,
              not_same_person_count INTEGER,
              insufficient_information_count INTEGER,
              analysis_incomplete_count INTEGER,
              dry_run BOOLEAN,
              full_backfill BOOLEAN,
              status VARCHAR,
              error_message VARCHAR
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_case_state (
              run_id VARCHAR,
              group_id VARCHAR,
              orphan_id VARCHAR,
              article_id BIGINT,
              case_status VARCHAR,
              stage_completed VARCHAR,
              decision_label VARCHAR,
              resolved_entity_id VARCHAR,
              decision_hash VARCHAR,
              error_message VARCHAR,
              updated_at TIMESTAMP,
              PRIMARY KEY (run_id, orphan_id)
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_queue_run (
              run_id VARCHAR,
              queue_pos INTEGER,
              group_id VARCHAR,
              orphan_id VARCHAR,
              article_id BIGINT,
              city_id BIGINT,
              year BIGINT,
              month BIGINT,
              midpoint_day DOUBLE,
              created_at TIMESTAMP DEFAULT NOW()
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_cache_readiness (
              run_id VARCHAR,
              queue_pos INTEGER,
              orphan_id VARCHAR,
              article_id BIGINT,
              year BIGINT,
              readiness_status VARCHAR,
              readiness_reason VARCHAR,
              pass1_idempotency_key VARCHAR,
              rank_idempotency_key VARCHAR,
              created_at TIMESTAMP DEFAULT NOW()
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_candidates_bc (
              run_id VARCHAR,
              group_id VARCHAR,
              orphan_id VARCHAR,
              stage_name VARCHAR,
              entity_uid VARCHAR,
              det_score DOUBLE,
              features_json JSON,
              created_at TIMESTAMP DEFAULT NOW()
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_candidates_c2 (
              run_id VARCHAR,
              group_id VARCHAR,
              orphan_id VARCHAR,
              entity_uid VARCHAR,
              source_article_id BIGINT,
              query_variant VARCHAR,
              det_score DOUBLE,
              splink_match_weight DOUBLE,
              features_json JSON,
              created_at TIMESTAMP DEFAULT NOW()
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_candidates_merged (
              run_id VARCHAR,
              group_id VARCHAR,
              orphan_id VARCHAR,
              entity_uid VARCHAR,
              det_score DOUBLE,
              splink_match_weight DOUBLE,
              source_stages VARCHAR,
              features_json JSON,
              created_at TIMESTAMP DEFAULT NOW()
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adj_stage_metrics (
              run_id VARCHAR,
              group_id VARCHAR,
              orphan_id VARCHAR,
              stage_name VARCHAR,
              query_id VARCHAR,
              row_count BIGINT,
              notes VARCHAR,
              recorded_at TIMESTAMP DEFAULT NOW()
            );
            """
            ),
            SQL(
                "ALTER TABLE orphan_adj_candidates_c2 ADD COLUMN IF NOT EXISTS"
                " splink_match_weight DOUBLE;"
            ),
            SQL(
                "ALTER TABLE orphan_adj_candidates_merged ADD COLUMN IF NOT EXISTS"
                " splink_match_weight DOUBLE;"
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS llm_cache (
              stage VARCHAR,
              idempotency_key VARCHAR,
              model VARCHAR,
              prompt_version VARCHAR,
              input_json JSON,
              response_json JSON,
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW(),
              PRIMARY KEY(stage, idempotency_key)
            );
            """
            ),
            SQL(
                """
            CREATE TABLE IF NOT EXISTS orphan_adjudication_history (
              history_id BIGINT,
              run_id VARCHAR,
              orphan_id VARCHAR,
              prior_resolution_label VARCHAR,
              prior_resolved_entity_id VARCHAR,
              prior_confidence DOUBLE,
              new_resolution_label VARCHAR,
              new_resolved_entity_id VARCHAR,
              new_confidence DOUBLE,
              reason_summary VARCHAR,
              evidence_json JSON,
              analyst_mode VARCHAR,
              changed_at TIMESTAMP
            );
            """
            ),
        )
    )
    return (
        array_traverse_run(statements, sql_exec)
        if statements.length > 0
        else pure(Array.empty())
    ) >> (lambda _: pure(None))


def _migrate_legacy_labels_run_v2() -> Run[Any]:
    statements = Array.make(
        (
            SQL(
                """
            UPDATE orphan_adjudication_overrides
            SET resolution_label = 'matched',
                updated_at = NOW()
            WHERE resolution_label = 'likely_missed_match';
            """
            ),
            SQL(
                """
            UPDATE orphan_adjudication_overrides
            SET resolution_label = 'not_same_person',
                updated_at = NOW()
            WHERE resolution_label = 'unlikely';
            """
            ),
            SQL(
                """
            UPDATE orphan_adjudication_overrides
            SET resolution_label = 'analysis_incomplete',
                updated_at = NOW()
            WHERE resolution_label = 'possible_but_weak';
            """
            ),
        )
    )
    return (
        array_traverse_run(statements, sql_exec)
        if statements.length > 0
        else pure(Array.empty())
    ) >> (lambda _: pure(None))


def _query_rows_run_v2(
    sql: str, params: SQLParams = SQLParams(()), *, sqlite: bool = False
) -> Run[list[dict[str, Any]]]:
    prog = sql_query(SQL(sql), params) >> (lambda rows: pure(_rows_to_dicts(rows)))
    return _with_sqlite(prog) if sqlite else prog


def _queue_universe_rows_run_v2() -> Run[Array[CacheReadinessRow]]:
    return (
        _query_rows_run_v2(
            """
        WITH queue_base AS (
          SELECT
            o.uid AS orphan_id,
            o.article_id,
            o.city_id,
            o.year,
            o.month,
            o.midpoint_day,
            o.weapon,
            o.circumstance,
            o.match_id
          FROM orphan_matches_final_current o
          WHERE o.rec_type = 'orphan'
            AND o.match_id LIKE 'orphan_%'
        )
        SELECT
          qb.*,
          ROW_NUMBER() OVER (
            ORDER BY
              qb.midpoint_day NULLS LAST,
              qb.orphan_id
          ) AS queue_pos
        FROM queue_base qb
        ORDER BY queue_pos;
        """
        )
        >> (
            lambda rows: pure(
                Array.make(
                    tuple(_cache_readiness_row_from_dict_v2(row) for row in rows)
                )
            )
        )
    )


def _load_orphan_dossier_run_v2(orphan_id: str) -> Run[Dossier]:
    """Load the normalized dossier and article context for one orphan ID."""

    def _load_article_context_for_dossier(base: dict[str, Any]) -> Run[Dossier]:
        def _build_dossier(summary_rows: list[dict[str, Any]],
                           article_rows: list[dict[str, Any]]) -> Dossier:
            return _dossier_from_dict_v2(
                {
                    **base,
                    "incident_summary_gpt": (
                        _safe_text(summary_rows[0].get("summary"))
                        if len(summary_rows) > 0
                        else ""
                    ),
                    "article_title": (
                        _safe_text(article_rows[0].get("Title"))
                        if len(article_rows) > 0
                        else ""
                    ),
                    "article_text": (
                        _safe_text(article_rows[0].get("FullText"))[
                            :MAX_FULLTEXT_CHARS
                        ]
                        if len(article_rows) > 0
                        else ""
                    ),
                    "article_pub_date": (
                        _safe_text(article_rows[0].get("PubDate"))
                        if len(article_rows) > 0
                        else ""
                    ),
                }
            )

        def _load_article_rows(summary_rows: list[dict[str, Any]]) -> Run[Dossier]:
            return (
                _query_rows_run_v2(
                    """
                    SELECT Title, FullText, PubDate
                    FROM articles
                    WHERE RecordId = ?
                      AND Dataset = 'CLASS_WP'
                      AND gptClass = 'M'
                    LIMIT 1;
                    """,
                    _sql_params((base.get("article_id"),)),
                    sqlite=True,
                )
                if base.get("article_id") is not None
                else pure([])
            ) >> (
                lambda article_rows: pure(_build_dossier(summary_rows, article_rows))
            )

        return (
            (
                _query_rows_run_v2(
                    """
                    SELECT summary
                    FROM incidents_cached
                    WHERE article_id = ?
                      AND incident_idx = ?
                    LIMIT 1;
                    """,
                    _sql_params(
                        (
                            base.get("article_id"),
                            _incident_idx_from_orphan_uid(orphan_id),
                        )
                    ),
                )
                if base.get("article_id") is not None
                and _incident_idx_from_orphan_uid(orphan_id) is not None
                else pure([])
            )
            >> _load_article_rows
        )

    return (
        _query_rows_run_v2(
            """
        SELECT
          unique_id,
          article_id,
          city_id,
          year,
          month,
          midpoint_day,
          date_precision,
          incident_date,
          victim_count,
          weapon,
          circumstance,
          geo_address_norm,
          geo_address_short,
          geo_address_short_2,
          relationship
        FROM orphan_link_input
        WHERE unique_id = ?
        LIMIT 1;
        """,
            _sql_params((orphan_id,)),
        )
        >> (
            lambda rows: throw(ErrorPayload(f"orphan_dossier_missing:{orphan_id}"))
            if len(rows) == 0
            else _load_article_context_for_dossier(rows[0])
        )
    )


def _build_pass1_request_v2(dossier: Dossier) -> Pass1Request:
    return Pass1Request(
        article_title=dossier.article_title,
        article_text=dossier.article_text,
        article_date=_format_article_date(dossier.article_pub_date),
        incident_date=_format_incident_date(
            dossier.incident_date, year=dossier.year, month=dossier.month
        ),
        incident_location=dossier.geo_address_norm,
        victim_count=dossier.victim_count,
        incident_summary=dossier.incident_summary_gpt,
    )


def _adjudication_cache_key_v2(dossier: Dossier) -> str:
    return dossier.orphan_id


def _llm_cache_payload_from_rows_v2(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(rows) == 0:
        return None
    raw = rows[0].get("response_json")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _llm_cache_get_run_v2(
    stage: str, idempotency_key: str
) -> Run[dict[str, Any] | None]:
    return (
        _query_rows_run_v2(
            """
        SELECT response_json
        FROM llm_cache
        WHERE stage = ?
          AND idempotency_key = ?
        LIMIT 1;
        """,
            _sql_params((stage, idempotency_key)),
        )
        >> (lambda rows: pure(_llm_cache_payload_from_rows_v2(rows)))
    )


def _llm_cache_put_run_v2(
    *,
    stage: str,
    idempotency_key: str,
    model: str,
    prompt_version: str,
    input_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> Run[Any]:
    return sql_exec(
        SQL(
            """
            INSERT INTO llm_cache (
              stage, idempotency_key, model, prompt_version, input_json,
              response_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?::JSON, ?::JSON, NOW(), NOW())
            ON CONFLICT(stage, idempotency_key) DO UPDATE SET
              model = EXCLUDED.model,
              prompt_version = EXCLUDED.prompt_version,
              input_json = EXCLUDED.input_json,
              response_json = EXCLUDED.response_json,
              updated_at = NOW();
            """
        ),
        _sql_params(
            (
                stage,
                idempotency_key,
                model,
                prompt_version,
                _canonical_json(input_payload),
                _canonical_json(response_payload),
            )
        ),
    )


def _materialize_cache_readiness_run_v2(
    run_id: str, rows: Array[CacheReadinessRow]
) -> Run[Any]:
    def _insert_one(row: CacheReadinessRow) -> Run[Any]:
        return sql_exec(
            SQL(
                """
                INSERT INTO orphan_adj_cache_readiness (
                  run_id, queue_pos, orphan_id, article_id, year,
                  readiness_status, readiness_reason, pass1_idempotency_key,
                  rank_idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW());
                """
            ),
            _sql_params(
                (
                    run_id,
                    row.queue_pos,
                    row.orphan_id,
                    row.article_id,
                    row.year,
                    row.readiness_status,
                    row.readiness_reason,
                    row.pass1_idempotency_key,
                    row.rank_idempotency_key,
                )
            ),
        )

    def _insert_readiness_rows(_: Any) -> Run[Array[Any]]:
        return (
            array_traverse_run(rows, _insert_one)
            if rows.length > 0
            else pure(Array.empty())
        )

    return (
        sql_exec(
            SQL("DELETE FROM orphan_adj_cache_readiness WHERE run_id = ?;"),
            _sql_params((run_id,)),
        )
        >> _insert_readiness_rows
        >> (lambda _: pure(None))
    )


def _compute_cache_readiness_row_run_v2(
    base: CacheReadinessRow,
) -> Run[CacheReadinessRow]:
    cache_key = base.orphan_id
    return _llm_cache_get_run_v2(E2E_CACHE_STAGE, cache_key) >> (
        lambda cached: pure(
            base.with_readiness(
                readiness_status=(
                    "decision_ready_from_cache" if cached is not None else "needs_api"
                ),
                readiness_reason=(
                    "e2e_cache_ready" if cached is not None else "missing_e2e_cache"
                ),
                pass1_idempotency_key=cache_key,
                rank_idempotency_key="",
            )
        )
    )


def _precompute_cache_readiness_run() -> Run[tuple[Array[CacheReadinessRow], int]]:
    def _compute_readiness(
        rows: Array[CacheReadinessRow],
    ) -> Run[Array[CacheReadinessRow]]:
        return (
            array_traverse_run(rows, _compute_cache_readiness_row_run_v2)
            if rows.length > 0
            else pure(Array.empty())
        )

    def _persist_readiness_preview(
        readiness: Array[CacheReadinessRow],
    ) -> Run[tuple[Array[CacheReadinessRow], int]]:
        return _materialize_cache_readiness_run_v2(
            f"k_preview_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            readiness,
        ) ^ pure(
            (
                readiness,
                sum(1 for row in readiness if row.readiness_status == "needs_api"),
            )
        )

    return (
        _ensure_tables_run_v2()
        >> (lambda _: _queue_universe_rows_run_v2())
        >> _compute_readiness
        >> _persist_readiness_preview
    )


def _format_cache_readiness_summary_v2(rows: Array[CacheReadinessRow]) -> str:
    return _format_cache_readiness_summary(
        [_cache_readiness_row_to_dict_v2(row) for row in rows]
    )


def _select_needs_api_rows_v2(
    rows: Array[CacheReadinessRow], params: KParams
) -> Array[CacheReadinessRow]:
    selected = _select_needs_api_rows(
        [_cache_readiness_row_to_dict_v2(row) for row in rows], params
    )
    return Array.make(tuple(_cache_readiness_row_from_dict_v2(row) for row in selected))


def _build_groups_v2(
    rows: Array[CacheReadinessRow], group_same_incident: bool, run_id: str
) -> Array[Array[CacheReadinessRow]]:
    groups = _build_groups(
        [_cache_readiness_row_to_dict_v2(row) for row in rows],
        group_same_incident,
        run_id,
    )
    return Array.make(
        tuple(
            Array.make(tuple(_cache_readiness_row_from_dict_v2(row) for row in group))
            for group in groups
        )
    )


def _insert_queue_rows_run_v2(
    run_id: str, groups: Array[Array[CacheReadinessRow]]
) -> Run[Any]:
    def _insert_group(group: Array[CacheReadinessRow]) -> Run[Array[Any]]:
        def _insert_one(row: CacheReadinessRow) -> Run[Any]:
            return (
                sql_exec(
                    SQL(
                        """
                    INSERT INTO orphan_adj_queue_run (
                      run_id, queue_pos, group_id, orphan_id, article_id,
                      city_id, year, month, midpoint_day
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """
                    ),
                    _sql_params(
                        (
                            run_id,
                            row.queue_pos,
                            row.group_id,
                            row.orphan_id,
                            row.article_id,
                            row.city_id if row.city_id != "" else None,
                            row.year,
                            row.month,
                            row.midpoint_day,
                        )
                    ),
                )
                ^ sql_exec(
                    SQL(
                        """
                    INSERT INTO orphan_adj_case_state (
                      run_id, group_id, orphan_id, article_id,
                      case_status, stage_completed, decision_label, resolved_entity_id,
                      decision_hash, error_message, updated_at
                    ) VALUES (?, ?, ?, ?, 'queued', '', NULL, NULL, NULL, NULL, NOW())
                    ON CONFLICT(run_id, orphan_id) DO UPDATE SET
                      group_id = EXCLUDED.group_id,
                      article_id = EXCLUDED.article_id,
                      case_status = 'queued',
                      stage_completed = '',
                      error_message = NULL,
                      updated_at = NOW();
                    """
                    ),
                    _sql_params((run_id, row.group_id, row.orphan_id, row.article_id)),
                )
            )

        return (
            array_traverse_run(group, _insert_one)
            if group.length > 0
            else pure(Array.empty())
        )

    return (
        array_traverse_run(groups, _insert_group)
        if groups.length > 0
        else pure(Array.empty())
    ) >> (lambda _: pure(None))


def _set_case_state_run_v2(
    run_id: str,
    orphan_id: str,
    *,
    case_status: str,
    stage_completed: str,
    decision_label: str | None = None,
    resolved_entity_id: str | None = None,
    decision_hash: str | None = None,
    error_message: str | None = None,
) -> Run[Any]:
    return sql_exec(
        SQL(
            """
            UPDATE orphan_adj_case_state
            SET case_status = ?,
                stage_completed = ?,
                decision_label = ?,
                resolved_entity_id = ?,
                decision_hash = ?,
                error_message = ?,
                updated_at = NOW()
            WHERE run_id = ?
              AND orphan_id = ?;
            """
        ),
        _sql_params(
            (
                case_status,
                stage_completed,
                decision_label,
                resolved_entity_id,
                decision_hash,
                error_message,
                run_id,
                orphan_id,
            )
        ),
    )


def _log_stage_metric_run_v2(
    run_id: str,
    group_id: str,
    orphan_id: str,
    stage_name: str,
    query_id: str,
    row_count: int,
    notes: str,
) -> Run[Any]:
    return sql_exec(
        SQL(
            """
            INSERT INTO orphan_adj_stage_metrics (
              run_id, group_id, orphan_id, stage_name, query_id, row_count,
              notes, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NOW());
            """
        ),
        _sql_params(
            (run_id, group_id, orphan_id, stage_name, query_id, row_count, notes)
        ),
    )


def _display_article_run_v2(dossier: Dossier) -> Run[Any]:
    article_text_display = (
        dossier.article_text if dossier.article_text != "" else "[empty]"
    )
    fallback = (
        put_line(
            f"[K][{dossier.orphan_id}]"
            " article_id="
            f"{dossier.article_id} title={dossier.article_title}"
        )
        ^ put_line(f"[K][{dossier.orphan_id}] full_text:\n{article_text_display}")
    )
    return _display_article_by_id_run_v2(
        dossier.orphan_id,
        dossier.article_id,
        empty_message="No article_id available for display.",
        fallback=fallback,
    )


def _display_article_by_id_run_v2(
    orphan_id: str,
    article_id: int | None,
    *,
    empty_message: str,
    fallback: Run[Any] | None = None,
) -> Run[Any]:
    if article_id is None:
        return put_line(f"[K][{orphan_id}] {empty_message}")

    def _render_article_row(rows: list[dict[str, Any]]) -> Run[Any]:
        if len(rows) == 0:
            return fallback if fallback is not None else put_line(
                f"[K][{orphan_id}] Article {article_id} not found for display."
            )
        return run_except(
            pure(dict(rows[0])).map(lambda row: Article(row, current=0, total=1))
        ) >> _render_display_article_result

    def _render_display_article_result(result: Any) -> Run[Any]:
        return (
            put_line(f"[K][{orphan_id}] Retrieved article:\n {result.r}")
            if isinstance(result, Right)
            else (
                put_line(
                    f"[K][{orphan_id}] display_article render"
                    f" failed: {_safe_text(result.l)}"
                )
                ^ (
                    fallback
                    if fallback is not None
                    else put_line(
                        f"[K][{orphan_id}] Article {article_id} could not be"
                        " displayed from stored data."
                    )
                )
            )
        )

    return (
        _query_rows_run_v2(
            """
        SELECT
          RecordId,
          Title,
          Publication,
          PubDate,
          FullText,
          Status AS status,
          AssignStatus AS assignstatus,
          gptClass AS GPTClass,
          gptVictimJson,
          Notes
        FROM articles
        WHERE RecordId = ?
          AND Dataset = 'CLASS_WP'
          AND gptClass = 'M'
        LIMIT 1;
        """,
            _sql_params((article_id,)),
            sqlite=True,
        )
        >> _render_article_row
    )


def _matched_article_id_from_decision(decision: CaseDecision) -> int | None:
    narrative = decision.evidence_json.get("narrative_evidence")
    if not isinstance(narrative, dict):
        return None
    candidate_anchors = narrative.get("candidate_anchors")
    if not isinstance(candidate_anchors, list) or len(candidate_anchors) == 0:
        return None
    first_candidate = candidate_anchors[0]
    if not isinstance(first_candidate, dict):
        return None
    raw = first_candidate.get("source_article_id")
    return int(raw) if isinstance(raw, int) else _int_or_none(raw)


def _display_matched_article_run_v2(decision: CaseDecision) -> Run[Any]:
    matched_article_id = _matched_article_id_from_decision(decision)
    if decision.label != "matched" or matched_article_id is None:
        return pure(None)
    return put_line(f"[K][{decision.orphan_id}] Matched article:") ^ (
        _display_article_by_id_run_v2(
            decision.orphan_id,
            matched_article_id,
            empty_message="Matched article_id missing for display.",
        )
    )


def _log_api_usage_and_reasoning_run_v2(
    orphan_id: str, stage: str, response_t: GPTResponseTuple
) -> Run[Any]:
    return (
        put_line(f"[K][{orphan_id}] {stage} usage/cost:")
        ^ put_line(str(response_t.parsed.usage))
        ^ put_line(f"[K][{orphan_id}] {stage} GPT reasoning summary:")
        ^ put_line(str(response_t.parsed.reasoning))
    )


def _log_pass1_api_input_run_v2(orphan_id: str, payload: Pass1Request) -> Run[Any]:
    return put_line(f"[K][{orphan_id}] pass1_input_variables:") ^ put_line(
        _pretty_json(payload.to_display_payload())
    )


def _log_rank_api_input_run_v2(orphan_id: str, payload: RankRequest) -> Run[Any]:
    return put_line(f"[K][{orphan_id}] rank_input_variables:") ^ put_line(
        _pretty_json(payload.to_display_payload())
    )


def _require_gpt_tuple_v2(result: Any) -> Run[GPTResponseTuple]:
    return (
        pure(result.r)
        if isinstance(result, Right)
        else throw(ErrorPayload(f"{result.l}"))
    )


def _call_pass1_run_v2(request: Pass1Request) -> Run[GPTResponseTuple]:
    return (
        to_gpt_tuple
        & response_with_gpt_prompt(
            PromptKey(PASS1_PROMPT_KEY),
            cast(dict[str, str | None], request.to_prompt_variables()),
            Pass1Response,
            PASS1_MODEL_KEY,
            effort="low",
            stream=False,
        )
    ) >> _require_gpt_tuple_v2


def _call_rank_run_v2(request: RankRequest) -> Run[GPTResponseTuple]:
    return (
        to_gpt_tuple
        & response_with_gpt_prompt(
            PromptKey(RANK_PROMPT_KEY),
            cast(dict[str, str | None], request.to_prompt_variables()),
            RankResponse,
            RANK_MODEL_KEY,
            effort="low",
            stream=False,
        )
    ) >> _require_gpt_tuple_v2


def _anchor_doc_frequency_run_v2(anchor_text: str) -> Run[int]:
    if anchor_text.strip() == "":
        return pure(0)
    return (
        _query_rows_run_v2(
            """
        SELECT COUNT(*) AS n
        FROM articles
        WHERE Dataset='CLASS_WP'
          AND gptClass='M'
          AND FullText LIKE ?;
        """,
            _sql_params((f"%{anchor_text.strip()}%",)),
            sqlite=True,
        )
        >> (lambda rows: pure(int(rows[0].get("n") or 0) if len(rows) > 0 else 0))
    )


def _validate_anchors_run_v2(
    pass1: Pass1Response,
) -> Run[tuple[Array[ValidatedAnchor], Array[str], list[str]]]:
    anchor_array = Array.make(tuple(pass1.anchors))
    init: tuple[list[ValidatedAnchor], list[str], list[str]] = ([], [], [])

    def _step(
        acc: tuple[list[ValidatedAnchor], list[str], list[str]], anchor: Anchor
    ) -> Run[tuple[list[ValidatedAnchor], list[str], list[str]]]:
        valid, variants, failures = acc
        anchor_text = _safe_text(anchor.anchor_theme).strip()
        if _norm(anchor_text) == "":
            return pure((valid, variants, failures + ["empty_anchor"]))
        raw_sets = [
            tuple(
                _safe_text(word).strip()
                for word in word_set.root
                if _safe_text(word).strip() != ""
            )
            for word_set in anchor.variants
        ]
        deduped_sets = [
            word_set for word_set in dict.fromkeys(raw_sets) if len(word_set) > 0
        ]
        filtered_variants = [
            " ".join(word_set).strip()
            for word_set in deduped_sets
            if _norm(" ".join(word_set).strip()) != ""
        ]
        def _build_anchor_result(
            doc_freq: int,
        ) -> tuple[list[ValidatedAnchor], list[str], list[str]]:
            if doc_freq > HIGH_FREQ_ANCHOR_DOC_THRESHOLD:
                return (
                    valid,
                    variants,
                    failures + [f"high_freq_anchor:{anchor_text[:40]}:{doc_freq}"],
                )
            return (
                valid
                + [
                    ValidatedAnchor(
                        anchor_text=anchor_text,
                        variants=tuple(
                            filtered_variants
                            if len(filtered_variants) > 0
                            else [anchor_text]
                        ),
                        variant_word_sets=tuple(
                            deduped_sets
                            if len(filtered_variants) > 0
                            else [(anchor_text,)]
                        ),
                        doc_freq=doc_freq,
                    )
                ],
                variants
                + (
                    filtered_variants
                    if len(filtered_variants) > 0
                    else [anchor_text]
                ),
                failures,
            )
        return _anchor_doc_frequency_run_v2(anchor_text) >> (
            lambda doc_freq: pure(_build_anchor_result(doc_freq))
        )

    def _finalize_anchor_validation(
        result: tuple[list[ValidatedAnchor], list[str], list[str]]
    ) -> Run[tuple[Array[ValidatedAnchor], Array[str], list[str]]]:
        return pure(
            (
                Array.make(tuple(result[0])),
                Array.make(tuple(dict.fromkeys(result[1]))),
                result[2]
                + (["no_valid_anchors"] if len(result[0]) < 1 else [])
                + (["no_query_variants"] if len(result[1]) < 1 else []),
            )
        )

    return (
        fold_run(anchor_array, init, _step) if anchor_array.length > 0 else pure(init)
    ) >> _finalize_anchor_validation


def _fts_article_hits_query_run_v2(match_q: str, limit_n: int) -> Run[Array[int]]:
    return (
        _query_rows_run_v2(
            """
        SELECT a.RecordId
        FROM articles_wp_m_fts f
        JOIN articles a
          ON a.RecordId = f.rowid
        WHERE a.Dataset='CLASS_WP'
          AND a.gptClass='M'
          AND articles_wp_m_fts MATCH ?
        LIMIT ?;
        """,
            _sql_params((match_q, limit_n)),
            sqlite=True,
        )
        >> (
            lambda rows: pure(
                Array.make(
                    tuple(
                        sorted(
                            dict.fromkeys(
                                int(row["RecordId"])
                                for row in rows
                                if row.get("RecordId") is not None
                            )
                        )
                    )
                )
            )
        )
    )


def _fts_article_hits_run_v2(fts_query: str, limit_n: int = 60) -> Run[Array[int]]:
    if fts_query.strip() == "":
        return pure(Array.empty())

    def _handle_fts_query_result(result: Any) -> Run[Array[int]]:
        return pure(result.r) if isinstance(result, Right) else pure(Array.empty())

    return run_except(_fts_article_hits_query_run_v2(fts_query, limit_n)) >> (
        _handle_fts_query_result
    )


def _entities_for_articles_run_v2(
    article_ids: Array[int], orphan_id: str, query_variant: str
) -> Run[Array[C2Candidate]]:
    if article_ids.length == 0:
        return pure(Array.empty())
    values_sql = ",".join("(?)" for _ in article_ids)
    params: tuple[Any, ...] = (*article_ids.a, orphan_id)
    return (
        _query_rows_run_v2(
            f"""
        WITH hits(article_id) AS (
          SELECT * FROM (VALUES {values_sql})
        ),
        o AS (
          SELECT *
          FROM orphan_link_input
          WHERE unique_id = ?
        )
        SELECT DISTINCT
          e.unique_id AS entity_uid,
          h.article_id AS source_article_id,
          e.midpoint_day AS midpoint_day,
          ABS(COALESCE(CAST(e.midpoint_day AS DOUBLE), 0) - COALESCE(o.midpoint_day, 0)) AS day_gap,
          CASE WHEN e.victim_sex = o.victim_sex THEN 1 ELSE 0 END AS sex_match,
          CASE WHEN e.weapon = o.weapon THEN 1 ELSE 0 END AS weapon_match,
          CASE WHEN e.circumstance = o.circumstance THEN 1 ELSE 0 END AS circumstance_match,
          array_cosine_similarity(e.summary_vec, o.summary_vec) AS summary_cosine
        FROM hits h
        JOIN entity_link_input e
          ON list_contains(string_split(COALESCE(e.article_ids_csv, ''), ','), CAST(h.article_id AS VARCHAR))
        CROSS JOIN o
        WHERE NOT list_contains(
          string_split(COALESCE(e.article_ids_csv, ''), ','),
          CAST(o.article_id AS VARCHAR)
        );
        """,
            SQLParams(params),
        )
        >> (
            lambda rows: pure(
                Array.make(
                    tuple(
                        C2Candidate(
                            entity_uid=_safe_text(row.get("entity_uid")),
                            source_article_id=(
                                _as_int(row.get("source_article_id"), 0)
                                if row.get("source_article_id") is not None
                                else None
                            ),
                            source_incident_idx=None,
                            midpoint_day=(
                                _as_float(row.get("midpoint_day"), 0.0)
                                if row.get("midpoint_day") is not None
                                else None
                            ),
                            day_gap=_as_float(row.get("day_gap"), 9999.0),
                            compat_count=_as_int(row.get("sex_match"), 0)
                            + _as_int(row.get("weapon_match"), 0)
                            + _as_int(row.get("circumstance_match"), 0),
                            summary_cosine=_as_float(row.get("summary_cosine"), 0.0),
                            det_score=1.2
                            + 0.8
                            * (
                                _as_int(row.get("sex_match"), 0)
                                + _as_int(row.get("weapon_match"), 0)
                                + _as_int(row.get("circumstance_match"), 0)
                            )
                            + 1.7 * _as_float(row.get("summary_cosine"), 0.0)
                            - 0.01 * _as_float(row.get("day_gap"), 9999.0),
                            query_variant=query_variant,
                        )
                        for row in rows
                    )
                )
            )
        )
    )


def _entities_for_articles_group_run_v2(
    article_ids: Array[int], target_article_id: int | None, query_variant: str
) -> Run[Array[C2Candidate]]:
    if article_ids.length == 0:
        return pure(Array.empty())
    values_sql = ",".join("(?)" for _ in article_ids)
    params_values: tuple[Any, ...]
    exclusion_sql = ""
    if target_article_id is None:
        params_values = article_ids.a
    else:
        exclusion_sql = (
            "WHERE NOT list_contains("
            "  string_split(COALESCE(e.article_ids_csv, ''), ','),"
            "  CAST(? AS VARCHAR)"
            ")"
        )
        params_values = (*article_ids.a, target_article_id)
    return (
        _query_rows_run_v2(
            f"""
        WITH hits(article_id) AS (
          SELECT * FROM (VALUES {values_sql})
        ),
        incident_victims AS (
          SELECT
            h.article_id AS source_article_id,
            i.incident_idx AS source_incident_idx,
            v.victim_row_id
          FROM hits h
          JOIN incidents_cached i
            ON i.article_id = h.article_id
          JOIN victims_cached_enh v
            ON v.article_id = i.article_id
           AND v.incident_idx = i.incident_idx
        ),
        incident_entities AS (
          SELECT DISTINCT
            iv.source_article_id,
            iv.source_incident_idx,
            m.victim_entity_id AS entity_uid
          FROM incident_victims iv
          JOIN victim_entity_members m
            ON m.victim_row_id = iv.victim_row_id
          UNION
          SELECT DISTINCT
            iv.source_article_id,
            iv.source_incident_idx,
            CAST(fom.entity_uid AS VARCHAR) AS entity_uid
          FROM incident_victims iv
          JOIN final_orphan_matches fom
            ON CAST(fom.orphan_uid AS VARCHAR) = iv.victim_row_id
        )
        SELECT DISTINCT
          ie.entity_uid,
          ie.source_article_id,
          ie.source_incident_idx,
          e.midpoint_day AS midpoint_day
        FROM incident_entities ie
        JOIN entity_link_input e
          ON e.unique_id = ie.entity_uid
        {exclusion_sql};
        """,
            _sql_params(params_values),
        )
        >> (
            lambda rows: pure(
                Array.make(
                    tuple(
                        C2Candidate(
                            entity_uid=_safe_text(row.get("entity_uid")),
                            source_article_id=(
                                _as_int(row.get("source_article_id"), 0)
                                if row.get("source_article_id") is not None
                                else None
                            ),
                            source_incident_idx=(
                                _as_int(row.get("source_incident_idx"), 0)
                                if row.get("source_incident_idx") is not None
                                else None
                            ),
                            midpoint_day=(
                                _as_float(row.get("midpoint_day"), 0.0)
                                if row.get("midpoint_day") is not None
                                else None
                            ),
                            day_gap=0.0,
                            compat_count=0,
                            summary_cosine=0.0,
                            det_score=0.0,
                            query_variant=query_variant,
                        )
                        for row in rows
                    )
                )
            )
        )
    )


def _score_group_candidates_for_orphan_run_v2(
    orphan_id: str, candidates: Array[C2Candidate]
) -> Run[Array[C2Candidate]]:
    if candidates.length == 0:
        return pure(Array.empty())
    values_sql = ",".join("(?, ?, ?, ?, ?)" for _ in candidates)
    params: list[Any] = []
    for candidate in candidates:
        params.extend(
            (
                candidate.entity_uid,
                candidate.source_article_id,
                candidate.source_incident_idx,
                candidate.query_variant,
                candidate.midpoint_day,
            )
        )
    params.append(orphan_id)

    def _attach_weights(
        scored_candidates: Array[C2Candidate],
    ) -> Run[Array[C2Candidate]]:
        def _finalize(weights: dict[str, float]) -> Run[Array[C2Candidate]]:
            return pure(
                Array.make(
                    tuple(
                        C2Candidate(
                            entity_uid=candidate.entity_uid,
                            source_article_id=candidate.source_article_id,
                            source_incident_idx=candidate.source_incident_idx,
                            midpoint_day=candidate.midpoint_day,
                            day_gap=candidate.day_gap,
                            compat_count=candidate.compat_count,
                            summary_cosine=candidate.summary_cosine,
                            det_score=candidate.det_score,
                            query_variant=candidate.query_variant,
                            splink_match_weight=weights.get(candidate.entity_uid),
                        )
                        for candidate in scored_candidates
                    )
                )
            )

        return _score_c2_candidate_weights_run_v2(orphan_id, scored_candidates) >> (
            _finalize
        )

    return (
        _query_rows_run_v2(
            f"""
        WITH candidate_entities(
          entity_uid,
          source_article_id,
          source_incident_idx,
          query_variant,
          midpoint_day
        ) AS (
          SELECT * FROM (VALUES {values_sql})
        ),
        o AS (
          SELECT *
          FROM orphan_link_input
          WHERE unique_id = ?
        )
        SELECT DISTINCT
          c.entity_uid,
          c.source_article_id,
          c.source_incident_idx,
          c.query_variant,
          c.midpoint_day,
          ABS(
            COALESCE(CAST(e.midpoint_day AS DOUBLE), 0)
            - COALESCE(o.midpoint_day, 0)
          ) AS day_gap,
          CASE WHEN e.victim_sex = o.victim_sex THEN 1 ELSE 0 END AS sex_match,
          CASE WHEN e.weapon = o.weapon THEN 1 ELSE 0 END AS weapon_match,
          CASE
            WHEN e.circumstance = o.circumstance THEN 1 ELSE 0
          END AS circumstance_match,
          array_cosine_similarity(e.summary_vec, o.summary_vec) AS summary_cosine
        FROM candidate_entities c
        JOIN entity_link_input e
          ON e.unique_id = c.entity_uid
        CROSS JOIN o;
        """,
            _sql_params(tuple(params)),
        )
        >> (
            lambda rows: pure(
                Array.make(
                    tuple(
                        C2Candidate(
                            entity_uid=_safe_text(row.get("entity_uid")),
                            source_article_id=(
                                _as_int(row.get("source_article_id"), 0)
                                if row.get("source_article_id") is not None
                                else None
                            ),
                            source_incident_idx=(
                                _as_int(row.get("source_incident_idx"), 0)
                                if row.get("source_incident_idx") is not None
                                else None
                            ),
                            midpoint_day=(
                                _as_float(row.get("midpoint_day"), 0.0)
                                if row.get("midpoint_day") is not None
                                else None
                            ),
                            day_gap=_as_float(row.get("day_gap"), 9999.0),
                            compat_count=_as_int(row.get("sex_match"), 0)
                            + _as_int(row.get("weapon_match"), 0)
                            + _as_int(row.get("circumstance_match"), 0),
                            summary_cosine=_as_float(row.get("summary_cosine"), 0.0),
                            det_score=1.2
                            + 0.8
                            * (
                                _as_int(row.get("sex_match"), 0)
                                + _as_int(row.get("weapon_match"), 0)
                                + _as_int(row.get("circumstance_match"), 0)
                            )
                            + 1.7 * _as_float(row.get("summary_cosine"), 0.0)
                            - 0.01 * _as_float(row.get("day_gap"), 9999.0),
                            query_variant=_safe_text(row.get("query_variant")),
                        )
                        for row in rows
                    )
                )
            )
        )
    ) >> _attach_weights


def _aggregate_group_c2_candidates_v2(
    per_orphan_rows: Array[Array[C2Candidate]],
) -> Array[C2Candidate]:
    by_candidate: dict[tuple[str, int | None, int | None, str], dict[str, Any]] = {}
    for candidate_rows in per_orphan_rows:
        for candidate in candidate_rows:
            key = (
                candidate.entity_uid,
                candidate.source_article_id,
                candidate.source_incident_idx,
                candidate.query_variant,
            )
            existing = by_candidate.get(key)
            if existing is None:
                by_candidate[key] = {
                    "best_det_candidate": candidate,
                    "splink_match_weight": candidate.splink_match_weight,
                }
                continue
            best_det_candidate = cast(C2Candidate, existing["best_det_candidate"])
            if candidate.det_score > best_det_candidate.det_score:
                existing["best_det_candidate"] = candidate
            row_weight = candidate.splink_match_weight
            existing_weight = existing.get("splink_match_weight")
            if row_weight is not None and (
                existing_weight is None
                or _as_float(row_weight, 0.0) > _as_float(existing_weight, 0.0)
            ):
                existing["splink_match_weight"] = row_weight

    aggregated = [
        C2Candidate(
            entity_uid=record["best_det_candidate"].entity_uid,
            source_article_id=record["best_det_candidate"].source_article_id,
            source_incident_idx=record["best_det_candidate"].source_incident_idx,
            midpoint_day=record["best_det_candidate"].midpoint_day,
            day_gap=record["best_det_candidate"].day_gap,
            compat_count=record["best_det_candidate"].compat_count,
            summary_cosine=record["best_det_candidate"].summary_cosine,
            det_score=record["best_det_candidate"].det_score,
            query_variant=record["best_det_candidate"].query_variant,
            splink_match_weight=record.get("splink_match_weight"),
        )
        for record in by_candidate.values()
    ]
    aggregated.sort(
        key=lambda candidate: (
            -candidate.det_score,
            -_as_float(candidate.splink_match_weight, float("-inf")),
            -candidate.summary_cosine,
            -candidate.compat_count,
            candidate.day_gap,
            candidate.entity_uid,
            candidate.query_variant,
        )
    )
    return Array.make(tuple(aggregated))


def _candidate_incident_rows_run_v2(
    candidates: Array[C2Candidate],
) -> Run[list[dict[str, Any]]]:
    incident_map: dict[tuple[int | None, int | None], dict[str, Any]] = {}
    for candidate in candidates:
        if (
            candidate.source_article_id is None
            or candidate.source_incident_idx is None
        ):
            continue
        incident_key = (candidate.source_article_id, candidate.source_incident_idx)
        existing = incident_map.get(incident_key)
        if existing is None:
            incident_map[incident_key] = {
                "source_article_id": candidate.source_article_id,
                "source_incident_idx": candidate.source_incident_idx,
                "query_variants": (
                    [] if candidate.query_variant == "" else [candidate.query_variant]
                ),
                "eligible_entity_uids": (
                    [] if candidate.entity_uid == "" else [candidate.entity_uid]
                ),
                "best_det_candidate": candidate,
                "max_splink_match_weight": candidate.splink_match_weight,
            }
            continue
        if (
            candidate.query_variant != ""
            and candidate.query_variant not in existing["query_variants"]
        ):
            cast(list[str], existing["query_variants"]).append(candidate.query_variant)
        if (
            candidate.entity_uid != ""
            and candidate.entity_uid not in existing["eligible_entity_uids"]
        ):
            cast(list[str], existing["eligible_entity_uids"]).append(
                candidate.entity_uid
            )
        best_det_candidate = cast(C2Candidate, existing["best_det_candidate"])
        if candidate.det_score > best_det_candidate.det_score:
            existing["best_det_candidate"] = candidate
        candidate_weight = candidate.splink_match_weight
        existing_weight = existing.get("max_splink_match_weight")
        if candidate_weight is not None and (
            existing_weight is None
            or _as_float(candidate_weight, 0.0) > _as_float(existing_weight, 0.0)
        ):
            existing["max_splink_match_weight"] = candidate_weight

    return pure(
        [
            {
                "source_article_id": record["source_article_id"],
                "source_incident_idx": record["source_incident_idx"],
                "midpoint_day": cast(
                    C2Candidate, record["best_det_candidate"]
                ).midpoint_day,
                "day_gap": cast(C2Candidate, record["best_det_candidate"]).day_gap,
                "compat_count": cast(
                    C2Candidate, record["best_det_candidate"]
                ).compat_count,
                "summary_cosine": cast(
                    C2Candidate, record["best_det_candidate"]
                ).summary_cosine,
                "det_score": cast(C2Candidate, record["best_det_candidate"]).det_score,
                "splink_match_weight": record.get("max_splink_match_weight"),
                "query_variants": record["query_variants"],
                "eligible_entity_uids": record["eligible_entity_uids"],
            }
            for record in incident_map.values()
        ]
    )


def _incident_representatives_from_rows_v2(
    rows: list[dict[str, Any]],
) -> Array[IncidentRepresentativeCandidate]:
    by_incident: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_article_id = (
            _as_int(row.get("source_article_id"), 0)
            if row.get("source_article_id") is not None
            else None
        )
        source_incident_idx = (
            _as_int(row.get("source_incident_idx"), 0)
            if row.get("source_incident_idx") is not None
            else None
        )
        incident_key = _incident_key_from_candidate(
            source_article_id, source_incident_idx
        )
        existing = by_incident.get(incident_key)
        summary_cosine = _as_float(row.get("summary_cosine"), 0.0)
        day_gap = _as_float(row.get("day_gap"), 9999.0)
        query_variants = [
            _safe_text(value)
            for value in (row.get("query_variants") or [])
            if _safe_text(value) != ""
        ]
        eligible_entity_uids = [
            _safe_text(value)
            for value in (row.get("eligible_entity_uids") or [])
            if _safe_text(value) != ""
        ]
        if existing is None:
            representative_uid = (
                sorted(eligible_entity_uids)[0]
                if len(eligible_entity_uids) > 0
                else ""
            )
            by_incident[incident_key] = {
                "entity_uid": representative_uid,
                "source_article_id": source_article_id,
                "source_incident_idx": source_incident_idx,
                "midpoint_day": (
                    _as_float(row.get("midpoint_day"), 0.0)
                    if row.get("midpoint_day") is not None
                    else None
                ),
                "day_gap": day_gap,
                "compat_count": _as_int(row.get("compat_count"), 0),
                "summary_cosine": summary_cosine,
                "det_score": _as_float(row.get("det_score"), 0.0),
                "splink_match_weight": (
                    _as_float(row.get("splink_match_weight"), 0.0)
                    if row.get("splink_match_weight") is not None
                    else None
                ),
                "query_variants": query_variants,
                "eligible_entity_uids": eligible_entity_uids,
            }
            continue

        for query_variant in query_variants:
            if query_variant not in existing["query_variants"]:
                cast(list[str], existing["query_variants"]).append(query_variant)
        for entity_uid in eligible_entity_uids:
            if entity_uid not in existing["eligible_entity_uids"]:
                cast(list[str], existing["eligible_entity_uids"]).append(entity_uid)
        if _as_float(row.get("det_score"), 0.0) > _as_float(
            existing.get("det_score"), 0.0
        ):
            existing["midpoint_day"] = (
                _as_float(row.get("midpoint_day"), 0.0)
                if row.get("midpoint_day") is not None
                else None
            )
            existing["day_gap"] = day_gap
            existing["compat_count"] = _as_int(row.get("compat_count"), 0)
            existing["summary_cosine"] = summary_cosine
            existing["det_score"] = _as_float(row.get("det_score"), 0.0)
        row_splink = (
            _as_float(row.get("splink_match_weight"), 0.0)
            if row.get("splink_match_weight") is not None
            else None
        )
        existing_splink = existing.get("splink_match_weight")
        if row_splink is not None and (
            existing_splink is None
            or _as_float(row_splink, 0.0) > _as_float(existing_splink, 0.0)
        ):
            existing["splink_match_weight"] = row_splink
        sorted_entities = sorted(cast(list[str], existing["eligible_entity_uids"]))
        existing["entity_uid"] = sorted_entities[0] if len(sorted_entities) > 0 else ""

    representatives = [
        IncidentRepresentativeCandidate(
            entity_uid=_safe_text(record.get("entity_uid")),
            source_article_id=cast(int | None, record.get("source_article_id")),
            source_incident_idx=cast(int | None, record.get("source_incident_idx")),
            midpoint_day=cast(float | None, record.get("midpoint_day")),
            day_gap=_as_float(record.get("day_gap"), 9999.0),
            compat_count=_as_int(record.get("compat_count"), 0),
            summary_cosine=_as_float(record.get("summary_cosine"), 0.0),
            det_score=_as_float(record.get("det_score"), 0.0),
            splink_match_weight=(
                _as_float(record.get("splink_match_weight"), 0.0)
                if record.get("splink_match_weight") is not None
                else None
            ),
            anchor_hit_count=len(cast(list[str], record.get("query_variants") or [])),
            query_variant=(
                cast(list[str], record.get("query_variants") or [""])[0]
                if len(cast(list[str], record.get("query_variants") or [])) > 0
                else ""
            ),
            query_variants=tuple(cast(list[str], record.get("query_variants") or [])),
            eligible_entity_uids=tuple(
                sorted(cast(list[str], record.get("eligible_entity_uids") or []))
            ),
        )
        for record in by_incident.values()
        if len(cast(list[str], record.get("eligible_entity_uids") or [])) > 0
    ]
    representatives.sort(
        key=lambda candidate: (
            -candidate.det_score,
            -_as_float(candidate.splink_match_weight, float("-inf")),
            -candidate.anchor_hit_count,
            -candidate.summary_cosine,
            -candidate.compat_count,
            candidate.day_gap,
            _safe_text(candidate.entity_uid),
        )
    )
    return Array.make(tuple(representatives[:MAX_MERGED_CANDIDATES_FOR_API2]))


def _incident_representative_rows_for_display_v2(
    candidates: Array[IncidentRepresentativeCandidate],
) -> list[dict[str, Any]]:
    return [candidate.to_dict() for candidate in candidates]


def _candidate_incident_context_by_key_run_v2(
    representative_entity_uid: str,
    source_article_id: int | None,
    source_incident_idx: int | None,
) -> Run[dict[str, Any]]:
    def _load_exact_incident_context(_: dict[str, Any]) -> Run[dict[str, Any]]:
        if source_article_id is None or source_incident_idx is None:
            return _candidate_incident_context_run_v2(
                representative_entity_uid, source_article_id
            )
        return (
            _query_rows_run_v2(
                """
            SELECT
              incident_date,
              year,
              month,
              location_raw,
              victim_count,
              summary
            FROM incidents_cached
            WHERE article_id = ?
              AND incident_idx = ?
            LIMIT 1;
            """,
                _sql_params((source_article_id, source_incident_idx)),
            )
            >> (
                lambda rows: pure(
                    {
                        "incident_date": "",
                        "incident_location": "",
                        "victim_count": "",
                        "incident_summary": "",
                    }
                )
                if len(rows) == 0
                else pure(
                    {
                        "incident_date": _format_incident_date(
                            rows[0].get("incident_date"),
                            year=rows[0].get("year"),
                            month=rows[0].get("month"),
                        ),
                        "incident_location": _safe_text(rows[0].get("location_raw")),
                        "victim_count": _safe_text(rows[0].get("victim_count")),
                        "incident_summary": _safe_text(rows[0].get("summary")),
                    }
                )
            )
        )

    return pure({}) >> _load_exact_incident_context


def _build_rank_request_for_group_run_v2(
    group_dossier: GroupDossier, candidates: Array[IncidentRepresentativeCandidate]
) -> Run[RankRequest]:
    def _build_one(
        candidate: IncidentRepresentativeCandidate,
    ) -> Run[RankCandidateContext]:
        def _build_candidate_context(
            article_ctx: dict[str, Any]
        ) -> Run[RankCandidateContext]:
            return _candidate_incident_context_by_key_run_v2(
                candidate.entity_uid,
                candidate.source_article_id,
                candidate.source_incident_idx,
            ) >> (
                lambda incident_ctx: pure(
                    RankCandidateContext(
                        entity_uid=candidate.entity_uid,
                        article_title=_safe_text(article_ctx.get("article_title")),
                        article_text=_safe_text(article_ctx.get("article_text")),
                        article_date=_safe_text(article_ctx.get("article_date")),
                        incident_date=_safe_text(incident_ctx.get("incident_date")),
                        incident_location=_safe_text(
                            incident_ctx.get("incident_location")
                        ),
                        victim_count=_safe_text(incident_ctx.get("victim_count")),
                        incident_summary=_safe_text(
                            incident_ctx.get("incident_summary")
                        ),
                    )
                )
            )

        return _candidate_article_context_run_v2(candidate.source_article_id) >> (
            _build_candidate_context
        )

    return (
        array_traverse_run(candidates, _build_one)
        if candidates.length > 0
        else pure(Array.empty())
    ) >> (
        lambda contexts: pure(
            RankRequest(
                article_title=group_dossier.article_title,
                article_text=group_dossier.article_text,
                article_date=_format_article_date(group_dossier.article_pub_date),
                incident_date=group_dossier.incident_date,
                incident_location=group_dossier.incident_location,
                victim_count=group_dossier.victim_count,
                incident_summary=group_dossier.incident_summary,
                candidates=tuple(contexts),
            )
        )
    )

def _score_c2_candidate_weights_run_v2(
    orphan_id: str, candidates: Array[C2Candidate]
) -> Run[dict[str, float]]:
    candidate_ids = tuple(
        sorted(
            {
                candidate.entity_uid
                for candidate in candidates
                if candidate.entity_uid != ""
            }
        )
    )
    if len(candidate_ids) == 0:
        return pure({})
    settings = _settings_for_c2_weight_scoring(_load_dedupe_model_settings())
    suffix = _sha256(f"{orphan_id}|{'|'.join(candidate_ids)}")[:12]
    left_table = _sanitize_sql_identifier(f"adj_c2_left_{suffix}", prefix="tmp")
    right_table = _sanitize_sql_identifier(f"adj_c2_right_{suffix}", prefix="tmp")
    pairs_table = _sanitize_sql_identifier(f"adj_c2_pairs_{suffix}", prefix="tmp")
    placeholders = ", ".join("?" for _ in candidate_ids)
    main = (
        sql_exec(
            SQL(
                f"CREATE OR REPLACE TEMP TABLE {left_table} AS SELECT * FROM"
                f" entity_link_input WHERE unique_id IN ({placeholders});"
            ),
            _sql_params(candidate_ids),
        )
        ^ sql_exec(
            SQL(
                f"CREATE OR REPLACE TEMP TABLE {right_table} AS SELECT * FROM"
                " orphan_link_input WHERE unique_id = ?;"
            ),
            _sql_params((orphan_id,)),
        )
        ^ splink_dedupe_job(
            input_table=PredictionInputTableNames((left_table, right_table)),
            settings=settings,
            predict_threshold=0.0,
            cluster_threshold=0.0,
            pairs_out=PairsTableName(pairs_table),
            train_first=False,
            skip_u_estimation=True,
            visualize=False,
            splink_key=SplinkType.ORPHAN_ADJ_SCORE,
            inference_only=True,
            capture_blocked_edges=False,
        )
        >> (
            lambda _: _query_rows_run_v2(
                "SELECT CAST(unique_id_l AS VARCHAR) AS entity_uid, match_weight FROM"
                f" {pairs_table} WHERE CAST(unique_id_r AS VARCHAR) = ?;",
                _sql_params((orphan_id,)),
            )
        )
    )
    cleanup = (
        sql_exec(SQL(f"DROP TABLE IF EXISTS {pairs_table};"))
        ^ sql_exec(SQL(f"DROP TABLE IF EXISTS {left_table};"))
        ^ sql_exec(SQL(f"DROP TABLE IF EXISTS {right_table};"))
    )
    def _finalize_splink_weight_scoring(result: Any) -> Run[dict[str, float]]:
        def _build_weight_map(_: Any) -> Run[dict[str, float]]:
            return (
                throw(
                    ErrorPayload(
                        f"splink_c2_weight_scoring_failed: {_safe_text(result.l)}"
                    )
                )
                if isinstance(result, Left)
                else pure(
                    {
                        _safe_text(row.get("entity_uid")): _as_float(
                            row.get("match_weight"), 0.0
                        )
                        for row in result.r
                        if _safe_text(row.get("entity_uid")) != ""
                        and row.get("match_weight") is not None
                    }
                )
            )

        return cleanup >> _build_weight_map

    return run_except(main) >> _finalize_splink_weight_scoring


def _insert_candidates_run_v2(
    run_id: str, group_id: str, orphan_id: str, candidates: Array[C2Candidate]
) -> Run[Any]:
    def _insert_one(candidate: C2Candidate) -> Run[Any]:
        return sql_exec(
            SQL(
                """
                INSERT INTO orphan_adj_candidates_c2 (
                  run_id, group_id, orphan_id, entity_uid, source_article_id,
                  query_variant, det_score, splink_match_weight, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::JSON);
                """
            ),
            _sql_params(
                (
                    run_id,
                    group_id,
                    orphan_id,
                    candidate.entity_uid,
                    candidate.source_article_id,
                    candidate.query_variant,
                    candidate.det_score,
                    candidate.splink_match_weight,
                    _canonical_json(candidate.to_dict()),
                )
            ),
        )

    return (
        array_traverse_run(candidates, _insert_one)
        if candidates.length > 0
        else pure(Array.empty())
    ) >> (lambda _: pure(None))


def _merge_candidates_v2(candidates: Array[C2Candidate]) -> Array[MergedCandidate]:
    merged = _merge_candidates(
        [], [_c2_candidate_to_dict_v2(candidate) for candidate in candidates]
    )
    return Array.make(tuple(_merged_candidate_from_dict_v2(row) for row in merged))


def _insert_merged_candidates_run_v2(
    run_id: str, group_id: str, orphan_id: str, merged: Array[MergedCandidate]
) -> Run[Any]:
    def _insert_one(candidate: MergedCandidate) -> Run[Any]:
        return sql_exec(
            SQL(
                """
                INSERT INTO orphan_adj_candidates_merged (
                  run_id, group_id, orphan_id, entity_uid, det_score,
                  splink_match_weight, source_stages, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON);
                """
            ),
            _sql_params(
                (
                    run_id,
                    group_id,
                    orphan_id,
                    candidate.entity_uid,
                    candidate.det_score,
                    candidate.splink_match_weight,
                    ",".join(candidate.source_stages),
                    _canonical_json(candidate.to_dict()),
                )
            ),
        )

    return (
        array_traverse_run(merged, _insert_one)
        if merged.length > 0
        else pure(Array.empty())
    ) >> (lambda _: pure(None))


def _insert_incident_representatives_run_v2(
    run_id: str,
    group_id: str,
    orphan_id: str,
    representatives: Array[IncidentRepresentativeCandidate],
) -> Run[Any]:
    def _insert_one(candidate: IncidentRepresentativeCandidate) -> Run[Any]:
        return sql_exec(
            SQL(
                """
                INSERT INTO orphan_adj_candidates_merged (
                  run_id, group_id, orphan_id, entity_uid, det_score,
                  splink_match_weight, source_stages, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON);
                """
            ),
            _sql_params(
                (
                    run_id,
                    group_id,
                    orphan_id,
                    candidate.entity_uid,
                    candidate.det_score,
                    None,
                    "incident_repr",
                    _canonical_json(candidate.to_dict()),
                )
            ),
        )

    return (
        array_traverse_run(representatives, _insert_one)
        if representatives.length > 0
        else pure(Array.empty())
    ) >> (lambda _: pure(None))


def _candidate_article_context_run_v2(article_id: int | None) -> Run[dict[str, Any]]:
    if article_id is None:
        return pure({"article_title": "", "article_text": "", "article_date": ""})
    return (
        _query_rows_run_v2(
            """
        SELECT RecordId, Title, FullText, PubDate
        FROM articles
        WHERE RecordId = ?
          AND Dataset = 'CLASS_WP'
          AND gptClass = 'M'
        LIMIT 1;
        """,
            _sql_params((article_id,)),
            sqlite=True,
        )
        >> (
            lambda rows: pure(
                {
                    "article_title": (
                        _safe_text(rows[0].get("Title")) if len(rows) > 0 else ""
                    ),
                    "article_text": (
                        _safe_text(rows[0].get("FullText"))[:MAX_FULLTEXT_CHARS]
                        if len(rows) > 0
                        else ""
                    ),
                    "article_date": (
                        _format_article_date(rows[0].get("PubDate"))
                        if len(rows) > 0
                        else ""
                    ),
                }
            )
        )
    )


def _candidate_incident_context_run_v2(
    entity_uid: str, source_article_id: int | None
) -> Run[dict[str, Any]]:
    """Load incident-level context for one ranked candidate entity."""

    def _build_incident_context(row: dict[str, Any]) -> Run[dict[str, Any]]:
        return (
            (
                _query_rows_run_v2(
                    """
                    SELECT i.summary
                    FROM incidents_cached i
                    LEFT JOIN entity_link_input e
                      ON e.unique_id = ?
                    WHERE i.article_id = ?
                    ORDER BY
                      CASE
                        WHEN i.summary_vec IS NOT NULL AND e.summary_vec IS NOT NULL
                          THEN array_cosine_similarity(i.summary_vec, e.summary_vec)
                        ELSE -1
                      END DESC,
                      i.incident_idx ASC
                    LIMIT 1;
                    """,
                    _sql_params((entity_uid, source_article_id)),
                )
                if source_article_id is not None
                else pure([])
            )
            >> (
                lambda summary_rows: pure(
                    {
                        "incident_date": _format_incident_date(
                            row.get("incident_date"),
                            year=row.get("year"),
                            month=row.get("month"),
                        ),
                        "incident_location": _safe_text(
                            row.get("geo_address_norm")
                        ),
                        "victim_count": _safe_text(row.get("victim_count")),
                        "incident_summary": (
                            _safe_text(summary_rows[0].get("summary"))
                            if len(summary_rows) > 0
                            and _safe_text(summary_rows[0].get("summary")).strip()
                            != ""
                            else " | ".join(
                                part
                                for part in [
                                    _safe_text(row.get("circumstance")),
                                    _safe_text(row.get("relationship")),
                                ]
                                if part.strip() != ""
                            )
                        ),
                    }
                )
            )
        )

    return (
        _query_rows_run_v2(
            """
        SELECT
          unique_id,
          incident_date,
          year,
          month,
          geo_address_norm,
          victim_count,
          circumstance,
          relationship
        FROM entity_link_input
        WHERE unique_id = ?
        LIMIT 1;
        """,
            _sql_params((entity_uid,)),
        )
        >> (
            lambda rows: pure(
                {
                    "incident_date": "",
                    "incident_location": "",
                    "victim_count": "",
                    "incident_summary": "",
                }
            )
            if len(rows) == 0
            else _build_incident_context(rows[0])
        )
    )


def _build_rank_request_run_v2(
    dossier: Dossier, merged: Array[MergedCandidate]
) -> Run[RankRequest]:
    def _build_one(candidate: MergedCandidate) -> Run[RankCandidateContext]:
        source_article_id = _candidate_source_article_id(candidate.to_dict())
        def _build_candidate_context(
            article_ctx: dict[str, Any]
        ) -> Run[RankCandidateContext]:
            return _candidate_incident_context_run_v2(
                candidate.entity_uid, source_article_id
            ) >> (
                lambda incident_ctx: pure(
                    RankCandidateContext(
                        entity_uid=candidate.entity_uid,
                        article_title=_safe_text(article_ctx.get("article_title")),
                        article_text=_safe_text(article_ctx.get("article_text")),
                        article_date=_safe_text(article_ctx.get("article_date")),
                        incident_date=_safe_text(incident_ctx.get("incident_date")),
                        incident_location=_safe_text(
                            incident_ctx.get("incident_location")
                        ),
                        victim_count=_safe_text(incident_ctx.get("victim_count")),
                        incident_summary=_safe_text(
                            incident_ctx.get("incident_summary")
                        ),
                    )
                )
            )

        return _candidate_article_context_run_v2(source_article_id) >> (
            _build_candidate_context
        )

    return (
        array_traverse_run(merged, _build_one)
        if merged.length > 0
        else pure(Array.empty())
    ) >> (
        lambda candidates: pure(
            RankRequest(
                article_title=dossier.article_title,
                article_text=dossier.article_text,
                article_date=_format_article_date(dossier.article_pub_date),
                incident_date=_format_incident_date(
                    dossier.incident_date, year=dossier.year, month=dossier.month
                ),
                incident_location=dossier.geo_address_norm,
                victim_count=dossier.victim_count,
                incident_summary=dossier.incident_summary_gpt,
                candidates=tuple(candidates),
            )
        )
    )


def _log_anchor_and_candidates_run_v2(
    orphan_id: str,
    valid_anchors: Array[ValidatedAnchor],
    variants_used: Array[str],
    c2_rows: Array[C2Candidate],
    merged: Array[MergedCandidate] | Array[IncidentRepresentativeCandidate],
    ranked: list[dict[str, Any]],
) -> Run[Any]:
    c2_dicts = [_c2_candidate_to_dict_v2(candidate) for candidate in c2_rows]
    merged_dicts = [candidate.to_dict() for candidate in merged]
    anchors_payload = {
        "anchors": [
            {"anchor_text": anchor.anchor_text, "variants": list(anchor.variants)}
            for anchor in valid_anchors
        ]
    }
    return (
        put_line(f"[K][{orphan_id}] anchor_usage:")
        ^ put_line(
            f"[K][{orphan_id}] C2 text-phrase criteria count={variants_used.length}"
        )
        ^ put_line(_pretty_json({"c2_text_phrases": list(variants_used)}))
        ^ put_line(_pretty_json(anchors_payload))
        ^ put_line(f"[K][{orphan_id}] candidates_C2 count={len(c2_dicts)}")
        ^ put_line(_format_c2_candidates_table(c2_dicts))
        ^ put_line(f"[K][{orphan_id}] candidates_merged count={len(merged_dicts)}")
        ^ put_line(
            _format_c2_candidates_table(merged_dicts, include_anchor_hit_count=True)
        )
        ^ put_line(f"[K][{orphan_id}] candidates_ranked count={len(ranked)}")
        ^ put_line(_pretty_json({"candidates_ranked": _compact_candidate_view(ranked)}))
    )


def _validate_rank_output_v2(
    response: RankResponse, merged: Array[MergedCandidate]
) -> list[dict[str, Any]]:
    return _validate_rank_output(
        response, [candidate.to_dict() for candidate in merged]
    )


def _validate_incident_rank_output_v2(
    response: RankResponse, candidates: Array[IncidentRepresentativeCandidate]
) -> list[dict[str, Any]]:
    if response.match_result != "match":
        return []
    allowed = {candidate.entity_uid for candidate in candidates}
    by_uid = {candidate.entity_uid: candidate for candidate in candidates}
    matched_uid = _safe_text(response.matched_entity_uid)
    if matched_uid == "" or matched_uid not in allowed:
        return []
    selected = by_uid[matched_uid]
    return [
        {
            "entity_uid": selected.entity_uid,
            "rank_score": 1.0,
            "det_score": selected.det_score,
            "anchor_hit_count": selected.anchor_hit_count,
            "summary_cosine": selected.summary_cosine,
            "day_gap": selected.day_gap,
            "source_article_id": selected.source_article_id,
            "source_incident_idx": selected.source_incident_idx,
            "query_variant": selected.query_variant,
            "query_variants": list(selected.query_variants),
            "eligible_entity_uids": list(selected.eligible_entity_uids),
            "stage_name": "incident_repr",
        }
    ]


def _make_assignment_candidate_dict_v2(
    *,
    entity_uid: str,
    source_article_id: int | None,
    source_incident_idx: int | None,
    splink_match_weight: float,
    ranked_incidents: list[dict[str, Any]],
) -> dict[str, Any]:
    top_incident = ranked_incidents[0] if len(ranked_incidents) > 0 else {}
    return {
        "entity_uid": entity_uid,
        "rank_score": 1.0,
        "det_score": _as_float(top_incident.get("det_score"), 0.0),
        "splink_match_weight": splink_match_weight,
        "anchor_hit_count": _as_int(top_incident.get("anchor_hit_count"), 0),
        "summary_cosine": _as_float(top_incident.get("summary_cosine"), 0.0),
        "day_gap": _as_float(top_incident.get("day_gap"), 9999.0),
        "source_article_id": source_article_id,
        "source_incident_idx": source_incident_idx,
        "query_variant": _safe_text(top_incident.get("query_variant")),
        "query_variants": top_incident.get("query_variants") or [],
        "stage_name": "incident_assignment",
    }


def _maximize_weight_assignment_v2(
    orphan_ids: tuple[str, ...],
    entity_uids: tuple[str, ...],
    weight_by_orphan: dict[str, dict[str, float]],
) -> dict[str, str]:
    if len(orphan_ids) == 0 or len(entity_uids) == 0:
        return {}
    sorted_orphans = tuple(sorted(orphan_ids))
    sorted_entities = tuple(sorted(entity_uids))
    target_size = min(len(sorted_orphans), len(sorted_entities))
    weights = tuple(
        tuple(
            _as_float(weight_by_orphan.get(orphan_id, {}).get(entity_uid), 0.0)
            for entity_uid in sorted_entities
        )
        for orphan_id in sorted_orphans
    )

    @lru_cache(maxsize=None)
    def _search(
        orphan_idx: int, used_mask: int, assigned_count: int
    ) -> tuple[float, tuple[tuple[str, str], ...]] | None:
        remaining_orphans = len(sorted_orphans) - orphan_idx
        remaining_needed = target_size - assigned_count
        if remaining_needed < 0 or remaining_orphans < remaining_needed:
            return None
        if orphan_idx == len(sorted_orphans):
            return (0.0, tuple()) if assigned_count == target_size else None

        best: tuple[float, tuple[tuple[str, str], ...]] | None = None
        if remaining_orphans > remaining_needed:
            best = _search(orphan_idx + 1, used_mask, assigned_count)

        for entity_idx, entity_uid in enumerate(sorted_entities):
            if used_mask & (1 << entity_idx):
                continue
            tail = _search(
                orphan_idx + 1, used_mask | (1 << entity_idx), assigned_count + 1
            )
            if tail is None:
                continue
            candidate = (
                weights[orphan_idx][entity_idx] + tail[0],
                ((sorted_orphans[orphan_idx], entity_uid),) + tail[1],
            )
            if best is None:
                best = candidate
                continue
            if candidate[0] > best[0]:
                best = candidate
                continue
            if candidate[0] == best[0] and candidate[1] < best[1]:
                best = candidate
        return best

    result = _search(0, 0, 0)
    if result is None:
        return {}
    return dict(result[1])


def _cache_terminal_decision_run_v2(decision: CaseDecision) -> Run[Any]:
    """Persist a terminal decision in the end-to-end adjudication cache."""

    if decision.label not in {"matched", "not_same_person", "insufficient_information"}:
        return pure(None)
    return _load_orphan_dossier_run_v2(decision.orphan_id) >> (
        lambda dossier: _llm_cache_put_run_v2(
            stage=E2E_CACHE_STAGE,
            idempotency_key=decision.orphan_id,
            model=f"{PASS1_MODEL}|{RANK_MODEL}",
            prompt_version=f"{PASS1_PROMPT_VERSION}|{RANK_PROMPT_VERSION}|e2e_v1",
            input_payload=_build_pass1_request_v2(dossier).to_cache_payload(),
            response_payload=_decision_to_cache_payload(decision),
        )
    )


def _cached_decisions_from_readiness_run_v2(
    readiness_rows: Array[CacheReadinessRow], excluded_orphan_ids: set[str]
) -> Run[Array[CaseDecision]]:
    selected = Array.make(
        tuple(
            row
            for row in readiness_rows
            if row.orphan_id not in excluded_orphan_ids
            and row.readiness_status == "decision_ready_from_cache"
            and row.pass1_idempotency_key != ""
        )
    )

    def _load_one(row: CacheReadinessRow) -> Run[CaseDecision | None]:
        def _after_cache(cached: dict[str, Any] | None) -> Run[CaseDecision | None]:
            if cached is None:
                return pure(None)
            return pure(_decision_from_cache_payload(cached, row.orphan_id))

        return (
            _llm_cache_get_run_v2(E2E_CACHE_STAGE, row.pass1_idempotency_key)
            >> _after_cache
        )

    def _filter_terminal_cached_decisions(
        rows: Array[CaseDecision | None],
    ) -> Run[Array[CaseDecision]]:
        return pure(
            Array.make(
                tuple(
                    row
                    for row in rows
                    if row is not None
                    and row.label
                    in {"matched", "not_same_person", "insufficient_information"}
                )
            )
        )

    return (
        array_traverse_run(selected, _load_one)
        if selected.length > 0
        else pure(Array.empty())
    ) >> _filter_terminal_cached_decisions


def _rebuild_overrides_table_run_v2(
    run_id: str, decisions: Array[CaseDecision], *, dry_run: bool
) -> Run[int]:
    terminal_by_orphan: dict[str, CaseDecision] = {}
    for decision in decisions:
        if decision.label not in {
            "matched",
            "not_same_person",
            "insufficient_information",
        }:
            continue
        if decision.orphan_id not in terminal_by_orphan:
            terminal_by_orphan[decision.orphan_id] = decision
    terminal = Array.make(tuple(terminal_by_orphan.values()))
    if dry_run:
        return pure(terminal.length)

    def _insert_terminal_rows(prior_by_orphan: dict[str, dict[str, Any]]) -> Run[Any]:
        def _insert_one(decision: CaseDecision) -> Run[Any]:
            decision_hash = _decision_hash_for_case(decision)
            evidence_with_hash = {
                **decision.evidence_json,
                "decision_hash": decision_hash,
            }
            prior = prior_by_orphan.get(decision.orphan_id)
            history_run: Run[Any]
            if _prior_decision_hash(prior) != decision_hash:
                history_run = sql_exec(
                    SQL(
                        """
                        INSERT INTO orphan_adjudication_history (
                          history_id,
                          run_id,
                          orphan_id,
                          prior_resolution_label,
                          prior_resolved_entity_id,
                          prior_confidence,
                          new_resolution_label,
                          new_resolved_entity_id,
                          new_confidence,
                          reason_summary,
                          evidence_json,
                          analyst_mode,
                          changed_at
                        ) VALUES (
                          NULL,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON,
                          'interactive_agent_k', NOW()
                        );
                        """
                    ),
                    _sql_params(
                        (
                            run_id,
                            decision.orphan_id,
                            prior.get("resolution_label") if prior else None,
                            prior.get("resolved_entity_id") if prior else None,
                            prior.get("confidence") if prior else None,
                            decision.label,
                            decision.resolved_entity_id,
                            decision.confidence,
                            decision.reason_summary,
                            _canonical_json(evidence_with_hash),
                        )
                    ),
                )
            else:
                history_run = pure(None)
            return history_run ^ sql_exec(
                SQL(
                    """
                    INSERT INTO orphan_adj_overrides_next_tmp (
                      orphan_id,
                      resolution_label,
                      resolved_entity_id,
                      confidence,
                      reason_summary,
                      evidence_json,
                      analyst_mode,
                      created_at,
                      updated_at
                    ) VALUES (
                      ?, ?, ?, ?, ?, ?::JSON,
                      'interactive_agent_k', NOW(), NOW()
                    );
                    """
                ),
                _sql_params(
                    (
                        decision.orphan_id,
                        decision.label,
                        decision.resolved_entity_id,
                        decision.confidence,
                        decision.reason_summary,
                        _canonical_json(evidence_with_hash),
                    )
                ),
            )

        return (
            array_traverse_run(terminal, _insert_one)
            if terminal.length > 0
            else pure(Array.empty())
        )

    def _commit_swap() -> Run[int]:
        def _finish_transaction(result: Any) -> Run[int]:
            return (
                sql_exec(SQL("COMMIT;"))
                ^ sql_exec(SQL("DROP TABLE orphan_adj_overrides_next_tmp;"))
                ^ pure(terminal.length)
            ) if isinstance(result, Right) else (
                sql_exec(SQL("ROLLBACK;"))
                ^ sql_exec(SQL("DROP TABLE IF EXISTS orphan_adj_overrides_next_tmp;"))
                ^ throw(ErrorPayload(f"{result.l}"))
            )

        return sql_exec(SQL("BEGIN TRANSACTION;")) ^ run_except(
            sql_exec(SQL("DELETE FROM orphan_adjudication_overrides;"))
            ^ sql_exec(
                SQL(
                    "INSERT INTO orphan_adjudication_overrides SELECT * FROM"
                    " orphan_adj_overrides_next_tmp;"
                )
            )
        ) >> _finish_transaction

    def _stage_next_override_rows(
        prior_rows: list[dict[str, Any]],
    ) -> Run[int]:
        prior_by_orphan = {_safe_text(row.get("orphan_id")): row for row in prior_rows}
        return (
            sql_exec(
                SQL(
                    """
                    CREATE TEMP TABLE orphan_adj_overrides_next_tmp AS
                    SELECT *
                    FROM orphan_adjudication_overrides
                    WHERE 1=0;
                    """
                )
            )
            ^ _insert_terminal_rows(prior_by_orphan)
        ) >> (lambda _: _commit_swap())

    return (
        _query_rows_run_v2(
            """
        SELECT
          orphan_id,
          resolution_label,
          resolved_entity_id,
          confidence,
          reason_summary,
          evidence_json
        FROM orphan_adjudication_overrides;
        """
        )
        >> _stage_next_override_rows
    )


def _finalize_decision_run_v2(
    run_id: str,
    decision: CaseDecision,
    *,
    display_matched_article: bool = True,
) -> Run[CaseDecision]:
    decision_hash = _decision_hash_for_case(decision)
    stage_trace = decision.evidence_json.get("stage_trace")
    error_message = None
    if (
        decision.label == "analysis_incomplete"
        and isinstance(stage_trace, list)
        and len(stage_trace) > 0
        and isinstance(stage_trace[0], dict)
    ):
        error_message = _safe_text(stage_trace[0].get("error"))[:800] or None
    return (
        _set_case_state_run_v2(
            run_id,
            decision.orphan_id,
            case_status=(
                "completed" if decision.label != "analysis_incomplete" else "failed"
            ),
            stage_completed="decision",
            decision_label=decision.label,
            resolved_entity_id=decision.resolved_entity_id,
            decision_hash=decision_hash,
            error_message=error_message,
        )
        ^ _cache_terminal_decision_run_v2(decision)
        ^ (
            _display_matched_article_run_v2(decision)
            if display_matched_article
            else pure(None)
        )
        ^ pure(decision)
    )


def _log_group_match_assignments_run_v2(
    decisions: Array[CaseDecision],
) -> Run[Any]:
    """Log the final orphan-to-entity matches chosen for one processed group."""
    matched = tuple(
        decision
        for decision in decisions
        if decision.label == "matched"
        and _safe_text(decision.resolved_entity_id) != ""
    )
    if len(matched) == 0:
        return pure(None)
    leader_orphan_id = decisions[0].orphan_id if decisions.length > 0 else ""
    lines = [f"[K][{leader_orphan_id}] final_group_matches:"]
    lines.extend(
        f"  {_display_safe_id(decision.orphan_id)} -> "
        f"{_display_safe_id(decision.resolved_entity_id)}"
        for decision in matched
    )
    return put_line("\n".join(lines))


def _display_group_matched_article_run_v2(
    decisions: Array[CaseDecision],
) -> Run[Any]:
    """Display the matched source article once for a processed group."""
    matched = tuple(
        decision for decision in decisions if decision.label == "matched"
    )
    if len(matched) == 0:
        return pure(None)
    leader_orphan_id = decisions[0].orphan_id if decisions.length > 0 else ""
    article_ids = tuple(
        article_id
        for article_id in (
            _matched_article_id_from_decision(decision) for decision in matched
        )
        if article_id is not None
    )
    if len(article_ids) == 0:
        return pure(None)
    return put_line(f"[K][{leader_orphan_id}] Matched article:") ^ (
        _display_article_by_id_run_v2(
            leader_orphan_id,
            article_ids[0],
            empty_message="Matched article_id missing for display.",
        )
    )


def _finalize_group_decisions_run_v2(
    run_id: str,
    decisions: Array[CaseDecision],
) -> Run[Array[CaseDecision]]:
    """Persist per-decision state, then emit one consolidated match log per group."""
    if decisions.length == 0:
        return pure(Array.empty())

    def _display_group_outcome(
        finalized: Array[CaseDecision],
    ) -> Run[Array[CaseDecision]]:
        return (
            _log_group_match_assignments_run_v2(finalized)
            ^ _display_group_matched_article_run_v2(finalized)
            ^ pure(finalized)
        )

    return array_traverse_run(
        decisions,
        lambda decision: _finalize_decision_run_v2(
            run_id, decision, display_matched_article=False
        ),
    ) >> _display_group_outcome


def _processing_error_work_v2(
    orphan_id: str, article_id: int | None, message: str
) -> OrphanWork:
    return OrphanWork(
        orphan_id=orphan_id,
        article_id=article_id,
        insufficient=False,
        insufficient_reason=None,
        stage_trace=[
            {"stage": "error", "row_count": 0, "gate": "fail", "error": message}
        ],
        anchors=[],
        valid_variants=[],
        merged_candidates=[],
        ranked_candidates=[],
        provisional_reason="processing error",
    )


def _process_single_orphan_run_v2(
    run_id: str, group_id: str, row: CacheReadinessRow
) -> Run[OrphanWork]:
    orphan_id = row.orphan_id

    def _rank_stage(
        dossier: Dossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        anchor_failures: list[str],
        scored_candidates: Array[C2Candidate],
        weights: dict[str, float],
    ) -> Run[OrphanWork]:
        merged = _merge_candidates_v2(scored_candidates)

        def _build_ranked_work(rank_t: GPTResponseTuple) -> Run[OrphanWork]:
            rank_response = cast(RankResponse, rank_t.parsed.output)
            ranked = _validate_rank_output_v2(rank_response, merged)
            return (
                put_line(f"[K][{orphan_id}] rank_api_call: completed")
                ^ _log_api_usage_and_reasoning_run_v2(orphan_id, "rank", rank_t)
                ^ put_line(f"[K][{orphan_id}] rank_result (api_call):")
                ^ put_line(_pretty_json(json.loads(rank_response.model_dump_json())))
                ^ _log_stage_metric_run_v2(
                    run_id,
                    group_id,
                    orphan_id,
                    "rank",
                    "rank_validate",
                    len(ranked),
                    "direct_ranker",
                )
                ^ pure(
                    OrphanWork(
                        orphan_id=orphan_id,
                        article_id=dossier.article_id,
                        insufficient=(
                            rank_response.match_result
                            == "insufficient_information"
                        ),
                        insufficient_reason=(
                            "rank"
                            if rank_response.match_result
                            == "insufficient_information"
                            else None
                        ),
                        stage_trace=[
                            {
                                "stage": "pass1",
                                "row_count": valid_anchors.length,
                                "gate": "pass",
                                "anchor_failures": anchor_failures,
                            },
                            {"stage": "B", "row_count": 0, "gate": "bypassed"},
                            {"stage": "C", "row_count": 0, "gate": "bypassed"},
                            {
                                "stage": "splink_c2_weight",
                                "row_count": len(weights),
                                "gate": "pass",
                            },
                            {
                                "stage": "C2",
                                "row_count": scored_candidates.length,
                                "gate": "pass",
                            },
                            {
                                "stage": "rank",
                                "row_count": len(ranked),
                                "gate": "pass",
                            },
                        ],
                        anchors=[
                            _validated_anchor_to_dict_v2(anchor)
                            for anchor in valid_anchors
                        ],
                        valid_variants=list(variants_used),
                        merged_candidates=[
                            candidate.to_dict() for candidate in merged
                        ],
                        ranked_candidates=ranked,
                        provisional_reason=(
                            "insufficient information from rank api"
                            if rank_response.match_result
                            == "insufficient_information"
                            else "ranked candidates available"
                        ),
                    )
                )
            )

        return (
            (
                _insert_merged_candidates_run_v2(run_id, group_id, orphan_id, merged)
                ^ _log_stage_metric_run_v2(
                    run_id,
                    group_id,
                    orphan_id,
                    "merge",
                    "candidate_merge",
                    merged.length,
                    f"top{MAX_MERGED_CANDIDATES_FOR_API2}",
                )
                ^ _log_anchor_and_candidates_run_v2(
                    orphan_id,
                    valid_anchors,
                    variants_used,
                    scored_candidates,
                    merged,
                    [],
                )
                ^ _build_rank_request_run_v2(dossier, merged)
            )
            >> (
                lambda rank_request: _log_rank_api_input_run_v2(orphan_id, rank_request)
                ^ put_line(f"[K][{orphan_id}] rank_api_call: start")
                ^ _call_rank_run_v2(rank_request)
            )
            >> _build_ranked_work
        )

    def _candidate_stage(
        dossier: Dossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        anchor_failures: list[str],
    ) -> Run[OrphanWork]:
        query_specs = _query_specs_from_validated_anchors(valid_anchors)

        def _score_and_rank_grouped_candidates(
            grouped_candidates: Array[Array[C2Candidate]],
        ) -> Run[OrphanWork]:
            flattened = Array.make(
                tuple(
                    candidate
                    for candidate_group in grouped_candidates
                    for candidate in candidate_group
                )
            )

            def _attach_splink_weights(weights: dict[str, float]) -> Run[OrphanWork]:
                scored_candidates = Array.make(
                    tuple(
                        C2Candidate(
                            entity_uid=candidate.entity_uid,
                            source_article_id=candidate.source_article_id,
                            source_incident_idx=candidate.source_incident_idx,
                            midpoint_day=candidate.midpoint_day,
                            day_gap=candidate.day_gap,
                            compat_count=candidate.compat_count,
                            summary_cosine=candidate.summary_cosine,
                            det_score=candidate.det_score,
                            query_variant=candidate.query_variant,
                            splink_match_weight=weights.get(candidate.entity_uid),
                        )
                        for candidate in flattened
                    )
                )
                requested_count = len(
                    {
                        candidate.entity_uid
                        for candidate in scored_candidates
                        if candidate.entity_uid != ""
                    }
                )
                unscored_count = sum(
                    1
                    for candidate in scored_candidates
                    if candidate.splink_match_weight is None
                )
                return (
                    put_line(
                        f"[K][{orphan_id}] splink_c2_weight_scoring: "
                        f"requested_candidates={requested_count}, "
                        f"scored_candidates={len(weights)}, "
                        f"unscored_candidates={unscored_count}"
                    )
                    ^ _log_stage_metric_run_v2(
                        run_id,
                        group_id,
                        orphan_id,
                        "splink_c2_weight",
                        "candidate_entity_vs_orphan",
                        len(weights),
                        f"missing={unscored_count}",
                    )
                    ^ _insert_candidates_run_v2(
                        run_id, group_id, orphan_id, scored_candidates
                    )
                    ^ _log_stage_metric_run_v2(
                        run_id,
                        group_id,
                        orphan_id,
                        "C2",
                        "fts_variant_union",
                        scored_candidates.length,
                        f"variants={query_specs.length}",
                    )
                    ^ _rank_stage(
                        dossier,
                        valid_anchors,
                        variants_used,
                        anchor_failures,
                        scored_candidates,
                        weights,
                    )
                )

            return _score_c2_candidate_weights_run_v2(orphan_id, flattened) >> (
                _attach_splink_weights
            )

        return (
            _log_stage_metric_run_v2(
                run_id, group_id, orphan_id, "B", "bypassed", 0, "bypassed_focus_c2"
            )
            ^ _log_stage_metric_run_v2(
                run_id, group_id, orphan_id, "C", "bypassed", 0, "bypassed_focus_c2"
            )
            ^ (
                array_traverse_run(
                    query_specs,
                    lambda query_spec: _fts_article_hits_run_v2(
                        query_spec.fts_query, 60
                    )
                    >> (
                        lambda article_ids: _entities_for_articles_run_v2(
                            article_ids, orphan_id, query_spec.display_text
                        )
                    ),
                )
                if query_specs.length > 0
                else pure(Array.empty())
            )
        ) >> _score_and_rank_grouped_candidates

    def _validated_stage(
        dossier: Dossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        anchor_failures: list[str],
    ) -> Run[OrphanWork]:
        return _log_stage_metric_run_v2(
            run_id,
            group_id,
            orphan_id,
            "pass1",
            "anchor_validation",
            valid_anchors.length,
            ";".join(anchor_failures),
        ) ^ (
            _log_anchor_and_candidates_run_v2(
                orphan_id,
                valid_anchors,
                Array.empty(),
                Array.empty(),
                Array.empty(),
                [],
            )
            ^ pure(
                OrphanWork(
                    orphan_id=orphan_id,
                    article_id=dossier.article_id,
                    insufficient=True,
                    insufficient_reason="pass1",
                    stage_trace=[
                        {
                            "stage": "pass1",
                            "row_count": valid_anchors.length,
                            "gate": "fail",
                            "anchor_failures": anchor_failures,
                        }
                    ],
                    anchors=[
                        _validated_anchor_to_dict_v2(anchor) for anchor in valid_anchors
                    ],
                    valid_variants=list(variants_used),
                    merged_candidates=[],
                    ranked_candidates=[],
                    provisional_reason="insufficient anchors from pass1",
                )
            )
            if _pass1_gate_failed(
                [_validated_anchor_to_dict_v2(anchor) for anchor in valid_anchors],
                anchor_failures,
            )
            else _candidate_stage(
                dossier, valid_anchors, variants_used, anchor_failures
            )
        )

    def _call_pass1_stage(dossier: Dossier) -> Run[GPTResponseTuple]:
        request = _build_pass1_request_v2(dossier)
        return (
            _display_article_run_v2(dossier)
            ^ _log_pass1_api_input_run_v2(orphan_id, request)
            ^ put_line(f"[K][{orphan_id}] pass1_api_call: start")
            ^ _call_pass1_run_v2(request)
        )

    def _finalize_pass1_response(
        dossier: Dossier, pass1_t: GPTResponseTuple
    ) -> Run[OrphanWork]:
        return (
            put_line(f"[K][{orphan_id}] pass1_api_call: completed")
            ^ _log_api_usage_and_reasoning_run_v2(orphan_id, "pass1", pass1_t)
            ^ put_line(f"[K][{orphan_id}] pass1_result (api_call):")
            ^ put_line(
                _format_pass1_response_for_display(
                    cast(Pass1Response, pass1_t.parsed.output)
                )
            )
            ^ _validate_anchors_run_v2(cast(Pass1Response, pass1_t.parsed.output))
        ) >> (
            lambda validated: _validated_stage(
                dossier, validated[0], validated[1], validated[2]
            )
        )

    return _load_orphan_dossier_run_v2(orphan_id) >> (
        lambda dossier: _call_pass1_stage(dossier)
        >> (lambda pass1_t: _finalize_pass1_response(dossier, pass1_t))
    )


def _process_group_run_v2(
    run_id: str, group: Array[CacheReadinessRow]
) -> Run[Array[CaseDecision]]:
    if group.length == 0:
        return pure(Array.empty())
    group_id = group[0].group_id
    leader = group[0]
    leader_orphan_id = leader.orphan_id
    group_orphan_ids = tuple(row.orphan_id for row in group)
    group_key = _incident_group_key_from_orphan_id(leader_orphan_id)

    def _mark_group_in_progress() -> Run[Any]:
        return array_traverse_run(
            group,
            lambda row: _set_case_state_run_v2(
                run_id,
                row.orphan_id,
                case_status="in_progress",
                stage_completed="start",
            ),
        )

    def _log_shared_stage_metric(
        stage: str, operation: str, row_count: int, notes: str
    ) -> Run[Any]:
        return array_traverse_run(
            group,
            lambda row: _log_stage_metric_run_v2(
                run_id, group_id, row.orphan_id, stage, operation, row_count, notes
            ),
        )

    def _insert_shared_candidates(candidates: Array[C2Candidate]) -> Run[Any]:
        return array_traverse_run(
            group,
            lambda row: _insert_candidates_run_v2(
                run_id, group_id, row.orphan_id, candidates
            ),
        )

    def _insert_shared_representatives(
        representatives: Array[IncidentRepresentativeCandidate],
    ) -> Run[Any]:
        return array_traverse_run(
            group,
            lambda row: _insert_incident_representatives_run_v2(
                run_id, group_id, row.orphan_id, representatives
            ),
        )

    def _build_shared_works(
        *,
        article_id: int | None,
        anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        ranked_candidates: list[dict[str, Any]],
        stage_trace: list[dict[str, Any]],
        provisional_reason: str,
        insufficient: bool = False,
        insufficient_reason: str | None = None,
    ) -> Array[OrphanWork]:
        anchor_dicts = [_validated_anchor_to_dict_v2(anchor) for anchor in anchors]
        return Array.make(
            tuple(
                OrphanWork(
                    orphan_id=row.orphan_id,
                    article_id=article_id,
                    insufficient=insufficient,
                    insufficient_reason=insufficient_reason,
                    stage_trace=stage_trace,
                    anchors=anchor_dicts,
                    valid_variants=list(variants_used),
                    merged_candidates=ranked_candidates,
                    ranked_candidates=ranked_candidates,
                    provisional_reason=provisional_reason,
                )
                for row in group
            )
        )

    def _finalize_decisions_from_group_work(
        work_items: Array[OrphanWork],
        assigned_candidates: dict[str, dict[str, Any]],
    ) -> Run[Array[CaseDecision]]:
        return pure(
            Array.make(
                tuple(
                    _decision_from_work(work, assigned_candidates.get(work.orphan_id))
                    for work in work_items
                )
            )
        )

    def _analysis_incomplete_decisions(message: str) -> Array[CaseDecision]:
        return Array.make(
            tuple(
                CaseDecision(
                    orphan_id=row.orphan_id,
                    article_id=row.article_id,
                    label="analysis_incomplete",
                    resolved_entity_id=None,
                    confidence=None,
                    reason_summary=(
                        "Required adjudication stages failed due to execution error"
                        " and the case must be retried."
                    ),
                    evidence_json={
                        "stage_trace": [
                            {
                                "stage": "error",
                                "row_count": 0,
                                "gate": "fail",
                                "error": message,
                            }
                        ],
                        "reason_code": "execution_failure",
                        "execution_audit": {
                            "query_mode": "interactive_sql",
                            "fts_used": False,
                            "fts_query_list": [],
                        },
                    },
                )
                for row in group
            )
        )

    def _handle_group_result(result: Any) -> Run[Array[CaseDecision]]:
        if isinstance(result, Right):
            return pure(result.r)
        message = _safe_text(result.l)
        return (
            array_traverse_run(
                group,
                lambda row: put_line(
                    f"[K][{row.orphan_id}] processing_error: {message}"
                )
                ^ _log_stage_metric_run_v2(
                    run_id,
                    group_id,
                    row.orphan_id,
                    "error",
                    "exception",
                    0,
                    message[:500],
                )
                ^ _set_case_state_run_v2(
                    run_id,
                    row.orphan_id,
                    case_status="failed",
                    stage_completed="error",
                    decision_label="analysis_incomplete",
                    error_message=message[:800],
                ),
            )
            ^ pure(_analysis_incomplete_decisions(message))
        )

    def _load_group_dossier() -> Run[GroupDossier]:
        return _load_orphan_dossier_run_v2(leader_orphan_id) >> (
            lambda dossier: pure(
                _group_dossier_from_dossier(group_key, group_orphan_ids, dossier)
            )
        )

    def _build_assignment_decisions(
        group_dossier: GroupDossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        representatives: Array[IncidentRepresentativeCandidate],
        ranked_candidates: list[dict[str, Any]],
    ) -> Run[Array[CaseDecision]]:
        selected = ranked_candidates[0] if len(ranked_candidates) > 0 else {}
        allowed_entities = tuple(
            sorted(
                _safe_text(uid)
                for uid in (selected.get("eligible_entity_uids") or [])
                if _safe_text(uid) != ""
            )
        )
        if len(allowed_entities) == 0:
            return throw(
                ErrorPayload(
                    "selected_incident_has_no_eligible_entities_after_exclusion"
                )
            )
        entity_candidates = Array.make(
            tuple(
                C2Candidate(
                    entity_uid=entity_uid,
                    source_article_id=selected.get("source_article_id"),
                    source_incident_idx=selected.get("source_incident_idx"),
                    midpoint_day=None,
                    day_gap=0.0,
                    compat_count=0,
                    summary_cosine=0.0,
                    det_score=0.0,
                    query_variant=_safe_text(selected.get("query_variant")),
                )
                for entity_uid in allowed_entities
            )
        )

        def _collect_weight_maps(
            weight_rows: Array[tuple[str, dict[str, float]]],
        ) -> Run[Array[CaseDecision]]:
            weight_by_orphan = dict(weight_rows)
            assignment = _maximize_weight_assignment_v2(
                group_orphan_ids, allowed_entities, weight_by_orphan
            )
            shared_stage_trace = [
                {
                    "stage": "pass1",
                    "row_count": valid_anchors.length,
                    "gate": "pass",
                    "group_size": len(group_orphan_ids),
                },
                {"stage": "B", "row_count": 0, "gate": "bypassed"},
                {"stage": "C", "row_count": 0, "gate": "bypassed"},
                {
                    "stage": "C2",
                    "row_count": representatives.length,
                    "gate": "pass",
                },
                {
                    "stage": "rank",
                    "row_count": len(ranked_candidates),
                    "gate": "pass",
                },
                {
                    "stage": "assignment",
                    "row_count": len(assignment),
                    "gate": "pass",
                },
            ]
            work_items = _build_shared_works(
                article_id=group_dossier.article_id,
                anchors=valid_anchors,
                variants_used=variants_used,
                ranked_candidates=ranked_candidates,
                stage_trace=shared_stage_trace,
                provisional_reason="matched incident selected but orphan unassigned",
            )
            assigned_candidates = {
                orphan_id: _make_assignment_candidate_dict_v2(
                    entity_uid=entity_uid,
                    source_article_id=selected.get("source_article_id"),
                    source_incident_idx=selected.get("source_incident_idx"),
                    splink_match_weight=_as_float(
                        weight_by_orphan.get(orphan_id, {}).get(entity_uid), 0.0
                    ),
                    ranked_incidents=ranked_candidates,
                )
                for orphan_id, entity_uid in assignment.items()
            }
            return _finalize_decisions_from_group_work(work_items, assigned_candidates)

        return array_traverse_run(
            group,
            lambda row: _score_c2_candidate_weights_run_v2(
                row.orphan_id, entity_candidates
            ).map(lambda weights: (row.orphan_id, weights)),
        ) >> _collect_weight_maps

    def _finalize_rank_response(
        group_dossier: GroupDossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        representatives: Array[IncidentRepresentativeCandidate],
        rank_t: GPTResponseTuple,
    ) -> Run[Array[CaseDecision]]:
        rank_response = cast(RankResponse, rank_t.parsed.output)
        ranked = _validate_incident_rank_output_v2(rank_response, representatives)
        ranked_for_evidence = ranked + [
            candidate.to_dict()
            for candidate in representatives
            if candidate.entity_uid
            not in {_safe_text(row.get("entity_uid")) for row in ranked}
        ]
        shared_stage_trace = [
            {
                "stage": "pass1",
                "row_count": valid_anchors.length,
                "gate": "pass",
                "group_size": len(group_orphan_ids),
            },
            {"stage": "B", "row_count": 0, "gate": "bypassed"},
            {"stage": "C", "row_count": 0, "gate": "bypassed"},
            {"stage": "C2", "row_count": representatives.length, "gate": "pass"},
            {"stage": "rank", "row_count": len(ranked), "gate": "pass"},
        ]
        if rank_response.match_result == "insufficient_information":
            return _finalize_decisions_from_group_work(
                _build_shared_works(
                    article_id=group_dossier.article_id,
                    anchors=valid_anchors,
                    variants_used=variants_used,
                    ranked_candidates=ranked_for_evidence,
                    stage_trace=shared_stage_trace,
                    provisional_reason="insufficient information from rank api",
                    insufficient=True,
                    insufficient_reason="rank",
                ),
                {},
            )
        if rank_response.match_result != "match" or len(ranked) == 0:
            return _finalize_decisions_from_group_work(
                _build_shared_works(
                    article_id=group_dossier.article_id,
                    anchors=valid_anchors,
                    variants_used=variants_used,
                    ranked_candidates=ranked_for_evidence,
                    stage_trace=shared_stage_trace,
                    provisional_reason="no matching incident from rank api",
                ),
                {},
            )
        return _build_assignment_decisions(
            group_dossier,
            valid_anchors,
            variants_used,
            representatives,
            ranked_for_evidence,
        )

    def _rank_group_candidates(
        group_dossier: GroupDossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        c2_candidates: Array[C2Candidate],
        representatives: Array[IncidentRepresentativeCandidate],
    ) -> Run[Array[CaseDecision]]:
        return (
            _insert_shared_candidates(c2_candidates)
            ^ _insert_shared_representatives(representatives)
            ^ _log_shared_stage_metric(
                "C2",
                "fts_variant_union",
                c2_candidates.length,
                f"variants={variants_used.length}",
            )
            ^ _log_shared_stage_metric(
                "merge",
                "candidate_incident_merge",
                representatives.length,
                f"top{MAX_MERGED_CANDIDATES_FOR_API2}",
            )
            ^ _log_anchor_and_candidates_run_v2(
                leader_orphan_id,
                valid_anchors,
                variants_used,
                c2_candidates,
                representatives,
                [],
            )
            ^ _build_rank_request_for_group_run_v2(group_dossier, representatives)
        ) >> (
            lambda rank_request: _log_rank_api_input_run_v2(
                leader_orphan_id, rank_request
            )
            ^ put_line(f"[K][{leader_orphan_id}] rank_api_call: start")
            ^ _call_rank_run_v2(rank_request)
        ) >> (
            lambda rank_t: put_line(
                f"[K][{leader_orphan_id}] rank_api_call: completed"
            )
            ^ _log_api_usage_and_reasoning_run_v2(leader_orphan_id, "rank", rank_t)
            ^ put_line(f"[K][{leader_orphan_id}] rank_result (api_call):")
            ^ put_line(
                _pretty_json(
                    json.loads(
                        cast(RankResponse, rank_t.parsed.output).model_dump_json()
                    )
                )
            )
            ^ _finalize_rank_response(
                group_dossier, valid_anchors, variants_used, representatives, rank_t
            )
        )

    def _candidate_stage(
        group_dossier: GroupDossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
    ) -> Run[Array[CaseDecision]]:
        query_specs = _query_specs_from_validated_anchors(valid_anchors)

        def _build_representatives(
            grouped_candidates: Array[Array[C2Candidate]],
        ) -> Run[Array[CaseDecision]]:
            base_candidates = Array.make(
                tuple(
                    candidate
                    for candidate_group in grouped_candidates
                    for candidate in candidate_group
                )
            )
            return array_traverse_run(
                group,
                lambda row: _score_group_candidates_for_orphan_run_v2(
                    row.orphan_id, base_candidates
                ),
            ) >> (
                lambda per_orphan_rows: _candidate_incident_rows_run_v2(
                    _aggregate_group_c2_candidates_v2(per_orphan_rows)
                )
                >> (
                    lambda incident_rows: _rank_group_candidates(
                        group_dossier,
                        valid_anchors,
                        variants_used,
                        _aggregate_group_c2_candidates_v2(per_orphan_rows),
                        _incident_representatives_from_rows_v2(incident_rows),
                    )
                )
            )

        return (
            _log_shared_stage_metric("B", "bypassed", 0, "bypassed_focus_c2")
            ^ _log_shared_stage_metric("C", "bypassed", 0, "bypassed_focus_c2")
            ^ (
                array_traverse_run(
                    query_specs,
                    lambda query_spec: _fts_article_hits_run_v2(
                        query_spec.fts_query, 60
                    )
                    >> (
                        lambda article_ids: _entities_for_articles_group_run_v2(
                            article_ids,
                            group_dossier.article_id,
                            query_spec.display_text,
                        )
                    ),
                )
                if query_specs.length > 0
                else pure(Array.empty())
            )
        ) >> _build_representatives

    def _handle_validated_anchors(
        group_dossier: GroupDossier,
        valid_anchors: Array[ValidatedAnchor],
        variants_used: Array[str],
        anchor_failures: list[str],
    ) -> Run[Array[CaseDecision]]:
        shared_stage_trace = [
            {
                "stage": "pass1",
                "row_count": valid_anchors.length,
                    "gate": (
                        "fail"
                        if _pass1_gate_failed(
                            [
                                _validated_anchor_to_dict_v2(anchor)
                                for anchor in valid_anchors
                            ],
                            anchor_failures,
                        )
                        else "pass"
                ),
                "anchor_failures": anchor_failures,
                "group_size": len(group_orphan_ids),
            }
        ]
        return _log_shared_stage_metric(
            "pass1",
            "anchor_validation",
            valid_anchors.length,
            ";".join(anchor_failures),
        ) ^ (
            _log_anchor_and_candidates_run_v2(
                leader_orphan_id,
                valid_anchors,
                Array.empty(),
                Array.empty(),
                Array.empty(),
                [],
            )
            ^ _finalize_decisions_from_group_work(
                _build_shared_works(
                    article_id=group_dossier.article_id,
                    anchors=valid_anchors,
                    variants_used=variants_used,
                    ranked_candidates=[],
                    stage_trace=shared_stage_trace,
                    provisional_reason="insufficient anchors from pass1",
                    insufficient=True,
                    insufficient_reason="pass1",
                ),
                {},
            )
            if _pass1_gate_failed(
                [_validated_anchor_to_dict_v2(anchor) for anchor in valid_anchors],
                anchor_failures,
            )
            else _candidate_stage(group_dossier, valid_anchors, variants_used)
        )

    def _run_group_pipeline(group_dossier: GroupDossier) -> Run[Array[CaseDecision]]:
        request = _build_pass1_request_for_group_v2(group_dossier)
        return (
            _display_article_run_v2(
                _dossier_from_dict_v2(
                    {
                        "unique_id": leader_orphan_id,
                        "article_id": group_dossier.article_id,
                        "city_id": group_dossier.city_id,
                        "year": group_dossier.year,
                        "month": group_dossier.month,
                        "midpoint_day": group_dossier.midpoint_day,
                        "date_precision": "",
                        "incident_date": group_dossier.incident_date,
                        "victim_count": group_dossier.victim_count,
                        "weapon": "",
                        "circumstance": "",
                        "geo_address_norm": group_dossier.incident_location,
                        "geo_address_short": "",
                        "geo_address_short_2": "",
                        "relationship": "",
                        "incident_summary_gpt": group_dossier.incident_summary,
                        "article_title": group_dossier.article_title,
                        "article_text": group_dossier.article_text,
                        "article_pub_date": group_dossier.article_pub_date,
                    }
                )
            )
            ^ _log_pass1_api_input_run_v2(leader_orphan_id, request)
            ^ put_line(f"[K][{leader_orphan_id}] pass1_api_call: start")
            ^ _call_pass1_run_v2(request)
        ) >> (
            lambda pass1_t: put_line(
                f"[K][{leader_orphan_id}] pass1_api_call: completed"
            )
            ^ _log_api_usage_and_reasoning_run_v2(leader_orphan_id, "pass1", pass1_t)
            ^ put_line(f"[K][{leader_orphan_id}] pass1_result (api_call):")
            ^ put_line(
                _format_pass1_response_for_display(
                    cast(Pass1Response, pass1_t.parsed.output)
                )
            )
            ^ _validate_anchors_run_v2(cast(Pass1Response, pass1_t.parsed.output))
        ) >> (
            lambda validated: _handle_validated_anchors(
                group_dossier, validated[0], validated[1], validated[2]
            )
        )

    return (
        _mark_group_in_progress()
        ^ run_except(_load_group_dossier() >> _run_group_pipeline)
    ) >> _handle_group_result


def _run_pipeline_run(
    params: KParams, readiness_rows: Array[CacheReadinessRow]
) -> Run[RunSummary]:
    run_id = f"k_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    selected = _select_needs_api_rows_v2(readiness_rows, params)
    groups = _build_groups_v2(selected, params.group_same_incident, run_id)
    selected_ids = {row.orphan_id for row in selected}

    def _finish(grouped_decisions: Array[Array[CaseDecision]]) -> Run[RunSummary]:
        new_decisions = Array.make(
            tuple(decision for group in grouped_decisions for decision in group)
        )

        def _combine_cached_and_live_decisions(
            finalized: Array[CaseDecision],
        ) -> Run[RunSummary]:
            def _rebuild_from_decisions(cached: Array[CaseDecision]) -> Run[RunSummary]:
                all_decisions = Array.make(tuple(finalized.a + cached.a))

                def _persist_run_summary(
                    rebuilt_terminal_rows: int,
                ) -> Run[RunSummary]:
                    matched = sum(
                        1 for decision in finalized if decision.label == "matched"
                    )
                    not_same = sum(
                        1
                        for decision in finalized
                        if decision.label == "not_same_person"
                    )
                    insufficient = sum(
                        1
                        for decision in finalized
                        if decision.label == "insufficient_information"
                    )
                    incomplete = sum(
                        1
                        for decision in finalized
                        if decision.label == "analysis_incomplete"
                    )
                    return sql_exec(
                        SQL(
                            """
                            UPDATE orphan_adj_run
                            SET
                              finished_at = NOW(),
                              processed_count = ?,
                              grouped_count = ?,
                              matched_count = ?,
                              not_same_person_count = ?,
                              insufficient_information_count = ?,
                              analysis_incomplete_count = ?,
                              status = 'completed'
                            WHERE run_id = ?;
                            """
                        ),
                        _sql_params(
                            (
                                finalized.length,
                                groups.length,
                                matched,
                                not_same,
                                insufficient,
                                incomplete,
                                run_id,
                            )
                        ),
                    ) ^ pure(
                        RunSummary(
                            needs_api_total=sum(
                                1
                                for row in readiness_rows
                                if row.readiness_status == "needs_api"
                            ),
                            selected_needs_api=selected.length,
                            decision_ready_from_cache=sum(
                                1
                                for row in readiness_rows
                                if row.readiness_status
                                == "decision_ready_from_cache"
                            ),
                            processed=finalized.length,
                            grouped=groups.length,
                            matched=matched,
                            not_same_person=not_same,
                            insufficient_information=insufficient,
                            analysis_incomplete=incomplete,
                            rebuilt_terminal_rows=rebuilt_terminal_rows,
                            dry_run=params.dry_run,
                        )
                    )

                return _rebuild_overrides_table_run_v2(
                    run_id, all_decisions, dry_run=params.dry_run
                ) >> _persist_run_summary

            return _cached_decisions_from_readiness_run_v2(
                readiness_rows, selected_ids
            ) >> _rebuild_from_decisions

        def _flatten_finalized_groups(
            finalized_groups: Array[Array[CaseDecision]],
        ) -> Run[RunSummary]:
            return _combine_cached_and_live_decisions(
                Array.make(
                    tuple(
                        decision
                        for finalized_group in finalized_groups
                        for decision in finalized_group
                    )
                )
            )

        return (
            array_traverse_run(
                grouped_decisions,
                lambda decisions: _finalize_group_decisions_run_v2(run_id, decisions),
            )
            if new_decisions.length > 0
            else pure(Array.empty())
        ) >> _flatten_finalized_groups

    return (
        _ensure_tables_run_v2()
        ^ _migrate_legacy_labels_run_v2()
        ^ sql_exec(
            SQL(
                """
            INSERT INTO orphan_adj_run (
              run_id, started_at, requested_limit, processed_count, grouped_count,
              matched_count, not_same_person_count, insufficient_information_count,
              analysis_incomplete_count, dry_run, full_backfill, status
            ) VALUES (?, NOW(), ?, 0, 0, 0, 0, 0, 0, ?, ?, 'running');
            """
            ),
            _sql_params((run_id, params.limit, params.dry_run, params.full_backfill)),
        )
        ^ _insert_queue_rows_run_v2(run_id, groups)
        >> (
            lambda _: run_except(
                array_traverse_run(
                    groups, lambda group: _process_group_run_v2(run_id, group)
                )
                if groups.length > 0
                else pure(Array.empty())
            )
            >> (
                lambda result: _finish(result.r)
                if isinstance(result, Right)
                else (
                    sql_exec(
                        SQL(
                            """
                    UPDATE orphan_adj_run
                    SET
                      finished_at = NOW(),
                      processed_count = 0,
                      grouped_count = ?,
                      matched_count = 0,
                      not_same_person_count = 0,
                      insufficient_information_count = 0,
                      analysis_incomplete_count = 1,
                      status = 'failed',
                      error_message = ?
                    WHERE run_id = ?;
                    """
                        ),
                        _sql_params(
                            (groups.length, _safe_text(result.l)[:800], run_id)
                        ),
                    )
                    ^ throw(ErrorPayload(f"{result.l}"))
                )
            )
        )
    )


def _prepare_cache_readiness() -> Run[tuple[Array[CacheReadinessRow], int]]:
    return _precompute_cache_readiness_run() >> (
        lambda prep: put_line(f"[K] API 1 JSON schema:\n{to_json(Pass1Response)}")
        ^ put_line(f"[K] API 2 JSON schema:\n{to_json(RankResponse)}")
        ^ put_line(_format_cache_readiness_summary_v2(prep[0]))
        ^ pure(prep)
    )


def _execute_k(
    params: KParams, readiness_rows: Array[CacheReadinessRow]
) -> Run[NextStep]:
    def _validate_prompt_templates_and_execute(
        env: Any, pass1_prompt_template: Any
    ) -> Run[NextStep]:
        if pass1_prompt_template.id.startswith("pmpt_replace_"):
            return (
                put_line(
                    f"[K] Stored prompt key '{PASS1_PROMPT_KEY}' is still a placeholder"
                    f" id ({pass1_prompt_template.id}). Replace it with your dashboard"
                    " prompt_id and rerun."
                )
                ^ pure(NextStep.CONTINUE)
            )

        def _handle_rank_prompt(rank_prompt_template: Any) -> Run[NextStep]:
            if rank_prompt_template.id.startswith("pmpt_replace_"):
                return (
                    put_line(
                        f"[K] Stored prompt key '{RANK_PROMPT_KEY}' is still a"
                        f" placeholder id ({rank_prompt_template.id}). Replace it"
                        " with your dashboard prompt_id and rerun."
                    )
                    ^ pure(NextStep.CONTINUE)
                )
            return _run_pipeline_run(params, readiness_rows) >> _display_k_summary

        return resolve_prompt_template(env, PromptKey(RANK_PROMPT_KEY)) >> (
            _handle_rank_prompt
        )

    def _display_k_summary(summary: RunSummary) -> Run[NextStep]:
        return put_line(
            (
                "[K] Orphan adjudication completed:"
                f" needs_api_total={summary.needs_api_total},"
                f" selected_needs_api={summary.selected_needs_api},"
                " decision_ready_from_cache="
                f"{summary.decision_ready_from_cache},"
                f" processed={summary.processed},"
                f" groups={summary.grouped},"
                f" matched={summary.matched},"
                " not_same_person="
                f"{summary.not_same_person},"
                " insufficient_information="
                f"{summary.insufficient_information},"
                " analysis_incomplete="
                f"{summary.analysis_incomplete},"
                " rebuilt_terminal_rows="
                f"{summary.rebuilt_terminal_rows},"
                f" dry_run={summary.dry_run}"
            )
        ) ^ pure(NextStep.CONTINUE)

    return ask() >> (
        lambda env: resolve_prompt_template(env, PromptKey(PASS1_PROMPT_KEY))
        >> (lambda pass1_prompt_template: _validate_prompt_templates_and_execute(
            env, pass1_prompt_template
        ))
    )


def _prompt_and_execute_k(
    readiness_rows: Array[CacheReadinessRow], needs_api_total: int
) -> Run[NextStep]:
    limit_prompt = (
        f"Enter number of records requiring new API calls [{needs_api_total}]: "
    )
    def _collect_remaining_prompts(limit_raw: Any) -> Run[NextStep]:
        parsed_limit = _parse_limit(str(limit_raw), default=needs_api_total)
        if parsed_limit == 0:
            return put_line("[K] limit=0; returning to main menu.") ^ pure(
                NextStep.CONTINUE
            )

        def _build_params(
            start_after_raw: Any, group_raw: Any, dry_raw: Any, full_raw: Any
        ) -> KParams:
            return KParams(
                limit=parsed_limit,
                starting_after_orphan_id=str(start_after_raw).strip(),
                group_same_incident=_parse_bool(str(group_raw), True),
                dry_run=_parse_bool(str(dry_raw), False),
                full_backfill=_parse_bool(str(full_raw), False),
            )

        return input_with_prompt(PromptKey("k_start_after")) >> (
            lambda start_after_raw: input_with_prompt(
                PromptKey("k_group_same_incident")
            )
            >> (
                lambda group_raw: input_with_prompt(PromptKey("k_dry_run"))
                >> (
                    lambda dry_raw: input_with_prompt(PromptKey("k_full_backfill"))
                    >> (
                        lambda full_raw: _execute_k(
                            _build_params(
                                start_after_raw, group_raw, dry_raw, full_raw
                            ),
                            readiness_rows,
                        )
                    )
                )
            )
        )

    return input_with_prompt(InputPrompt(limit_prompt)) >> _collect_remaining_prompts


def adjudicate_orphans_controller() -> Run[NextStep]:
    """Run controller [K] under the application's standard Run stack."""

    prog = _prepare_cache_readiness() >> (
        lambda prep: _prompt_and_execute_k(prep[0], prep[1])
    )
    return with_namespace(
        Namespace("orphan_adj_k"),
        to_prompts(ORPHAN_ADJ_PROMPTS),
        with_models(
            {
                PASS1_MODEL_KEY: GPTModel.GPT_5_MINI,
                RANK_MODEL_KEY: GPTModel.GPT_5_MINI,
            },
            with_duckdb(prog),
        ),
    )
