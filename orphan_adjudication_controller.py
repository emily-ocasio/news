"""
Controller [K]: Orphan adjudication replacement flow with deterministic staging,
LLM-assisted anchor extraction/ranking, and idempotent caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Literal, cast

import duckdb
from splink import DuckDBAPI, Linker
from openai import OpenAI
from openai.types.responses import ResponsePromptParam
from openai.types.responses.response_prompt_param import Variables
from pydantic import BaseModel, ConfigDict, RootModel
from article import Article
from pymonad.openai import GPTModel, GPTUsage, GPTPromptTemplate
from blocking import ORPHAN_DETERMINISTIC_BLOCKS
from splink_types import SplinkType

from menuprompts import NextStep
from pymonad import (
    DbBackend,
    InputPrompt,
    Namespace,
    PromptKey,
    Run,
    ask,
    input_with_prompt,
    put_line,
    pure,
    resolve_prompt_template,
    to_json,
    to_prompts,
    with_namespace,
)

PASS1_MODEL = "gpt-5-mini"
RANK_MODEL = "gpt-5-mini"
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
    pass


class Anchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anchor_theme: str
    variants: list[WordSet]


class Pass1Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anchors: list[Anchor]


class RankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match_result: Literal["match", "no_match", "insufficient_information"]
    matched_entity_uid: str | None


@dataclass(frozen=True)
class KParams:
    limit: int
    starting_after_orphan_id: str
    group_same_incident: bool
    dry_run: bool
    full_backfill: bool


@dataclass
class CaseDecision:
    orphan_id: str
    article_id: int | None
    label: str
    resolved_entity_id: str | None
    confidence: float | None
    reason_summary: str
    evidence_json: dict[str, Any]


@dataclass
class OrphanWork:
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


@dataclass(frozen=True)
class KRuntimeDeps:
    duck_con: duckdb.DuckDBPyConnection
    sqlite_con: sqlite3.Connection
    openai_client: OpenAI


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        raise RuntimeError("dedupe_splink_model_missing: run [D] before C2 Splink scoring")
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


def _log_k(message: str) -> None:
    print(message, flush=True)


def _with_elapsed_timer(fn: Any) -> Any:
    stop = threading.Event()
    start = time.monotonic()

    def tick() -> None:
        while not stop.wait(1):
            elapsed = int(time.monotonic() - start)
            minutes, seconds = divmod(elapsed, 60)
            print(f"\rElapsed time: {minutes:02d}:{seconds:02d}", end="", flush=True)

    thread = threading.Thread(target=tick, daemon=True)
    print("\rElapsed time: 00:00", end="", flush=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join()
        elapsed = int(time.monotonic() - start)
        minutes, seconds = divmod(elapsed, 60)
        print(f"\rElapsed time: {minutes:02d}:{seconds:02d}")


def _pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


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


def _duck_query(duck_con: duckdb.DuckDBPyConnection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur = duck_con.execute(sql, params)
    cols = [c[0] for c in (cur.description or [])]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _duck_exec(duck_con: duckdb.DuckDBPyConnection, sql: str, params: tuple[Any, ...] = ()) -> None:
    duck_con.execute(sql, params)


def _sqlite_query(sqlite_con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur = sqlite_con.execute(sql, params)
    cols = [c[0] for c in (cur.description or [])]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _ensure_tables(duck_con: duckdb.DuckDBPyConnection) -> None:
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(
        duck_con,
        "ALTER TABLE orphan_adj_candidates_c2 ADD COLUMN IF NOT EXISTS splink_match_weight DOUBLE;",
    )
    _duck_exec(
        duck_con,
        "ALTER TABLE orphan_adj_candidates_merged ADD COLUMN IF NOT EXISTS splink_match_weight DOUBLE;",
    )
    _duck_exec(duck_con, 
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
    )
    _duck_exec(duck_con, 
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
    )


def _migrate_legacy_labels(duck_con: duckdb.DuckDBPyConnection) -> None:
    _duck_exec(duck_con, 
        """
        UPDATE orphan_adjudication_overrides
        SET resolution_label = 'matched',
            updated_at = NOW()
        WHERE resolution_label = 'likely_missed_match';
        """
    )
    _duck_exec(duck_con, 
        """
        UPDATE orphan_adjudication_overrides
        SET resolution_label = 'not_same_person',
            updated_at = NOW()
        WHERE resolution_label = 'unlikely';
        """
    )
    _duck_exec(duck_con, 
        """
        UPDATE orphan_adjudication_overrides
        SET resolution_label = 'analysis_incomplete',
            updated_at = NOW()
        WHERE resolution_label = 'possible_but_weak';
        """
    )


def _queue_universe_rows(duck_con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    return _duck_query(
        duck_con,
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
              qb.match_id,
              qb.orphan_id
          ) AS queue_pos
        FROM queue_base qb
        ORDER BY queue_pos;
        """,
    )


def _materialize_cache_readiness(
    duck_con: duckdb.DuckDBPyConnection,
    run_id: str,
    rows: list[dict[str, Any]],
) -> None:
    _duck_exec(duck_con, "DELETE FROM orphan_adj_cache_readiness WHERE run_id = ?;", (run_id,))
    for r in rows:
        _duck_exec(duck_con, 
            """
            INSERT INTO orphan_adj_cache_readiness (
              run_id, queue_pos, orphan_id, article_id, year, readiness_status, readiness_reason,
              pass1_idempotency_key, rank_idempotency_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NOW());
            """,
            (
                run_id,
                r.get("queue_pos"),
                _safe_text(r.get("orphan_id")),
                r.get("article_id"),
                r.get("year"),
                _safe_text(r.get("readiness_status")),
                _safe_text(r.get("readiness_reason")),
                _safe_text(r.get("pass1_idempotency_key")),
                _safe_text(r.get("rank_idempotency_key")),
            ),
        )


def _count_by_year(rows: list[dict[str, Any]], *, status: str | None = None) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in rows:
        if status is not None and _safe_text(r.get("readiness_status")) != status:
            continue
        year_val = r.get("year")
        year_key = "unknown" if year_val is None else str(year_val)
        counts[year_key] = counts.get(year_key, 0) + 1
    return sorted(counts.items(), key=lambda t: (t[0] == "unknown", t[0]))


def _format_cache_readiness_summary(rows: list[dict[str, Any]]) -> str:
    needs_total = sum(1 for r in rows if _safe_text(r.get("readiness_status")) == "needs_api")

    lines: list[str] = ["[K] Orphans requiring new API calls by year:"]
    for year_key, n in _count_by_year(rows, status="needs_api"):
        lines.append(f"  {year_key}: {n}")
    lines.append(f"  total: {needs_total}")
    return "\n".join(lines)


def _select_needs_api_rows(readiness_rows: list[dict[str, Any]], params: KParams) -> list[dict[str, Any]]:
    needs_rows = [r for r in readiness_rows if _safe_text(r.get("readiness_status")) == "needs_api"]
    needs_rows.sort(key=lambda r: (int(r.get("queue_pos") or 0), _safe_text(r.get("orphan_id"))))

    if params.starting_after_orphan_id.strip() != "":
        start_pos = None
        for r in needs_rows:
            if _safe_text(r.get("orphan_id")) == params.starting_after_orphan_id:
                start_pos = int(r.get("queue_pos") or 0)
                break
        if start_pos is not None:
            needs_rows = [r for r in needs_rows if int(r.get("queue_pos") or 0) > start_pos]

    if params.full_backfill:
        return needs_rows
    return needs_rows[: params.limit]


def _precompute_cache_readiness(
    duck_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], int]:
    _ensure_tables(duck_con)
    universe = _queue_universe_rows(duck_con)
    out_rows: list[dict[str, Any]] = []

    for base in universe:
        orphan_id = _safe_text(base.get("orphan_id"))
        try:
            dossier = _load_orphan_dossier(duck_con, sqlite_con, orphan_id)
            e2e_key = _adjudication_cache_key(dossier)
            e2e_cached = _llm_cache_get(duck_con, E2E_CACHE_STAGE, e2e_key)
            if e2e_cached is None:
                out_rows.append(
                    {
                        **base,
                        "readiness_status": "needs_api",
                        "readiness_reason": "missing_e2e_cache",
                        "pass1_idempotency_key": e2e_key,
                        "rank_idempotency_key": "",
                    }
                )
                continue

            out_rows.append(
                {
                    **base,
                    "readiness_status": "decision_ready_from_cache",
                    "readiness_reason": "e2e_cache_ready",
                    "pass1_idempotency_key": e2e_key,
                    "rank_idempotency_key": "",
                }
            )
        except Exception as ex:  # noqa: BLE001
            out_rows.append(
                {
                    **base,
                    "readiness_status": "needs_api",
                    "readiness_reason": f"readiness_error:{_safe_text(ex)[:160]}",
                    "pass1_idempotency_key": "",
                    "rank_idempotency_key": "",
                }
            )

    preview_run_id = f"k_preview_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _materialize_cache_readiness(duck_con, preview_run_id, out_rows)
    needs_total = sum(1 for r in out_rows if _safe_text(r.get("readiness_status")) == "needs_api")
    return out_rows, needs_total


def _incident_compatible(prev_row: dict[str, Any], row: dict[str, Any]) -> bool:
    if prev_row.get("article_id") == row.get("article_id"):
        return True
    if prev_row.get("city_id") != row.get("city_id"):
        return False

    p_year = prev_row.get("year")
    r_year = row.get("year")
    if p_year is not None and r_year is not None and abs(int(p_year) - int(r_year)) > 1:
        return False

    p_mid = prev_row.get("midpoint_day")
    r_mid = row.get("midpoint_day")
    if p_mid is not None and r_mid is not None and abs(float(p_mid) - float(r_mid)) > 7:
        return False

    p_weapon = _norm(_safe_text(prev_row.get("weapon")))
    r_weapon = _norm(_safe_text(row.get("weapon")))
    if p_weapon and r_weapon and p_weapon != r_weapon:
        return False

    p_circ = _norm(_safe_text(prev_row.get("circumstance")))
    r_circ = _norm(_safe_text(row.get("circumstance")))
    if p_circ and r_circ and p_circ != r_circ:
        return False

    return True


def _build_groups(rows: list[dict[str, Any]], group_same_incident: bool, run_id: str) -> list[list[dict[str, Any]]]:
    if not group_same_incident:
        return [[r] for r in rows]
    if len(rows) == 0:
        return []

    grouped: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [rows[0]]
    for row in rows[1:]:
        if _incident_compatible(current[-1], row):
            current.append(row)
        else:
            grouped.append(current)
            current = [row]
    grouped.append(current)

    for i, grp in enumerate(grouped, start=1):
        gid = f"{run_id}_G{i:04d}"
        for r in grp:
            r["group_id"] = gid
    return grouped


def _insert_queue_rows(duck_con: duckdb.DuckDBPyConnection, run_id: str, groups: list[list[dict[str, Any]]]) -> None:
    for grp in groups:
        for r in grp:
            _duck_exec(duck_con, 
                """
                INSERT INTO orphan_adj_queue_run (
                  run_id, queue_pos, group_id, orphan_id, article_id, city_id, year, month, midpoint_day
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    r.get("queue_pos"),
                    r.get("group_id"),
                    r.get("orphan_id"),
                    r.get("article_id"),
                    r.get("city_id"),
                    r.get("year"),
                    r.get("month"),
                    r.get("midpoint_day"),
                ),
            )
            _duck_exec(duck_con, 
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
                """,
                (
                    run_id,
                    r.get("group_id"),
                    r.get("orphan_id"),
                    r.get("article_id"),
                ),
            )


def _set_case_state(
    duck_con: duckdb.DuckDBPyConnection,
    run_id: str,
    orphan_id: str,
    *,
    case_status: str,
    stage_completed: str,
    decision_label: str | None = None,
    resolved_entity_id: str | None = None,
    decision_hash: str | None = None,
    error_message: str | None = None,
) -> None:
    _duck_exec(duck_con, 
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
        """,
        (
            case_status,
            stage_completed,
            decision_label,
            resolved_entity_id,
            decision_hash,
            error_message,
            run_id,
            orphan_id,
        ),
    )


def _log_stage_metric(
    duck_con: duckdb.DuckDBPyConnection,
    run_id: str,
    group_id: str,
    orphan_id: str,
    stage_name: str,
    query_id: str,
    row_count: int,
    notes: str,
) -> None:
    _duck_exec(duck_con, 
        """
        INSERT INTO orphan_adj_stage_metrics (
          run_id, group_id, orphan_id, stage_name, query_id, row_count, notes, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NOW());
        """,
        (run_id, group_id, orphan_id, stage_name, query_id, row_count, notes),
    )


def _llm_cache_get(
    duck_con: duckdb.DuckDBPyConnection,
    stage: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    rows = _duck_query(
        duck_con,
        """
        SELECT response_json
        FROM llm_cache
        WHERE stage = ?
          AND idempotency_key = ?
        LIMIT 1;
        """,
        (stage, idempotency_key),
    )
    if len(rows) == 0:
        return None
    raw = rows[0].get("response_json")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return None


def _llm_cache_put(
    duck_con: duckdb.DuckDBPyConnection,
    *,
    stage: str,
    idempotency_key: str,
    model: str,
    prompt_version: str,
    input_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> None:
    _duck_exec(duck_con, 
        """
        INSERT INTO llm_cache (
          stage, idempotency_key, model, prompt_version, input_json, response_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?::JSON, ?::JSON, NOW(), NOW())
        ON CONFLICT(stage, idempotency_key) DO UPDATE SET
          model = EXCLUDED.model,
          prompt_version = EXCLUDED.prompt_version,
          input_json = EXCLUDED.input_json,
          response_json = EXCLUDED.response_json,
          updated_at = NOW();
        """,
        (
            stage,
            idempotency_key,
            model,
            prompt_version,
            _canonical_json(input_payload),
            _canonical_json(response_payload),
        ),
    )


def _case_fingerprint(payload: dict[str, Any], *, model: str, prompt_version: str) -> str:
    return _sha256(_canonical_json({"model": model, "prompt_version": prompt_version, "payload": payload}))


def _adjudication_cache_key(dossier: dict[str, Any]) -> str:
    pass1_input = _build_pass1_input(dossier)
    return _case_fingerprint(pass1_input, model=PASS1_MODEL, prompt_version=PASS1_PROMPT_VERSION)


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


def _decision_from_cache_payload(payload: dict[str, Any], fallback_orphan_id: str) -> CaseDecision:
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
        orphan_id=_safe_text(payload.get("orphan_id")) or fallback_orphan_id,
        article_id=article_id,
        label=_safe_text(payload.get("label")),
        resolved_entity_id=_safe_text(payload.get("resolved_entity_id")) or None,
        confidence=confidence,
        reason_summary=_safe_text(payload.get("reason_summary")),
        evidence_json=evidence_json,
    )


def _load_orphan_dossier(
    duck_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
    orphan_id: str,
) -> dict[str, Any]:
    def _incident_idx_from_orphan_uid(uid: str) -> int | None:
        parts = uid.split(":")
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    rows = _duck_query(
        duck_con,
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
          victim_sex,
          victim_age,
          victim_count,
          offender_count,
          weapon,
          circumstance,
          geo_address_norm,
          geo_address_short,
          geo_address_short_2,
          relationship,
          summary_vec
        FROM orphan_link_input
        WHERE unique_id = ?
        LIMIT 1;
        """,
        (orphan_id,),
    )
    if len(rows) == 0:
        return {"orphan_id": orphan_id}
    row = dict(rows[0])
    row["incident_summary_gpt"] = ""
    article_id = row.get("article_id")
    if article_id is None:
        row["article_text"] = ""
        row["article_title"] = ""
        return row

    incident_idx = _incident_idx_from_orphan_uid(_safe_text(row.get("unique_id")))
    if incident_idx is not None:
        sum_rows = _duck_query(
            duck_con,
            """
            SELECT summary
            FROM incidents_cached
            WHERE article_id = ?
              AND incident_idx = ?
            LIMIT 1;
            """,
            (int(article_id), incident_idx),
        )
        if len(sum_rows) > 0:
            row["incident_summary_gpt"] = _safe_text(sum_rows[0].get("summary"))

    text_rows = _sqlite_query(
        sqlite_con,
        """
        SELECT Title, FullText, PubDate
        FROM articles
        WHERE RecordId = ?
          AND Dataset = 'CLASS_WP'
          AND gptClass = 'M'
        LIMIT 1;
        """,
        (int(article_id),),
    )
    if len(text_rows) == 0:
        row["article_text"] = ""
        row["article_title"] = ""
        row["article_pub_date"] = ""
    else:
        row["article_title"] = _safe_text(text_rows[0].get("Title"))
        row["article_text"] = _safe_text(text_rows[0].get("FullText"))[:MAX_FULLTEXT_CHARS]
        row["article_pub_date"] = _safe_text(text_rows[0].get("PubDate"))
    return row


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


def _call_pass1_api(
    client: OpenAI,
    payload: dict[str, Any],
    prompt_template: GPTPromptTemplate,
) -> tuple[Pass1Response, Any]:
    prompt_vars = {k: cast(Variables, v) for k, v in _stringify_prompt_vars(payload).items()}
    prompt_dict = ResponsePromptParam(id=prompt_template.id, variables=prompt_vars)
    if prompt_template.version is not None:
        prompt_dict["version"] = prompt_template.version

    response = _with_elapsed_timer(
        lambda: client.responses.parse(
            model=PASS1_MODEL,
            prompt=prompt_dict,
            text_format=Pass1Response,
            reasoning={"effort": "medium", "summary": "auto"},
            text={"verbosity": "low"},
            timeout=300.0,
        )
    )
    parsed = response.output_parsed
    if parsed is None:
        return Pass1Response(anchors=[]), response
    return parsed, response


def _call_rank_api(
    client: OpenAI,
    payload: dict[str, Any],
    prompt_template: GPTPromptTemplate,
) -> tuple[RankResponse, Any]:
    prompt_vars = {k: cast(Variables, v) for k, v in _stringify_prompt_vars(payload).items()}
    prompt_dict = ResponsePromptParam(id=prompt_template.id, variables=prompt_vars)
    if prompt_template.version is not None:
        prompt_dict["version"] = prompt_template.version

    response = _with_elapsed_timer(
        lambda: client.responses.parse(
            model=RANK_MODEL,
            prompt=prompt_dict,
            text_format=RankResponse,
            reasoning={"effort": "medium", "summary": "auto"},
            text={"verbosity": "low"},
            timeout=300.0,
        )
    )
    parsed = response.output_parsed
    if parsed is None:
        return RankResponse(match_result="no_match", matched_entity_uid=None), response
    return parsed, response


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


def _anchor_doc_frequency(sqlite_con: sqlite3.Connection, anchor_text: str) -> int:
    text = anchor_text.strip()
    if text == "":
        return 0
    rows = _sqlite_query(
        sqlite_con,
        """
        SELECT COUNT(*) AS n
        FROM articles
        WHERE Dataset='CLASS_WP'
          AND gptClass='M'
          AND FullText LIKE ?;
        """,
        (f"%{text}%",),
    )
    if len(rows) == 0:
        return 0
    return int(rows[0].get("n") or 0)


def _validate_anchors(
    sqlite_con: sqlite3.Connection,
    dossier: dict[str, Any],
    pass1: Pass1Response,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    valid_anchors: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []

    for a in pass1.anchors:
        anchor_text = _safe_text(a.anchor_theme).strip()
        norm_anchor = _norm(anchor_text)
        if norm_anchor == "":
            rejected_reasons.append("empty_anchor")
            continue

        doc_freq = _anchor_doc_frequency(sqlite_con, anchor_text)
        if doc_freq > HIGH_FREQ_ANCHOR_DOC_THRESHOLD:
            rejected_reasons.append(f"high_freq_anchor:{anchor_text[:40]}:{doc_freq}")
            continue

        variant_word_sets: list[list[str]] = []
        for ws in a.variants:
            words = [_safe_text(w).strip() for w in ws.root if _safe_text(w).strip() != ""]
            if len(words) == 0:
                continue
            variant_word_sets.append(words)

        # Canonicalize variant word-sets to keep idempotency stable.
        seen_word_sets: set[tuple[str, ...]] = set()
        deduped_word_sets: list[list[str]] = []
        for words in variant_word_sets:
            signature = tuple(_norm(w) for w in words if _norm(w) != "")
            if len(signature) == 0 or signature in seen_word_sets:
                continue
            seen_word_sets.add(signature)
            deduped_word_sets.append(words)

        filtered_variants: list[str] = []
        for words in deduped_word_sets:
            v = " ".join(words).strip()
            nv = _norm(v)
            if nv == "":
                continue
            filtered_variants.append(v)

        filtered_variant_word_sets = deduped_word_sets
        if len(filtered_variants) == 0:
            filtered_variants = [anchor_text]
            filtered_variant_word_sets = [[anchor_text]]

        valid_anchors.append(
            {
                "anchor_text": anchor_text,
                "anchor_type": "phrase",
                "source_span": "",
                "incident_link_reason": "",
                "variants": filtered_variants,
                "variant_word_sets": filtered_variant_word_sets,
                "doc_freq": doc_freq,
            }
        )

    anchor_types = {a["anchor_type"] for a in valid_anchors if _safe_text(a.get("anchor_type")) != ""}
    all_variants = list(dict.fromkeys(v for a in valid_anchors for v in a["variants"]))

    gate_failures: list[str] = []
    if len(valid_anchors) < 1:
        gate_failures.append("no_valid_anchors")
    if len(anchor_types) < 1:
        gate_failures.append("insufficient_anchor_diversity")
    if len(all_variants) < 1:
        gate_failures.append("no_query_variants")

    return valid_anchors, all_variants, rejected_reasons + gate_failures


def _pass1_gate_failed(valid_anchors: list[dict[str, Any]], anchor_failures: list[str]) -> bool:
    if len(valid_anchors) < 1:
        return True
    return any(
        failure in {"no_valid_anchors", "insufficient_anchor_diversity", "no_query_variants"}
        for failure in anchor_failures
    )


def _candidate_rows_bc(
    duck_con: duckdb.DuckDBPyConnection,
    orphan_id: str,
    stage_name: str,
    year_window: int,
    limit_n: int,
) -> list[dict[str, Any]]:
    rows = _duck_query(
        duck_con,
        """
        WITH o AS (
          SELECT *
          FROM orphan_link_input
          WHERE unique_id = ?
        )
        SELECT
          e.unique_id AS entity_uid,
          e.article_ids_csv,
          e.city_id,
          e.year,
          e.midpoint_day,
          e.victim_sex,
          e.weapon,
          e.circumstance,
          e.summary_vec,
          ABS(COALESCE(CAST(e.midpoint_day AS DOUBLE), 0) - COALESCE(o.midpoint_day, 0)) AS day_gap,
          CASE WHEN e.victim_sex = o.victim_sex THEN 1 ELSE 0 END AS sex_match,
          CASE WHEN e.weapon = o.weapon THEN 1 ELSE 0 END AS weapon_match,
          CASE WHEN e.circumstance = o.circumstance THEN 1 ELSE 0 END AS circumstance_match,
          array_cosine_similarity(e.summary_vec, o.summary_vec) AS summary_cosine
        FROM entity_link_input e
        CROSS JOIN o
        WHERE e.city_id = o.city_id
          AND ABS(COALESCE(e.year, 0) - COALESCE(o.year, 0)) <= ?
        ORDER BY
          (CASE WHEN e.circumstance = o.circumstance THEN 1 ELSE 0 END
           + CASE WHEN e.victim_sex = o.victim_sex THEN 1 ELSE 0 END
           + CASE WHEN e.weapon = o.weapon THEN 1 ELSE 0 END) DESC,
          ABS(COALESCE(CAST(e.midpoint_day AS DOUBLE), 0) - COALESCE(o.midpoint_day, 0)) ASC,
          e.unique_id
        LIMIT ?;
        """,
        (
            orphan_id,
            year_window,
            limit_n,
        ),
    )
    for r in rows:
        compat = _as_int(r.get("sex_match"), 0) + _as_int(r.get("weapon_match"), 0) + _as_int(r.get("circumstance_match"), 0)
        cosine = _as_float(r.get("summary_cosine"), 0.0)
        day_gap = _as_float(r.get("day_gap"), 9999.0)
        r["stage_name"] = stage_name
        r["compat_count"] = compat
        r["det_score"] = 0.8 * compat + 1.5 * cosine - 0.01 * day_gap
    return rows


def _fts_article_hits(sqlite_con: sqlite3.Connection, query_variant: str, limit_n: int = 50) -> list[int]:
    q = query_variant.strip()
    if q == "":
        return []

    def _run(match_q: str) -> list[int]:
        rows = _sqlite_query(
            sqlite_con,
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
            (match_q, limit_n),
        )
        ids = [int(r["RecordId"]) for r in rows if r.get("RecordId") is not None]
        # Stable dedupe/order so cache key generation is deterministic across runs.
        return sorted(dict.fromkeys(ids))

    try:
        return _run(q)
    except sqlite3.OperationalError:
        # Fallback to a strict phrase query when free-form syntax is invalid for FTS5.
        phrase_q = '"' + re.sub(r"\s+", " ", q).replace('"', '""').strip() + '"'
        try:
            return _run(phrase_q)
        except sqlite3.OperationalError:
            return []


def _entities_for_articles(
    duck_con: duckdb.DuckDBPyConnection,
    article_ids: list[int],
    orphan_id: str,
) -> list[dict[str, Any]]:
    if len(article_ids) == 0:
        return []
    values_sql = ",".join("(?)" for _ in article_ids)
    params: list[Any] = [*article_ids, orphan_id]
    rows = _duck_query(
        duck_con,
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
          e.article_ids_csv,
          e.midpoint_day AS midpoint_day,
          ABS(COALESCE(CAST(e.midpoint_day AS DOUBLE), 0) - COALESCE(o.midpoint_day, 0)) AS day_gap,
          CASE WHEN e.victim_sex = o.victim_sex THEN 1 ELSE 0 END AS sex_match,
          CASE WHEN e.weapon = o.weapon THEN 1 ELSE 0 END AS weapon_match,
          CASE WHEN e.circumstance = o.circumstance THEN 1 ELSE 0 END AS circumstance_match,
          array_cosine_similarity(e.summary_vec, o.summary_vec) AS summary_cosine
        FROM hits h
        JOIN entity_link_input e
          ON list_contains(string_split(COALESCE(e.article_ids_csv, ''), ','), CAST(h.article_id AS VARCHAR))
        CROSS JOIN o;
        """,
        tuple(params),
    )
    for r in rows:
        compat = _as_int(r.get("sex_match"), 0) + _as_int(r.get("weapon_match"), 0) + _as_int(r.get("circumstance_match"), 0)
        cosine = _as_float(r.get("summary_cosine"), 0.0)
        day_gap = _as_float(r.get("day_gap"), 9999.0)
        r["stage_name"] = "C2"
        r["compat_count"] = compat
        r["det_score"] = 1.2 + 0.8 * compat + 1.7 * cosine - 0.01 * day_gap
    rows.sort(
        key=lambda r: (
            _safe_text(r.get("entity_uid")),
            _as_int(r.get("source_article_id"), 0),
            -_as_float(r.get("det_score"), 0.0),
        )
    )
    return rows


def _score_c2_candidate_weights(
    duck_con: duckdb.DuckDBPyConnection,
    orphan_id: str,
    c2_rows: list[dict[str, Any]],
) -> dict[str, float]:
    candidate_ids = sorted(
        {
            _safe_text(r.get("entity_uid"))
            for r in c2_rows
            if _safe_text(r.get("entity_uid")) != ""
        }
    )
    if len(candidate_ids) == 0:
        return {}

    settings = _settings_for_c2_weight_scoring(_load_dedupe_model_settings())
    suffix = _sha256(f"{orphan_id}|{'|'.join(candidate_ids)}")[:12]
    left_table = _sanitize_sql_identifier(f"adj_c2_left_{suffix}", prefix="tmp")
    right_table = _sanitize_sql_identifier(f"adj_c2_right_{suffix}", prefix="tmp")
    placeholders = ", ".join("?" for _ in candidate_ids)
    prediction_table = ""
    rows: list[dict[str, Any]] = []
    try:
        _duck_exec(
            duck_con,
            f"""
            CREATE OR REPLACE TEMP TABLE {left_table} AS
            SELECT *
            FROM entity_link_input
            WHERE unique_id IN ({placeholders});
            """,
            tuple(candidate_ids),
        )
        _duck_exec(
            duck_con,
            f"""
            CREATE OR REPLACE TEMP TABLE {right_table} AS
            SELECT *
            FROM orphan_link_input
            WHERE unique_id = ?;
            """,
            (orphan_id,),
        )

        db_api = DuckDBAPI(connection=duck_con)
        linker = Linker(
            [left_table, right_table],
            settings
            | {
                "retain_intermediate_calculation_columns": True,
                "max_iterations": 100,
                "blocking_rules_to_generate_predictions": ["1=1"],
            },
            db_api=db_api,
        )
        adj_model_key = str(SplinkType.ORPHAN_ADJ_SCORE).replace("/", "_")
        adj_model_path = Path("splink_models") / f"splink_model_{adj_model_key}.json"
        adj_model_path.parent.mkdir(parents=True, exist_ok=True)
        linker.misc.save_model_to_json(str(adj_model_path), overwrite=True)
        linker.training.estimate_probability_two_random_records_match(
            list(ORPHAN_DETERMINISTIC_BLOCKS),
            recall=0.1,
        )
        prediction_table = _safe_text(
            linker.inference.predict(threshold_match_probability=0.0).physical_name
        )
        rows = _duck_query(
            duck_con,
            f"""
            SELECT
              CAST(unique_id_l AS VARCHAR) AS entity_uid,
              match_weight
            FROM {prediction_table}
            WHERE CAST(unique_id_r AS VARCHAR) = ?;
            """,
            (orphan_id,),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"splink_c2_weight_scoring_failed: {exc}") from exc
    finally:
        if prediction_table != "":
            _duck_exec(duck_con, f"DROP TABLE IF EXISTS {prediction_table};")
        _duck_exec(duck_con, f"DROP TABLE IF EXISTS {left_table};")
        _duck_exec(duck_con, f"DROP TABLE IF EXISTS {right_table};")

    weights: dict[str, float] = {}
    for row in rows:
        entity_uid = _safe_text(row.get("entity_uid"))
        if entity_uid == "":
            continue
        weight_raw = row.get("match_weight")
        if weight_raw is None:
            continue
        weight_val = _as_float(weight_raw, 0.0)
        current = weights.get(entity_uid)
        if current is None or weight_val > current:
            weights[entity_uid] = weight_val
    return weights


def _insert_candidates(
    duck_con: duckdb.DuckDBPyConnection,
    run_id: str,
    group_id: str,
    orphan_id: str,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    if table_name == "bc":
        for r in rows:
            _duck_exec(duck_con, 
                """
                INSERT INTO orphan_adj_candidates_bc (
                  run_id, group_id, orphan_id, stage_name, entity_uid, det_score, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?::JSON);
                """,
                (
                    run_id,
                    group_id,
                    orphan_id,
                    _safe_text(r.get("stage_name")),
                    _safe_text(r.get("entity_uid")),
                    float(r.get("det_score") or 0.0),
                    _canonical_json(r),
                ),
            )
    elif table_name == "c2":
        for r in rows:
            _duck_exec(duck_con, 
                """
                INSERT INTO orphan_adj_candidates_c2 (
                  run_id, group_id, orphan_id, entity_uid, source_article_id, query_variant,
                  det_score, splink_match_weight, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::JSON);
                """,
                (
                    run_id,
                    group_id,
                    orphan_id,
                    _safe_text(r.get("entity_uid")),
                    r.get("source_article_id"),
                    _safe_text(r.get("query_variant")),
                    float(r.get("det_score") or 0.0),
                    _as_float(r.get("splink_match_weight"), 0.0)
                    if r.get("splink_match_weight") is not None
                    else None,
                    _canonical_json(r),
                ),
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
                "source_article_hits": set([r.get("source_article_id")]) if r.get("source_article_id") is not None else set(),
                "rows": [r],
            }
            continue

        existing["det_score"] = max(_as_float(existing["det_score"], 0.0), _as_float(r.get("det_score"), 0.0))
        existing["day_gap"] = min(_as_float(existing["day_gap"], 9999.0), _as_float(r.get("day_gap"), 9999.0))
        existing["compat_count"] = max(_as_int(existing["compat_count"], 0), _as_int(r.get("compat_count"), 0))
        existing["summary_cosine"] = max(_as_float(existing["summary_cosine"], 0.0), _as_float(r.get("summary_cosine"), 0.0))
        row_weight = (
            _as_float(r.get("splink_match_weight"), 0.0)
            if r.get("splink_match_weight") is not None
            else None
        )
        existing_weight = existing.get("splink_match_weight")
        if row_weight is not None and (
            existing_weight is None or _as_float(row_weight, 0.0) > _as_float(existing_weight, 0.0)
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
                "source_article_id": source_article_ids[0] if len(source_article_ids) > 0 else None,
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


def _insert_merged_candidates(
    duck_con: duckdb.DuckDBPyConnection,
    run_id: str,
    group_id: str,
    orphan_id: str,
    merged: list[dict[str, Any]],
) -> None:
    for r in merged:
        _duck_exec(duck_con, 
            """
            INSERT INTO orphan_adj_candidates_merged (
              run_id, group_id, orphan_id, entity_uid, det_score, splink_match_weight, source_stages, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?::JSON);
            """,
            (
                run_id,
                group_id,
                orphan_id,
                _safe_text(r.get("entity_uid")),
                float(r.get("det_score") or 0.0),
                _as_float(r.get("splink_match_weight"), 0.0)
                if r.get("splink_match_weight") is not None
                else None,
                ",".join(r.get("source_stages") or []),
                _canonical_json(r),
            ),
        )


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


def _build_pass1_input(dossier: dict[str, Any]) -> dict[str, Any]:
    incident_summary = _safe_text(dossier.get("incident_summary_gpt"))
    return {
        "article_title": _safe_text(dossier.get("article_title")),
        "article_text": _safe_text(dossier.get("article_text")),
        "article_date": _format_article_date(dossier.get("article_pub_date")),
        "incident_date": _format_incident_date(
            dossier.get("incident_date"),
            year=dossier.get("year"),
            month=dossier.get("month"),
        ),
        "incident_location": _safe_text(dossier.get("geo_address_norm")),
        "victim_count": _safe_text(dossier.get("victim_count")),
        "incident_summary": incident_summary,
    }


def _candidate_source_article_id(merged_row: dict[str, Any]) -> int | None:
    direct_src = merged_row.get("source_article_id")
    if direct_src is not None:
        try:
            return int(_safe_text(direct_src))
        except (TypeError, ValueError):
            pass

    src_ids = merged_row.get("source_article_ids")
    if isinstance(src_ids, list):
        parsed_ids: list[int] = []
        for raw in src_ids:
            try:
                parsed_ids.append(int(_safe_text(raw)))
            except (TypeError, ValueError):
                continue
        if len(parsed_ids) > 0:
            return min(parsed_ids)

    rows = merged_row.get("rows")
    if isinstance(rows, list):
        parsed_row_ids: list[int] = []
        for r in rows:
            if isinstance(r, dict) and r.get("source_article_id") is not None:
                try:
                    src_id = r.get("source_article_id")
                    if src_id is None:
                        continue
                    parsed_row_ids.append(int(_safe_text(src_id)))
                except (TypeError, ValueError):
                    continue
        if len(parsed_row_ids) > 0:
            return min(parsed_row_ids)
    return None


def _candidate_article_context(sqlite_con: sqlite3.Connection, article_id: int | None) -> dict[str, Any]:
    if article_id is None:
        return {
            "article_title": "",
            "article_text": "",
            "article_date": "",
        }
    rows = _sqlite_query(
        sqlite_con,
        """
        SELECT RecordId, Title, FullText, PubDate
        FROM articles
        WHERE RecordId = ?
          AND Dataset = 'CLASS_WP'
          AND gptClass = 'M'
        LIMIT 1;
        """,
        (article_id,),
    )
    if len(rows) == 0:
        return {
            "article_title": "",
            "article_text": "",
            "article_date": "",
        }
    return {
        "article_title": _safe_text(rows[0].get("Title")),
        "article_text": _safe_text(rows[0].get("FullText"))[:MAX_FULLTEXT_CHARS],
        "article_date": _format_article_date(rows[0].get("PubDate")),
    }


def _candidate_incident_context(
    duck_con: duckdb.DuckDBPyConnection,
    entity_uid: str,
    source_article_id: int | None,
) -> dict[str, Any]:
    rows = _duck_query(
        duck_con,
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
        (entity_uid,),
    )
    if len(rows) == 0:
        return {
            "incident_date": "",
            "incident_location": "",
            "victim_count": "",
            "incident_summary": "",
        }
    row = rows[0]
    summary = ""
    if source_article_id is not None:
        sum_rows = _duck_query(
            duck_con,
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
            (entity_uid, int(source_article_id)),
        )
        if len(sum_rows) > 0:
            summary = _safe_text(sum_rows[0].get("summary"))
    if summary.strip() == "":
        summary = " | ".join(
            part for part in [
                _safe_text(row.get("circumstance")),
                _safe_text(row.get("relationship")),
            ] if part.strip() != ""
        )
    return {
        "incident_date": _format_incident_date(
            row.get("incident_date"),
            year=row.get("year"),
            month=row.get("month"),
        ),
        "incident_location": _safe_text(row.get("geo_address_norm")),
        "victim_count": _safe_text(row.get("victim_count")),
        "incident_summary": summary,
    }


def _build_rank_input(
    duck_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
    dossier: dict[str, Any],
    valid_anchors: list[dict[str, Any]],
    merged: list[dict[str, Any]],
) -> dict[str, Any]:
    target_summary = _safe_text(dossier.get("incident_summary_gpt"))
    candidates_payload: list[dict[str, Any]] = []
    for r in merged:
        entity_uid = _safe_text(r.get("entity_uid"))
        src_article_id = _candidate_source_article_id(r)
        candidates_payload.append(
            {
                "entity_uid": entity_uid,
                **_candidate_article_context(sqlite_con, src_article_id),
                **_candidate_incident_context(duck_con, entity_uid, src_article_id),
            }
        )
    return {
        "target_article_title": _safe_text(dossier.get("article_title")),
        "target_article_text": _safe_text(dossier.get("article_text")),
        "target_article_date": _format_article_date(dossier.get("article_pub_date")),
        "target_incident_date": _format_incident_date(
            dossier.get("incident_date"),
            year=dossier.get("year"),
            month=dossier.get("month"),
        ),
        "target_incident_location": _safe_text(dossier.get("geo_address_norm")),
        "target_victim_count": _safe_text(dossier.get("victim_count")),
        "target_incident_summary": target_summary,
        "candidates": candidates_payload,
    }


def _display_article_for_orphan(sqlite_con: sqlite3.Connection, dossier: dict[str, Any]) -> None:
    orphan_id = _safe_text(dossier.get("unique_id") or dossier.get("orphan_id"))
    article_id = dossier.get("article_id")
    if article_id is None:
        _log_k(f"[K][{orphan_id}] No article_id available for display.")
        return

    rows = _sqlite_query(
        sqlite_con,
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
        (int(article_id),),
    )
    if len(rows) == 0:
        title = _safe_text(dossier.get("article_title"))
        body = _safe_text(dossier.get("article_text"))
        _log_k(f"[K][{orphan_id}] Article context")
        _log_k(f"[K][{orphan_id}] article_id={article_id} title={title}")
        _log_k(f"[K][{orphan_id}] full_text:\n{body if body != '' else '[empty]'}")
        return

    try:
        article_obj = Article(rows[0], current=0, total=1)
        _log_k(f"[K][{orphan_id}] Retrieved article:\n {article_obj}")
    except Exception as ex:  # noqa: BLE001
        _log_k(f"[K][{orphan_id}] display_article render failed: {_safe_text(ex)}")
        title = _safe_text(dossier.get("article_title"))
        body = _safe_text(dossier.get("article_text"))
        _log_k(f"[K][{orphan_id}] article_id={article_id} title={title}")
        _log_k(f"[K][{orphan_id}] full_text:\n{body if body != '' else '[empty]'}")


def _reasoning_summary_text(response: Any) -> str:
    outputs = getattr(response, "output", None)
    if not isinstance(outputs, list):
        return "[no reasoning summary]"
    lines: list[str] = []
    for item in outputs:
        if getattr(item, "type", None) != "reasoning":
            continue
        summaries = getattr(item, "summary", None) or []
        for summary in summaries:
            text = _safe_text(getattr(summary, "text", ""))
            if text.strip() != "":
                lines.append(text.strip())
    return "\n".join(lines) if len(lines) > 0 else "[no reasoning summary]"


def _usage_summary_text(response: Any) -> str:
    usage = getattr(response, "usage", None)
    model_name = _safe_text(getattr(response, "model", ""))
    if usage is None:
        return f"GPT Usage for this response:\nModel: {model_name or 'None'}, \nNo usage data.\n"

    input_tokens = _as_int(getattr(usage, "input_tokens", 0), 0)
    output_tokens = _as_int(getattr(usage, "output_tokens", 0), 0)
    total_tokens = _as_int(getattr(usage, "total_tokens", input_tokens + output_tokens), input_tokens + output_tokens)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    cached_tokens = _as_int(getattr(input_details, "cached_tokens", 0), 0)
    reasoning_tokens = _as_int(getattr(output_details, "reasoning_tokens", 0), 0)
    model_enum = GPTModel.from_string(model_name) if model_name else None
    return str(
        GPTUsage(
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            model_used=model_enum,
        )
    )


def _log_api_usage_and_reasoning(orphan_id: str, stage: str, response: Any) -> None:
    _log_k(f"[K][{orphan_id}] {stage} usage/cost:")
    _log_k(_usage_summary_text(response))
    _log_k(f"[K][{orphan_id}] {stage} GPT reasoning summary:")
    _log_k(_reasoning_summary_text(response))


def _log_pass1_api_result(orphan_id: str, payload: dict[str, Any], cached: bool, response: Any | None = None) -> None:
    mode = "cache_hit" if cached else "api_call"
    _log_k(f"[K][{orphan_id}] pass1_result ({mode}):")
    _log_k(_pretty_json(payload))
    if not cached and response is not None:
        _log_api_usage_and_reasoning(orphan_id, "pass1", response)
    elif cached:
        _log_k(f"[K][{orphan_id}] pass1 usage/cost: [cache_hit: no live API call]")
        _log_k(f"[K][{orphan_id}] pass1 GPT reasoning summary: [cache_hit: no live API response]")


def _log_rank_api_result(orphan_id: str, payload: dict[str, Any], cached: bool, response: Any | None = None) -> None:
    mode = "cache_hit" if cached else "api_call"
    _log_k(f"[K][{orphan_id}] rank_result ({mode}):")
    _log_k(_pretty_json(payload))
    if not cached and response is not None:
        _log_api_usage_and_reasoning(orphan_id, "rank", response)
    elif cached:
        _log_k(f"[K][{orphan_id}] rank usage/cost: [cache_hit: no live API call]")
        _log_k(f"[K][{orphan_id}] rank GPT reasoning summary: [cache_hit: no live API response]")


def _log_rank_api_input(orphan_id: str, payload: dict[str, Any]) -> None:
    _log_k(f"[K][{orphan_id}] rank_input_variables:")
    _log_k(_pretty_json(payload))


def _log_pass1_api_input(orphan_id: str, payload: dict[str, Any]) -> None:
    _log_k(f"[K][{orphan_id}] pass1_input_variables:")
    _log_k(_pretty_json(payload))


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


def _format_c2_candidates_table(rows: list[dict[str, Any]], *, include_anchor_hit_count: bool = False) -> str:
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
        article_id = _safe_text(r.get("article_id") if r.get("article_id") is not None else r.get("source_article_id"))
        query_variant = _safe_text(r.get("query_variant"))
        midpoint_day = f"{_as_float(r.get('midpoint_day'), 0.0):.0f}" if r.get("midpoint_day") is not None else ""
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


def _log_anchor_and_candidates(
    orphan_id: str,
    *,
    valid_anchors: list[dict[str, Any]],
    variants_used: list[str],
    b_rows: list[dict[str, Any]],
    c_rows: list[dict[str, Any]],
    c2_rows: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
) -> None:
    _log_k(f"[K][{orphan_id}] anchor_usage:")
    _log_k(f"[K][{orphan_id}] C2 text-phrase criteria count={len(variants_used)}")
    _log_k(_pretty_json({"c2_text_phrases": variants_used}))
    _log_k(
        _pretty_json(
            {
                "anchors": [
                    {
                        "anchor_text": _safe_text(a.get("anchor_text")),
                        "variants": a.get("variants") or [],
                    }
                    for a in valid_anchors
                ]
            }
        )
    )
    _log_k(f"[K][{orphan_id}] candidates_B count={len(b_rows)}")
    _log_k(_pretty_json({"candidates_B": _compact_candidate_view(b_rows)}))
    _log_k(f"[K][{orphan_id}] candidates_C count={len(c_rows)}")
    _log_k(_pretty_json({"candidates_C": _compact_candidate_view(c_rows)}))
    _log_k(f"[K][{orphan_id}] candidates_C2 count={len(c2_rows)}")
    _log_k(_format_c2_candidates_table(c2_rows))
    _log_k(f"[K][{orphan_id}] candidates_merged count={len(merged)}")
    _log_k(_format_c2_candidates_table(merged, include_anchor_hit_count=True))
    _log_k(f"[K][{orphan_id}] candidates_ranked count={len(ranked)}")
    _log_k(_pretty_json({"candidates_ranked": _compact_candidate_view(ranked)}))


def _upsert_decision(
    duck_con: duckdb.DuckDBPyConnection,
    run_id: str,
    decision: CaseDecision,
    *,
    dry_run: bool,
) -> tuple[bool, str]:
    decision_payload = {
        "label": decision.label,
        "resolved_entity_id": decision.resolved_entity_id,
        "reason_summary": decision.reason_summary,
        "evidence_digest": {
            "top_candidates": decision.evidence_json.get("top_candidates"),
            "reason_code": decision.evidence_json.get("reason_code"),
        },
    }
    decision_hash = _sha256(_canonical_json(decision_payload))

    _set_case_state(
        duck_con,
        run_id,
        decision.orphan_id,
        case_status="completed" if decision.label != "analysis_incomplete" else "failed",
        stage_completed="decision",
        decision_label=decision.label,
        resolved_entity_id=decision.resolved_entity_id,
        decision_hash=decision_hash,
        error_message=None,
    )

    if decision.label == "analysis_incomplete" or dry_run:
        return False, decision_hash

    prior_rows = _duck_query(
        duck_con,
        """
        SELECT orphan_id, resolution_label, resolved_entity_id, confidence, reason_summary, evidence_json
        FROM orphan_adjudication_overrides
        WHERE orphan_id = ?
        LIMIT 1;
        """,
        (decision.orphan_id,),
    )
    prior = prior_rows[0] if len(prior_rows) > 0 else None

    prior_hash = None
    if prior is not None:
        ev = prior.get("evidence_json")
        if isinstance(ev, dict):
            prior_hash = _safe_text(ev.get("decision_hash"))
        else:
            try:
                prior_obj = json.loads(_safe_text(ev))
                if isinstance(prior_obj, dict):
                    prior_hash = _safe_text(prior_obj.get("decision_hash"))
            except json.JSONDecodeError:
                prior_hash = None

    changed = prior_hash != decision_hash
    if not changed:
        return False, decision_hash

    prior_label = prior.get("resolution_label") if prior else None
    prior_entity = prior.get("resolved_entity_id") if prior else None
    prior_conf = prior.get("confidence") if prior else None

    _duck_exec(duck_con, 
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
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON, 'interactive_agent_k', NOW()
        );
        """,
        (
            run_id,
            decision.orphan_id,
            prior_label,
            prior_entity,
            prior_conf,
            decision.label,
            decision.resolved_entity_id,
            decision.confidence,
            decision.reason_summary,
            _canonical_json({**decision.evidence_json, "decision_hash": decision_hash}),
        ),
    )

    _duck_exec(duck_con, 
        """
        INSERT INTO orphan_adjudication_overrides (
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
          ?, ?, ?, ?, ?, ?::JSON, 'interactive_agent_k', NOW(), NOW()
        )
        ON CONFLICT(orphan_id) DO UPDATE SET
          resolution_label = EXCLUDED.resolution_label,
          resolved_entity_id = EXCLUDED.resolved_entity_id,
          confidence = EXCLUDED.confidence,
          reason_summary = EXCLUDED.reason_summary,
          evidence_json = EXCLUDED.evidence_json,
          analyst_mode = EXCLUDED.analyst_mode,
          updated_at = NOW();
        """,
        (
            decision.orphan_id,
            decision.label,
            decision.resolved_entity_id,
            decision.confidence,
            decision.reason_summary,
            _canonical_json({**decision.evidence_json, "decision_hash": decision_hash}),
        ),
    )

    return True, decision_hash


def _process_single_orphan(
    duck_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
    client: OpenAI,
    run_id: str,
    group_id: str,
    orphan_id: str,
    pass1_prompt_template: GPTPromptTemplate,
    rank_prompt_template: GPTPromptTemplate,
    *,
    allow_api_calls: bool = True,
) -> OrphanWork:
    stage_trace: list[dict[str, Any]] = []

    dossier = _load_orphan_dossier(duck_con, sqlite_con, orphan_id)
    _display_article_for_orphan(sqlite_con, dossier)
    article_id = dossier.get("article_id")

    pass1_input = _build_pass1_input(dossier)
    if not allow_api_calls:
        raise RuntimeError("api_calls_disabled_for_needs_api_orphan")
    _log_pass1_api_input(orphan_id, pass1_input)
    _log_k(f"[K][{orphan_id}] pass1_api_call: start")
    pass1_resp, pass1_raw = _call_pass1_api(client, pass1_input, pass1_prompt_template)
    _log_k(f"[K][{orphan_id}] pass1_api_call: completed")
    _log_api_usage_and_reasoning(orphan_id, "pass1", pass1_raw)
    pass1_obj = json.loads(pass1_resp.model_dump_json())
    _log_pass1_api_result(orphan_id, pass1_obj, cached=False, response=None)

    valid_anchors, variants, anchor_failures = _validate_anchors(sqlite_con, dossier, pass1_resp)
    pass1_gate_failed = _pass1_gate_failed(valid_anchors, anchor_failures)
    stage_trace.append(
        {
            "stage": "pass1",
            "row_count": len(valid_anchors),
            "gate": "fail" if pass1_gate_failed else "pass",
            "anchor_failures": anchor_failures,
        }
    )
    _log_stage_metric(
        duck_con,
        run_id,
        group_id,
        orphan_id,
        "pass1",
        "anchor_validation",
        len(valid_anchors),
        ";".join(anchor_failures),
    )

    if pass1_gate_failed:
        _log_k(
            f"[K][{orphan_id}] rank_result (skipped): pass1 anchor validation failed; "
            f"reasons={';'.join(anchor_failures)}"
        )
        _log_anchor_and_candidates(
            orphan_id,
            valid_anchors=valid_anchors,
            variants_used=[],
            b_rows=[],
            c_rows=[],
            c2_rows=[],
            merged=[],
            ranked=[],
        )
        return OrphanWork(
            orphan_id=orphan_id,
            article_id=int(article_id) if article_id is not None else None,
            insufficient=True,
            insufficient_reason="pass1",
            stage_trace=stage_trace,
            anchors=valid_anchors,
            valid_variants=variants,
            merged_candidates=[],
            ranked_candidates=[],
            provisional_reason="insufficient anchors from pass1",
        )

    b_rows: list[dict[str, Any]] = []
    c_rows: list[dict[str, Any]] = []
    bc_rows: list[dict[str, Any]] = []
    _log_stage_metric(duck_con, run_id, group_id, orphan_id, "B", "bypassed", 0, "bypassed_focus_c2")
    _log_stage_metric(duck_con, run_id, group_id, orphan_id, "C", "bypassed", 0, "bypassed_focus_c2")
    stage_trace.append({"stage": "B", "row_count": 0, "gate": "bypassed"})
    stage_trace.append({"stage": "C", "row_count": 0, "gate": "bypassed"})

    variants_used = variants
    c2_rows: list[dict[str, Any]] = []
    for variant in variants_used:
        article_hits = _fts_article_hits(sqlite_con, variant, limit_n=60)
        mapped = _entities_for_articles(duck_con, article_hits, orphan_id)
        for m in mapped:
            m["query_variant"] = variant
            c2_rows.append(m)

    splink_weights = _score_c2_candidate_weights(duck_con, orphan_id, c2_rows)
    for row in c2_rows:
        entity_uid = _safe_text(row.get("entity_uid"))
        row["splink_match_weight"] = splink_weights.get(entity_uid)
    missing_splink_weights = sum(
        1 for row in c2_rows if row.get("splink_match_weight") is None
    )
    _log_k(
        f"[K][{orphan_id}] splink_c2_weight_scoring: "
        f"requested_candidates={len({ _safe_text(r.get('entity_uid')) for r in c2_rows if _safe_text(r.get('entity_uid')) != '' })}, "
        f"scored_candidates={len(splink_weights)}, "
        f"unscored_candidates={missing_splink_weights}"
    )
    _log_stage_metric(
        duck_con,
        run_id,
        group_id,
        orphan_id,
        "splink_c2_weight",
        "candidate_entity_vs_orphan",
        len(splink_weights),
        f"missing={missing_splink_weights}",
    )
    stage_trace.append({"stage": "splink_c2_weight", "row_count": len(splink_weights), "gate": "pass"})

    _insert_candidates(duck_con, run_id, group_id, orphan_id, "c2", c2_rows)
    _log_stage_metric(duck_con, run_id, group_id, orphan_id, "C2", "fts_variant_union", len(c2_rows), f"variants={len(variants)}")
    stage_trace.append({"stage": "C2", "row_count": len(c2_rows), "gate": "pass"})

    merged = _merge_candidates([], c2_rows)
    _insert_merged_candidates(duck_con, run_id, group_id, orphan_id, merged)
    _log_stage_metric(
        duck_con,
        run_id,
        group_id,
        orphan_id,
        "merge",
        "candidate_merge",
        len(merged),
        f"top{MAX_MERGED_CANDIDATES_FOR_API2}",
    )
    _log_anchor_and_candidates(
        orphan_id,
        valid_anchors=valid_anchors,
        variants_used=variants_used,
        b_rows=[],
        c_rows=[],
        c2_rows=c2_rows,
        merged=merged,
        ranked=[],
    )

    rank_input = _build_rank_input(duck_con, sqlite_con, dossier, valid_anchors, merged)
    _log_rank_api_input(orphan_id, rank_input)
    _log_k(f"[K][{orphan_id}] rank_api_call: start")
    rank_resp, rank_raw = _call_rank_api(client, rank_input, rank_prompt_template)
    _log_k(f"[K][{orphan_id}] rank_api_call: completed")
    _log_api_usage_and_reasoning(orphan_id, "rank", rank_raw)
    rank_obj = json.loads(rank_resp.model_dump_json())
    _log_rank_api_result(orphan_id, rank_obj, cached=False, response=None)

    rank_insufficient = rank_resp.match_result == "insufficient_information"
    ranked = _validate_rank_output(rank_resp, merged)
    _log_stage_metric(duck_con, run_id, group_id, orphan_id, "rank", "rank_validate", len(ranked), "direct_ranker")
    stage_trace.append({"stage": "rank", "row_count": len(ranked), "gate": "pass"})
    return OrphanWork(
        orphan_id=orphan_id,
        article_id=int(article_id) if article_id is not None else None,
        insufficient=rank_insufficient,
        insufficient_reason="rank" if rank_insufficient else None,
        stage_trace=stage_trace,
        anchors=valid_anchors,
        valid_variants=variants,
        merged_candidates=merged,
        ranked_candidates=ranked,
        provisional_reason="insufficient information from rank api" if rank_insufficient else "ranked candidates available",
    )


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
            "Pass-1 anchor extraction did not yield incident-specific, valid anchors after deterministic validation. "
            "Because no reliable anchor set was available, candidate expansion and ranking were not sufficient for identity adjudication."
        )
        conflict_analysis = "insufficient anchor detail"
        if work.insufficient_reason == "rank":
            reason_code = "insufficient_information_rank_api"
            reason_summary = (
                "Candidate retrieval completed, but the ranking API determined the article lacks enough distinguishing information "
                "to make a defensible one-candidate-vs-none decision."
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
        evidence = {
            "stage_trace": work.stage_trace,
            "narrative_evidence": {
                "orphan_anchors": work.anchors,
                "candidate_anchors": top,
                "conflict_analysis": "no candidate passed deterministic assignment gates",
            },
            "top_candidates": [_safe_text(c.get("entity_uid")) for c in top if _safe_text(c.get("entity_uid")) != ""],
            "reason_code": "no_valid_winner_after_rank_and_constraints",
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
            reason_summary=(
                "Candidate generation and ranking were completed, but no entity passed the deterministic one-to-one and conflict gates. "
                "Reviewed candidates lacked sufficient identity-level correspondence to support a defensible match."
            ),
            evidence_json=evidence,
        )

    uid = _safe_text(assigned_candidate.get("entity_uid"))
    evidence = {
        "stage_trace": work.stage_trace,
        "narrative_evidence": {
            "orphan_anchors": work.anchors,
            "candidate_anchors": [assigned_candidate],
            "conflict_analysis": "no blocking contradiction after post-rank constraints",
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
            "assignment_mode": "individual_or_bijective_greedy",
        },
    }
    return CaseDecision(
        orphan_id=work.orphan_id,
        article_id=work.article_id,
        label="matched",
        resolved_entity_id=uid,
        confidence=None,
        reason_summary=(
            f"The highest-ranked candidate {uid} is the only option that passed deterministic assignment constraints after anchor-led retrieval and ranking. "
            "Incident-specific anchor evidence and candidate-level compatibility supported same-person plausibility without a disqualifying conflict."
        ),
        evidence_json=evidence,
    )


def _process_group(
    duck_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
    client: OpenAI,
    run_id: str,
    group_rows: list[dict[str, Any]],
    allow_api_by_orphan: dict[str, bool],
    pass1_prompt_template: GPTPromptTemplate,
    rank_prompt_template: GPTPromptTemplate,
) -> list[CaseDecision]:
    group_id = _safe_text(group_rows[0].get("group_id"))
    work_items: list[OrphanWork] = []

    for r in group_rows:
        orphan_id = _safe_text(r.get("orphan_id"))
        _set_case_state(
            duck_con,
            run_id,
            orphan_id,
            case_status="in_progress",
            stage_completed="start",
        )
        try:
            work = _process_single_orphan(
                duck_con,
                sqlite_con,
                client,
                run_id,
                group_id,
                orphan_id,
                pass1_prompt_template,
                rank_prompt_template,
                allow_api_calls=allow_api_by_orphan.get(orphan_id, True),
            )
            work_items.append(work)
            _set_case_state(
                duck_con,
                run_id,
                orphan_id,
                case_status="in_progress",
                stage_completed="rank",
            )
        except Exception as ex:  # noqa: BLE001
            _log_k(f"[K][{orphan_id}] processing_error: {_safe_text(ex)}")
            _log_stage_metric(
                duck_con,
                run_id,
                group_id,
                orphan_id,
                "error",
                "exception",
                0,
                _safe_text(ex)[:500],
            )
            work_items.append(
                OrphanWork(
                    orphan_id=orphan_id,
                    article_id=int(r["article_id"]) if r.get("article_id") is not None else None,
                    insufficient=False,
                    insufficient_reason=None,
                    stage_trace=[{"stage": "error", "row_count": 0, "gate": "fail", "error": _safe_text(ex)}],
                    anchors=[],
                    valid_variants=[],
                    merged_candidates=[],
                    ranked_candidates=[],
                    provisional_reason="processing error",
                )
            )
            _set_case_state(
                duck_con,
                run_id,
                orphan_id,
                case_status="failed",
                stage_completed="error",
                decision_label="analysis_incomplete",
                error_message=_safe_text(ex)[:800],
            )

    assigned = _assign_group(work_items)
    decisions: list[CaseDecision] = []
    for w in work_items:
        if len(w.stage_trace) > 0 and _safe_text(w.stage_trace[0].get("stage")) == "error":
            evidence = {
                "stage_trace": w.stage_trace,
                "reason_code": "execution_failure",
                "execution_audit": {
                    "query_mode": "interactive_sql",
                    "fts_used": False,
                    "fts_query_list": [],
                },
            }
            decisions.append(
                CaseDecision(
                    orphan_id=w.orphan_id,
                    article_id=w.article_id,
                    label="analysis_incomplete",
                    resolved_entity_id=None,
                    confidence=None,
                    reason_summary="Required adjudication stages failed due to execution error and the case must be retried.",
                    evidence_json=evidence,
                )
            )
            continue

        decisions.append(_decision_from_work(w, assigned.get(w.orphan_id)))
    return decisions


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


def _cache_terminal_decision(
    duck_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
    decision: CaseDecision,
) -> None:
    if decision.label not in {"matched", "not_same_person", "insufficient_information"}:
        return
    dossier = _load_orphan_dossier(duck_con, sqlite_con, decision.orphan_id)
    pass1_input = _build_pass1_input(dossier)
    cache_key = _adjudication_cache_key(dossier)
    _llm_cache_put(
        duck_con,
        stage=E2E_CACHE_STAGE,
        idempotency_key=cache_key,
        model=f"{PASS1_MODEL}|{RANK_MODEL}",
        prompt_version=f"{PASS1_PROMPT_VERSION}|{RANK_PROMPT_VERSION}|e2e_v1",
        input_payload=pass1_input,
        response_payload=_decision_to_cache_payload(decision),
    )


def _cached_decisions_from_readiness(
    duck_con: duckdb.DuckDBPyConnection,
    readiness_rows: list[dict[str, Any]],
    excluded_orphan_ids: set[str],
) -> list[CaseDecision]:
    out: list[CaseDecision] = []
    for row in readiness_rows:
        orphan_id = _safe_text(row.get("orphan_id"))
        if orphan_id in excluded_orphan_ids:
            continue
        if _safe_text(row.get("readiness_status")) != "decision_ready_from_cache":
            continue
        cache_key = _safe_text(row.get("pass1_idempotency_key"))
        if cache_key == "":
            continue
        cached = _llm_cache_get(duck_con, E2E_CACHE_STAGE, cache_key)
        if cached is None:
            continue
        decision = _decision_from_cache_payload(cached, orphan_id)
        if decision.label in {"matched", "not_same_person", "insufficient_information"}:
            out.append(decision)
    return out


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


def _rebuild_overrides_table(
    duck_con: duckdb.DuckDBPyConnection,
    run_id: str,
    decisions: list[CaseDecision],
    *,
    dry_run: bool,
) -> int:
    terminal = [d for d in decisions if d.label in {"matched", "not_same_person", "insufficient_information"}]
    if dry_run:
        return len(terminal)

    prior_rows = _duck_query(
        duck_con,
        """
        SELECT orphan_id, resolution_label, resolved_entity_id, confidence, reason_summary, evidence_json
        FROM orphan_adjudication_overrides;
        """,
    )
    prior_by_orphan = {_safe_text(r.get("orphan_id")): r for r in prior_rows}

    _duck_exec(duck_con, "CREATE TEMP TABLE orphan_adj_overrides_next_tmp AS SELECT * FROM orphan_adjudication_overrides WHERE 1=0;")

    for d in terminal:
        dec_hash = _decision_hash_for_case(d)
        evidence_with_hash = {**d.evidence_json, "decision_hash": dec_hash}
        prior = prior_by_orphan.get(d.orphan_id)
        prior_hash = _prior_decision_hash(prior)
        if prior_hash != dec_hash:
            _duck_exec(duck_con, 
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
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON, 'interactive_agent_k', NOW()
                );
                """,
                (
                    run_id,
                    d.orphan_id,
                    prior.get("resolution_label") if prior else None,
                    prior.get("resolved_entity_id") if prior else None,
                    prior.get("confidence") if prior else None,
                    d.label,
                    d.resolved_entity_id,
                    d.confidence,
                    d.reason_summary,
                    _canonical_json(evidence_with_hash),
                ),
            )

        _duck_exec(duck_con, 
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
            ) VALUES (?, ?, ?, ?, ?, ?::JSON, 'interactive_agent_k', NOW(), NOW());
            """,
            (
                d.orphan_id,
                d.label,
                d.resolved_entity_id,
                d.confidence,
                d.reason_summary,
                _canonical_json(evidence_with_hash),
            ),
        )

    _duck_exec(duck_con, "BEGIN TRANSACTION;")
    try:
        _duck_exec(duck_con, "DELETE FROM orphan_adjudication_overrides;")
        _duck_exec(duck_con, 
            """
            INSERT INTO orphan_adjudication_overrides
            SELECT *
            FROM orphan_adj_overrides_next_tmp;
            """
        )
        _duck_exec(duck_con, "COMMIT;")
    except Exception:  # noqa: BLE001
        _duck_exec(duck_con, "ROLLBACK;")
        raise

    _duck_exec(duck_con, "DROP TABLE orphan_adj_overrides_next_tmp;")
    return len(terminal)


def _run_pipeline(
    duck_con: duckdb.DuckDBPyConnection,
    sqlite_con: sqlite3.Connection,
    client: OpenAI,
    params: KParams,
    readiness_rows: list[dict[str, Any]],
    pass1_prompt_template: GPTPromptTemplate,
    rank_prompt_template: GPTPromptTemplate,
) -> dict[str, int]:
    _ensure_tables(duck_con)
    _migrate_legacy_labels(duck_con)

    run_id = f"k_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _duck_exec(duck_con, 
        """
        INSERT INTO orphan_adj_run (
          run_id, started_at, requested_limit, processed_count, grouped_count,
          matched_count, not_same_person_count, insufficient_information_count,
          analysis_incomplete_count, dry_run, full_backfill, status
        ) VALUES (?, NOW(), ?, 0, 0, 0, 0, 0, 0, ?, ?, 'running');
        """,
        (
            run_id,
            params.limit,
            params.dry_run,
            params.full_backfill,
        ),
    )

    selected_needs_api = _select_needs_api_rows(readiness_rows, params)
    selected_needs_ids = {_safe_text(r.get("orphan_id")) for r in selected_needs_api}
    rows_to_process = sorted(
        selected_needs_api,
        key=lambda r: (int(r.get("queue_pos") or 0), _safe_text(r.get("orphan_id"))),
    )

    groups = _build_groups(rows_to_process, params.group_same_incident, run_id)
    _insert_queue_rows(duck_con, run_id, groups)
    allow_api_by_orphan = {_safe_text(r.get("orphan_id")): True for r in rows_to_process}
    processed = 0
    matched = 0
    not_same = 0
    insuff = 0
    incomplete = 0
    decisions_new: list[CaseDecision] = []

    try:
        for grp in groups:
            decisions = _process_group(
                duck_con,
                sqlite_con,
                client,
                run_id,
                grp,
                allow_api_by_orphan,
                pass1_prompt_template,
                rank_prompt_template,
            )
            for d in decisions:
                dec_hash = _decision_hash_for_case(d)
                decision_error_message = None
                if d.label == "analysis_incomplete":
                    stage_trace = d.evidence_json.get("stage_trace")
                    if isinstance(stage_trace, list) and len(stage_trace) > 0 and isinstance(stage_trace[0], dict):
                        decision_error_message = _safe_text(stage_trace[0].get("error"))[:800] or None
                _set_case_state(
                    duck_con,
                    run_id,
                    d.orphan_id,
                    case_status="completed" if d.label != "analysis_incomplete" else "failed",
                    stage_completed="decision",
                    decision_label=d.label,
                    resolved_entity_id=d.resolved_entity_id,
                    decision_hash=dec_hash,
                    error_message=decision_error_message,
                )
                decisions_new.append(d)
                _cache_terminal_decision(duck_con, sqlite_con, d)
                processed += 1
                if d.label == "matched":
                    matched += 1
                elif d.label == "not_same_person":
                    not_same += 1
                elif d.label == "insufficient_information":
                    insuff += 1
                else:
                    incomplete += 1

        cached_decisions = _cached_decisions_from_readiness(
            duck_con,
            readiness_rows,
            excluded_orphan_ids=selected_needs_ids,
        )

        rebuilt_terminal_rows = _rebuild_overrides_table(
            duck_con,
            run_id,
            cached_decisions + decisions_new,
            dry_run=params.dry_run,
        )

        _duck_exec(duck_con, 
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
              status = 'completed',
              error_message = NULL
            WHERE run_id = ?;
            """,
            (
                processed,
                len(groups),
                matched,
                not_same,
                insuff,
                incomplete,
                run_id,
            ),
        )

    except Exception as ex:  # noqa: BLE001
        _duck_exec(duck_con, 
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
              status = 'failed',
              error_message = ?
            WHERE run_id = ?;
            """,
            (
                processed,
                len(groups),
                matched,
                not_same,
                insuff,
                incomplete,
                _safe_text(ex)[:1000],
                run_id,
            ),
        )
        raise

    needs_api_total = sum(1 for r in readiness_rows if _safe_text(r.get("readiness_status")) == "needs_api")
    decision_ready_total = sum(
        1 for r in readiness_rows if _safe_text(r.get("readiness_status")) == "decision_ready_from_cache"
    )

    return {
        "requested_limit": params.limit,
        "needs_api_total": needs_api_total,
        "selected_needs_api": len(selected_needs_api),
        "decision_ready_from_cache": decision_ready_total,
        "processed": processed,
        "grouped": len(groups),
        "matched": matched,
        "not_same_person": not_same,
        "insufficient_information": insuff,
        "analysis_incomplete": incomplete,
        "rebuilt_terminal_rows": rebuilt_terminal_rows,
        "dry_run": int(params.dry_run),
    }


def _prepare_cache_readiness() -> Run[tuple[list[dict[str, Any]], int]]:
    return _precompute_cache_readiness_run() >> (
        lambda prep: (
            put_line(f"[K] API 1 JSON schema:\n{to_json(Pass1Response)}")
            ^ put_line(f"[K] API 2 JSON schema:\n{to_json(RankResponse)}")
            ^ put_line(_format_cache_readiness_summary(prep[0]))
            ^ pure(prep)
        )
    )


def _runtime_deps_run() -> Run[KRuntimeDeps]:
    def _with_env(env: Any) -> Run[KRuntimeDeps]:
        return pure(
            KRuntimeDeps(
                duck_con=env["connections"][DbBackend.DUCKDB],
                sqlite_con=env["connections"][DbBackend.SQLITE],
                openai_client=env["openai_client"](),
            )
        )

    return ask() >> _with_env


def _precompute_cache_readiness_run() -> Run[tuple[list[dict[str, Any]], int]]:
    return _runtime_deps_run() >> (
        lambda deps: pure(_precompute_cache_readiness(deps.duck_con, deps.sqlite_con))
    )


def _run_pipeline_run(
    params: KParams,
    readiness_rows: list[dict[str, Any]],
    pass1_prompt_template: GPTPromptTemplate,
    rank_prompt_template: GPTPromptTemplate,
) -> Run[dict[str, int]]:
    return _runtime_deps_run() >> (
        lambda deps: pure(
            _run_pipeline(
                deps.duck_con,
                deps.sqlite_con,
                deps.openai_client,
                params,
                readiness_rows,
                pass1_prompt_template,
                rank_prompt_template,
            )
        )
    )


def _execute_k(params: KParams, readiness_rows: list[dict[str, Any]]) -> Run[NextStep]:
    return ask() >> (
        lambda env: resolve_prompt_template(env, PromptKey(PASS1_PROMPT_KEY)) >> (
            lambda pass1_prompt_template: (
                put_line(
                    f"[K] Stored prompt key '{PASS1_PROMPT_KEY}' is still a placeholder id ({pass1_prompt_template.id}). "
                    "Replace it with your dashboard prompt_id and rerun."
                ) ^ pure(NextStep.CONTINUE)
                if pass1_prompt_template.id.startswith("pmpt_replace_")
                else resolve_prompt_template(env, PromptKey(RANK_PROMPT_KEY)) >> (
                    lambda rank_prompt_template: (
                        put_line(
                            f"[K] Stored prompt key '{RANK_PROMPT_KEY}' is still a placeholder id ({rank_prompt_template.id}). "
                            "Replace it with your dashboard prompt_id and rerun."
                        ) ^ pure(NextStep.CONTINUE)
                        if rank_prompt_template.id.startswith("pmpt_replace_")
                        else _run_pipeline_run(
                            params,
                            readiness_rows,
                            pass1_prompt_template,
                            rank_prompt_template,
                        ) >> (
                            lambda summary: put_line(
                                "[K] Orphan adjudication completed: "
                                f"needs_api_total={summary['needs_api_total']}, "
                                f"selected_needs_api={summary['selected_needs_api']}, "
                                f"decision_ready_from_cache={summary['decision_ready_from_cache']}, "
                                f"processed={summary['processed']}, "
                                f"groups={summary['grouped']}, "
                                f"matched={summary['matched']}, "
                                f"not_same_person={summary['not_same_person']}, "
                                f"insufficient_information={summary['insufficient_information']}, "
                                f"analysis_incomplete={summary['analysis_incomplete']}, "
                                f"rebuilt_terminal_rows={summary['rebuilt_terminal_rows']}, "
                                f"dry_run={bool(summary['dry_run'])}"
                            ) ^ pure(NextStep.CONTINUE)
                        )
                    )
                )
            )
        )
    )


def _prompt_and_execute_k(readiness_rows: list[dict[str, Any]], needs_api_total: int) -> Run[NextStep]:
    limit_prompt = f"Enter number of records requiring new API calls [{needs_api_total}]: "
    return input_with_prompt(InputPrompt(limit_prompt)) >> (
        lambda limit_raw: (
            put_line("[K] limit=0; returning to main menu.") ^ pure(NextStep.CONTINUE)
            if _parse_limit(str(limit_raw), default=needs_api_total) == 0
            else input_with_prompt(PromptKey("k_start_after")) >> (
                lambda start_after_raw: input_with_prompt(PromptKey("k_group_same_incident")) >> (
                    lambda group_raw: input_with_prompt(PromptKey("k_dry_run")) >> (
                        lambda dry_raw: input_with_prompt(PromptKey("k_full_backfill")) >> (
                            lambda full_raw: _execute_k(
                                KParams(
                                    limit=_parse_limit(str(limit_raw), default=needs_api_total),
                                    starting_after_orphan_id=str(start_after_raw).strip(),
                                    group_same_incident=_parse_bool(str(group_raw), True),
                                    dry_run=_parse_bool(str(dry_raw), False),
                                    full_backfill=_parse_bool(str(full_raw), False),
                                ),
                                readiness_rows,
                            )
                        )
                    )
                )
            )
        )
    )


def adjudicate_orphans_controller() -> Run[NextStep]:
    """
    Entry point for controller [K].
    """
    return with_namespace(
        Namespace("orphan_adj_k"),
        to_prompts(ORPHAN_ADJ_PROMPTS),
        _prepare_cache_readiness() >> (lambda prep: _prompt_and_execute_k(prep[0], prep[1])),
    )
