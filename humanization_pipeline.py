"""
SHR-level humanization pipeline using 3-step GPT processing with
incident-attribute cache keys and idempotent reruns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast, Any

from article import from_rows
from appstate import run_timer_name, run_timer_start_perf, user_name
from calculations import insert_gptresults_sql
from calculations.calc_core import elapsed_line
from gpt_filtering import render_as_failure
from menuprompts import NextStep
from pymonad import (
    Array,
    DbBackend,
    Either,
    EnvKey,
    ErrorPayload,
    GPTModel,
    GPTReasoning,
    GPTUsage,
    Just,
    Left,
    Maybe,
    Namespace,
    Nothing,
    ProcessAcc,
    PromptKey,
    Right,
    Run,
    SQL,
    SQLParams,
    StopRun,
    String,
    Tuple,
    Unit,
    ask,
    gpt_usage_reasoning_from_rows,
    input_number,
    monotonic_now,
    process_items,
    pure,
    put_line,
    resolve_prompt_template,
    rethrow,
    response_with_gpt_prompt,
    set_,
    sql_exec,
    sql_export,
    sql_query,
    to_gpt_tuple,
    to_json,
    to_prompts,
    throw,
    unit,
    view,
    with_duckdb,
    with_models,
    with_namespace,
    local,
)
from pymonad.openai import GPTResponseTuple
from pymonad.run import UserAbort
from state import (
    HumanizationClass,
    HumanizationDecisionResponse,
    HumanizationDeidentifyResponse,
    HumanizationExtractResponse,
)

RUN_TIMER_NAME = String("shr_humanization")

INPUT_LIMIT_KEY = "humanize_shr_number"
STEP1_PROMPT_KEY = "humanize_extract_incident"
STEP2_PROMPT_KEY = "humanize_deidentify"
STEP3_PROMPT_KEY = "humanize_classify"

STEP1_PROMPT_ID = "pmpt_69a4fe02d28c8195b31b05bdf8ff1cf204db837cf4b4950c"
STEP2_PROMPT_ID = "pmpt_69a5dcbfa1ac8196b2ee6928895d3b7c00d990936808e636"
STEP3_PROMPT_ID = "pmpt_69a5df5927a48196be40f44e292b7c5a03f6f6d7d2829b03"

STEP1_PROMPT_VERSION: str | None = None
STEP2_PROMPT_VERSION: str | None = None
STEP3_PROMPT_VERSION: str | None = None

STEP1_MODEL_KEY = EnvKey("humanize_extract")
STEP2_MODEL_KEY = EnvKey("humanize_deidentify")
STEP3_MODEL_KEY = EnvKey("humanize_classify")

STEP1_MODEL = GPTModel.GPT_5_MINI
STEP2_MODEL = GPTModel.GPT_5_MINI
STEP3_MODEL = GPTModel.GPT_5_MINI

PROMPTS: dict[str, str | tuple[str, str] | tuple[str,]] = {
    INPUT_LIMIT_KEY: "Enter number of SHR rows to process for humanization: ",
    STEP1_PROMPT_KEY: (STEP1_PROMPT_ID,),
    STEP2_PROMPT_KEY: (STEP2_PROMPT_ID,),
    STEP3_PROMPT_KEY: (STEP3_PROMPT_ID,),
}

MODELS = {
    STEP1_MODEL_KEY: STEP1_MODEL,
    STEP2_MODEL_KEY: STEP2_MODEL,
    STEP3_MODEL_KEY: STEP3_MODEL,
}


@dataclass(frozen=True)
class CandidateDecision:
    """
    Per-candidate article decision payload returned by cache lookup or GPT steps.
    """
    article_id: int
    decision_binary: int
    decision_label: str
    used_cache: bool
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    est_cost: float


@dataclass(frozen=True)
class ShrProcessResult:
    """
    Aggregated processing outcome and usage metrics for one SHR row.
    """
    shr_uid: str
    run_status: str
    humanizing_binary: int | None
    cache_hits: int
    cache_misses: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    est_cost: float


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    value = row[key]
    return default if value is None else value


def _prompt_token(value: Any) -> str:
    """
    Normalize prompt variable values so null/unknown are explicit tokens.
    """
    if value is None:
        return "NULL"
    s = str(value).strip()
    if s == "":
        return "NULL"
    if s.lower() == "unknown":
        return "unknown"
    return s


def _with_sqlite(subprog: Run[Any]) -> Run[Any]:
    """
    Run a subprogram using the SQLite backend inside a DuckDB controller flow.
    """
    return local(
        lambda env: {**env, "current_backend": DbBackend.SQLITE},
        subprog,
    )


def _display_article_for_analysis(article_id: int) -> Run[Unit]:
    """
    Display the candidate article in the same style used by GPT extraction flow.
    """
    return _with_sqlite(
        sql_query(
            SQL(
                """
                SELECT
                  RecordId,
                  Title,
                  Publication,
                  PubDate,
                  FullText,
                  Status AS status,
                  gptClass AS GPTClass,
                  AssignStatus AS assignstatus,
                  gptVictimJson
                FROM articles
                WHERE RecordId = ?
                LIMIT 1;
                """
            ),
            SQLParams((article_id,)),
        )
    ) >> (
        lambda rows: put_line(f"Article {article_id} not found in articles.\n")
        if len(rows) == 0
        else put_line(f"Analyzing article:\n{from_rows(rows)[0]}")
    ) ^ pure(unit)


def _display_latest_gpt_for_article(article_id: int) -> Run[Unit]:
    """
    Display latest gptResults usage/reasoning summary for the article if present.
    """
    def _after_query(maybe_usage: Maybe[Tuple[GPTUsage, GPTReasoning]]) -> Run[Unit]:
        match maybe_usage:
            case Just(tup):
                return (
                    put_line(str(tup.fst))
                    ^ put_line(f"GPT reasoning summary:\n{tup.snd}")
                    ^ pure(unit)
                )
            case _:
                return put_line("No GPT responses captured for this article.\n") ^ pure(unit)

    return (
        gpt_usage_reasoning_from_rows
        & _with_sqlite(
            sql_query(
                SQL(
                    """
                    SELECT *
                    FROM gptResults
                    WHERE RecordId = ?
                    ORDER BY TimeStamp DESC, ResultId DESC
                    LIMIT 1;
                    """
                ),
                SQLParams((article_id,)),
            )
        )
    ) >> _after_query


def _print_step_result(step_name: str, resp_t: GPTResponseTuple) -> Run[Unit]:
    """
    Print full per-step usage, reasoning summary, and structured output JSON.
    """
    return (
        put_line(f"{step_name} output:")
        ^ put_line(str(resp_t.parsed.usage))
        ^ put_line(f"GPT reasoning summary:\n{resp_t.parsed.reasoning}")
        ^ put_line(resp_t.parsed.output.model_dump_json(indent=2))
        ^ pure(unit)
    )


def save_humanization_step_gpt_result(
    article_id: int,
    step_prompt_key: str,
    variables: dict[str, str | None],
    resp_t: GPTResponseTuple,
    format_type: str,
) -> Run[Unit]:
    """
    Persist one successful humanization GPT step call to gptResults.
    """
    timestamp = String(datetime.now().isoformat())
    model = String(
        resp_t.parsed.usage.model_used.value if resp_t.parsed.usage.model_used else ""
    )
    variables_json = String(json.dumps(variables, default=str, indent=2))
    output_json = String(resp_t.parsed.output.model_dump_json(indent=2))
    usage = resp_t.parsed.usage

    return view(user_name) >> (lambda user:
        ask() >> (lambda env:
            resolve_prompt_template(env, PromptKey(step_prompt_key)) >> (lambda prompt_template:
                _with_sqlite(
                    sql_exec(
                        SQL(insert_gptresults_sql()),
                        SQLParams(
                            (
                                article_id,
                                String(user),
                                timestamp,
                                String(step_prompt_key),
                                String(prompt_template.id),
                                String(prompt_template.version) if prompt_template.version is not None else None,
                                variables_json,
                                model,
                                String(format_type),
                                output_json,
                                String(str(resp_t.parsed.reasoning)),
                                usage.input_tokens,
                                usage.cached_tokens,
                                usage.output_tokens,
                                usage.reasoning_tokens,
                                usage.cost(),
                            )
                        ),
                    )
                )
            )
        )
    )


def _step1_variables(candidate_row) -> dict[str, str | None]:
    return {
        "article_text": _prompt_token(_row_value(candidate_row, "article_text", None)),
        "incident_year": _prompt_token(_row_value(candidate_row, "year", None)),
        "incident_month": _prompt_token(_row_value(candidate_row, "month", None)),
        "incident_day": _prompt_token(_row_value(candidate_row, "day", None)),
        "incident_summary": _prompt_token(_row_value(candidate_row, "incident_summary_norm", None)),
        "victim_name": _prompt_token(_row_value(candidate_row, "victim_name_norm2", None)),
        "victim_age": _prompt_token(_row_value(candidate_row, "victim_age", None)),
    }


def _step2_variables(excerpt: str) -> dict[str, str | None]:
    return {
        "incident_excerpt": _prompt_token(excerpt),
    }


def _step3_variables(deidentified_excerpt: str) -> dict[str, str | None]:
    return {
        "deidentified_excerpt": _prompt_token(deidentified_excerpt),
    }


def start_run_timer(run_name: String) -> Run[Unit]:
    """
    Start or replace the named run timer in AppState.
    """
    return (set_(run_timer_name, run_name) ^ monotonic_now()) >> (
        lambda now: set_(run_timer_start_perf, now) ^ pure(unit)
    )


def read_elapsed_display(expected_run_name: String) -> Run[Maybe[String]]:
    """
    Read formatted elapsed time for the expected active timer, if available.
    """
    def _just_elapsed(now: float, start: float) -> Run[Maybe[String]]:
        return pure(Just(String(elapsed_line(now - start))))

    def _from_start(timer_name: str, start: float | None) -> Run[Maybe[String]]:
        if str(timer_name) != str(expected_run_name) or start is None:
            return pure(Nothing)
        return monotonic_now() >> (lambda now: _just_elapsed(now, start))

    return view(run_timer_name) >> (
        lambda timer_name: view(run_timer_start_perf)
        >> (lambda start: _from_start(timer_name, start))
    )


def _ensure_tables() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """
                CREATE TABLE IF NOT EXISTS humanization_stage_cache (
                  incident_cache_key VARCHAR PRIMARY KEY,
                  incident_date_norm VARCHAR,
                  incident_summary_norm VARCHAR,
                  victim_name_norm VARCHAR,
                  victim_age_norm VARCHAR,

                  step1_prompt_id VARCHAR,
                  step1_prompt_version VARCHAR,
                  step1_model VARCHAR,
                  step1_status VARCHAR,
                  step1_output_text VARCHAR,
                  step1_output_json VARCHAR,
                  step1_reasoning VARCHAR,
                  step1_input_tokens INTEGER,
                  step1_cached_tokens INTEGER,
                  step1_output_tokens INTEGER,
                  step1_reasoning_tokens INTEGER,
                  step1_cost DOUBLE,

                  step2_prompt_id VARCHAR,
                  step2_prompt_version VARCHAR,
                  step2_model VARCHAR,
                  step2_status VARCHAR,
                  step2_output_text VARCHAR,
                  step2_output_json VARCHAR,
                  step2_reasoning VARCHAR,
                  step2_input_tokens INTEGER,
                  step2_cached_tokens INTEGER,
                  step2_output_tokens INTEGER,
                  step2_reasoning_tokens INTEGER,
                  step2_cost DOUBLE,

                  step3_prompt_id VARCHAR,
                  step3_prompt_version VARCHAR,
                  step3_model VARCHAR,
                  step3_status VARCHAR,
                  step3_decision_label VARCHAR,
                  step3_output_json VARCHAR,
                  step3_reasoning VARCHAR,
                  step3_input_tokens INTEGER,
                  step3_cached_tokens INTEGER,
                  step3_output_tokens INTEGER,
                  step3_reasoning_tokens INTEGER,
                  step3_cost DOUBLE,

                  updated_at TIMESTAMP
                );
                """
            )
        )
        ^ sql_exec(
            SQL(
                """
                CREATE TABLE IF NOT EXISTS humanization_article_result (
                  run_id VARCHAR,
                  shr_uid VARCHAR,
                  entity_uid VARCHAR,
                  article_id BIGINT,
                  incident_idx INTEGER,
                  victim_idx INTEGER,
                  incident_cache_key VARCHAR,
                  specificity_score DOUBLE,
                  priority_rank INTEGER,
                  used_cache BOOLEAN,
                  decision_label VARCHAR,
                  decision_binary INTEGER,
                  status VARCHAR,
                  error_text VARCHAR,
                  input_tokens INTEGER,
                  cached_tokens INTEGER,
                  output_tokens INTEGER,
                  reasoning_tokens INTEGER,
                  est_cost DOUBLE,
                  processed_at TIMESTAMP
                );
                """
            )
        )
        ^ sql_exec(
            SQL(
                """
                CREATE TABLE IF NOT EXISTS humanization_shr_result (
                  shr_uid VARCHAR PRIMARY KEY,
                  run_id VARCHAR,
                  shr_year INTEGER,
                  entity_uid VARCHAR,
                  candidate_set_hash VARCHAR,
                  humanizing_binary INTEGER,
                  run_status VARCHAR,
                  trigger_article_id BIGINT,
                  articles_total INTEGER,
                  articles_attempted INTEGER,
                  articles_successful INTEGER,
                  articles_failed INTEGER,
                  processed_at TIMESTAMP
                );
                """
            )
        )
        ^ sql_exec(
            SQL(
                """
                CREATE TABLE IF NOT EXISTS humanization_run_history (
                  run_id VARCHAR PRIMARY KEY,
                  requested_count INTEGER,
                  processed_count INTEGER,
                  humanizing_count INTEGER,
                  not_humanizing_count INTEGER,
                  pending_retry_count INTEGER,
                  cache_hits INTEGER,
                  cache_misses INTEGER,
                  input_tokens BIGINT,
                  cached_tokens BIGINT,
                  output_tokens BIGINT,
                  reasoning_tokens BIGINT,
                  est_cost DOUBLE,
                  elapsed_display VARCHAR,
                  export_path VARCHAR,
                  created_at TIMESTAMP
                );
                """
            )
        )
    )


def _build_current_candidates() -> Run[Unit]:
    return sql_exec(
        SQL(
            """--sql
            CREATE OR REPLACE TABLE humanization_candidates_current AS
            WITH incident_counts AS (
              SELECT article_id, COUNT(DISTINCT incident_idx) AS incident_count
              FROM incidents_cached
              GROUP BY article_id
            ),
            base AS (
              SELECT
                CAST(sm.unique_id_r AS VARCHAR) AS shr_uid,
                CAST(sm.unique_id_l AS VARCHAR) AS entity_uid,
                CAST(sc.year AS INTEGER) AS shr_year,
                vm.article_id,
                vm.incident_idx,
                vm.victim_idx,
                vm.victim_age,
                vm.victim_name_raw,
                vm.victim_name_norm,
                vm.date_precision,
                vm.year,
                vm.month,
                vm.day,
                vm.publish_date,
                ic.article_title AS article_title,
                ic.article_text AS article_text,
                ic.publish_date AS article_date,
                COALESCE(ic.summary_norm, '') AS incident_summary_raw,
                COALESCE(cnt.incident_count, 0) AS incident_count
              FROM shr_max_weight_matches sm
              JOIN victim_entity_members vm
                ON CAST(vm.victim_entity_id AS VARCHAR) = CAST(sm.unique_id_l AS VARCHAR)
              LEFT JOIN incidents_cached ic
                ON vm.article_id = ic.article_id
               AND vm.incident_idx = ic.incident_idx
              LEFT JOIN incident_counts cnt
                ON vm.article_id = cnt.article_id
              LEFT JOIN shr_cached sc
                ON CAST(sc.unique_id AS VARCHAR) = CAST(sm.unique_id_r AS VARCHAR)
            ),
            norm AS (
              SELECT
                *,
                CASE
                  WHEN date_precision = 'day' AND year IS NOT NULL AND month IS NOT NULL AND day IS NOT NULL
                    THEN 'day:' || printf('%04d-%02d-%02d', year, month, day)
                  WHEN date_precision = 'month' AND year IS NOT NULL AND month IS NOT NULL
                    THEN 'month:' || printf('%04d-%02d', year, month)
                  WHEN year IS NOT NULL
                    THEN 'year:' || printf('%04d', year)
                  ELSE 'unknown'
                END AS incident_date_norm,
                trim(regexp_replace(lower(regexp_replace(COALESCE(incident_summary_raw, ''), '[[:punct:]]+', ' ', 'g')), '[[:space:]]+', ' ', 'g')) AS incident_summary_norm,
                trim(regexp_replace(lower(COALESCE(NULLIF(victim_name_norm, ''), COALESCE(victim_name_raw, ''))), '[[:space:]]+', ' ', 'g')) AS victim_name_norm2,
                COALESCE(CAST(victim_age AS VARCHAR), 'NULL') AS victim_age_norm,
                (
                  CASE WHEN incident_count = 1 THEN 40 ELSE 0 END
                  + CASE WHEN COALESCE(NULLIF(victim_name_norm, ''), NULLIF(victim_name_raw, '')) IS NOT NULL THEN 20 ELSE 0 END
                  + CASE WHEN victim_age IS NOT NULL THEN 10 ELSE 0 END
                  + CASE WHEN length(COALESCE(incident_summary_raw, '')) >= 120 THEN 20 ELSE 0 END
                  - (CASE WHEN incident_count > 1 THEN (incident_count - 1) * 5 ELSE 0 END)
                )::DOUBLE AS specificity_score
              FROM base
            )
            SELECT
              *,
              md5(
                coalesce(CAST(article_id AS VARCHAR), '') || '|'
                || coalesce(incident_date_norm, '') || '|'
                || coalesce(incident_summary_norm, '') || '|'
                || coalesce(victim_name_norm2, '') || '|'
                || coalesce(victim_age_norm, 'NULL')
              ) AS incident_cache_key
            FROM norm;
            """
        )
    )


def _build_reprocess_queue() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE humanization_shr_signatures_current AS
                SELECT
                  shr_uid,
                  MAX(shr_year) AS shr_year,
                  MAX(entity_uid) AS entity_uid,
                  md5(COALESCE(string_agg(DISTINCT incident_cache_key, '|' ORDER BY incident_cache_key), '')) AS candidate_set_hash,
                  COUNT(*) AS candidate_count
                FROM humanization_candidates_current
                GROUP BY shr_uid;
                """
            )
        )
        ^ sql_exec(
            SQL(
                f"""--sql
                CREATE OR REPLACE TABLE humanization_shr_cache_status AS
                WITH candidate_ready AS (
                  SELECT
                    c.shr_uid,
                    c.incident_cache_key,
                    h.step3_decision_label,
                    CASE
                      WHEN h.incident_cache_key IS NOT NULL
                       AND h.step1_status = 'success'
                       AND h.step2_status = 'success'
                       AND h.step3_status = 'success'
                       AND h.step3_decision_label IN ('{HumanizationClass.HUMANIZING.value}', '{HumanizationClass.NOT_HUMANIZING.value}')
                       AND h.step1_model = '{STEP1_MODEL.value}'
                       AND h.step2_model = '{STEP2_MODEL.value}'
                       AND h.step3_model = '{STEP3_MODEL.value}'
                      THEN 1 ELSE 0
                    END AS is_ready
                  FROM humanization_candidates_current c
                  LEFT JOIN humanization_stage_cache h
                    ON h.incident_cache_key = c.incident_cache_key
                )
                SELECT
                  shr_uid,
                  COUNT(*) AS candidate_count,
                  COUNT(*) FILTER (WHERE is_ready = 1) AS ready_count,
                  COUNT(*) FILTER (
                    WHERE is_ready = 1
                      AND step3_decision_label = '{HumanizationClass.HUMANIZING.value}'
                  ) AS ready_humanizing_count,
                  COUNT(*) FILTER (
                    WHERE is_ready = 1
                      AND step3_decision_label = '{HumanizationClass.NOT_HUMANIZING.value}'
                  ) AS ready_not_humanizing_count,
                  COUNT(*) FILTER (WHERE is_ready = 0) AS missing_cache_count
                FROM candidate_ready
                GROUP BY shr_uid;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE humanization_shr_reprocess_queue AS
                WITH status AS (
                  SELECT
                    s.shr_uid,
                    s.shr_year,
                    s.entity_uid,
                    s.candidate_set_hash,
                    s.candidate_count,
                    COALESCE(cs.ready_count, 0) AS ready_count,
                    COALESCE(cs.ready_humanizing_count, 0) AS ready_humanizing_count,
                    COALESCE(cs.ready_not_humanizing_count, 0) AS ready_not_humanizing_count,
                    COALESCE(cs.missing_cache_count, 0) AS missing_cache_count,
                    r.run_status AS prev_run_status,
                    r.candidate_set_hash AS prev_candidate_set_hash,
                    CASE
                      WHEN COALESCE(cs.ready_humanizing_count, 0) > 0 THEN 1
                      ELSE 0
                    END AS has_humanizing_cache_hit,
                    CASE
                      WHEN s.candidate_count > 0
                       AND COALESCE(cs.ready_not_humanizing_count, 0) = s.candidate_count
                      THEN 1 ELSE 0
                    END AS all_candidates_cached_not_humanizing
                  FROM humanization_shr_signatures_current s
                  LEFT JOIN humanization_shr_result r
                    ON r.shr_uid = s.shr_uid
                  LEFT JOIN humanization_shr_cache_status cs
                    ON cs.shr_uid = s.shr_uid
                )
                SELECT
                  shr_uid,
                  shr_year,
                  entity_uid,
                  candidate_set_hash,
                  candidate_count,
                  ready_count,
                  ready_humanizing_count,
                  ready_not_humanizing_count,
                  missing_cache_count,
                  prev_run_status,
                  prev_candidate_set_hash,
                  has_humanizing_cache_hit,
                  all_candidates_cached_not_humanizing,
                  CASE
                    WHEN has_humanizing_cache_hit = 1 THEN 'cache_short_circuit_humanizing'
                    WHEN all_candidates_cached_not_humanizing = 1 THEN 'cache_complete_not_humanizing'
                    WHEN prev_run_status = 'pending_retry' THEN 'pending_retry'
                    WHEN COALESCE(prev_candidate_set_hash, '') <> COALESCE(candidate_set_hash, '') THEN 'candidate_set_changed'
                    WHEN missing_cache_count > 0 THEN 'cache_incomplete'
                    ELSE 'needs_analysis'
                  END AS reason_code,
                  ROW_NUMBER() OVER (
                    ORDER BY
                      CASE
                        WHEN prev_run_status = 'pending_retry' THEN 0
                        WHEN COALESCE(prev_candidate_set_hash, '') <> COALESCE(candidate_set_hash, '') THEN 1
                        WHEN missing_cache_count > 0 THEN 1
                        ELSE 9
                      END,
                      shr_year,
                      shr_uid
                  ) AS priority_rank
                FROM status
                WHERE has_humanizing_cache_hit = 0
                  AND all_candidates_cached_not_humanizing = 0;
                """
            )
        )
    )


def _display_queue_counts() -> Run[Unit]:
    return sql_query(
        SQL(
            """
            SELECT CAST(shr_year AS VARCHAR) AS ShrYear, COUNT(*) AS ReadyCount
            FROM humanization_shr_reprocess_queue
            GROUP BY ShrYear
            ORDER BY ShrYear;
            """
        )
    ) >> (
        lambda rows: (
            put_line("No SHR rows currently require (re)processing.\n")
            if len(rows) == 0
            else put_line(
                "SHR rows requiring (re)processing by year:\n"
                + "\n".join(f"{r['ShrYear']}: {r['ReadyCount']}" for r in rows)
                + "\n"
            )
        )
        ^ pure(unit)
    )


def _input_number_to_process() -> Run[int]:
    def check_if_zero(num: int) -> Run[int]:
        if num < 0:
            return throw(ErrorPayload("", "USER_ABORT"))
        return pure(num)

    return input_number(PromptKey(INPUT_LIMIT_KEY)) >> check_if_zero


def _retrieve_shr_queue(limit_n: int) -> Run[Array]:
    return sql_query(
        SQL(
            """
            SELECT *
            FROM humanization_shr_reprocess_queue
            ORDER BY priority_rank
            LIMIT ?;
            """
        ),
        SQLParams((limit_n,)),
    )


def _lookup_cache(incident_cache_key: str) -> Run[Array]:
    return sql_query(
        SQL(
            """
            SELECT *
            FROM humanization_stage_cache
            WHERE incident_cache_key = ?
              AND step1_status = 'success'
              AND step2_status = 'success'
              AND step3_status = 'success'
              AND step3_decision_label IN ('humanizing', 'not_humanizing')
              AND step1_model = ?
              AND step2_model = ?
              AND step3_model = ?
            LIMIT 1;
            """
        ),
        SQLParams(
            (
                String(incident_cache_key),
                String(STEP1_MODEL.value),
                String(STEP2_MODEL.value),
                String(STEP3_MODEL.value),
            )
        ),
    )


def _run_step1(variables: dict[str, str | None]) -> Run[GPTResponseTuple]:
    return (
        to_gpt_tuple
        & response_with_gpt_prompt(
            PromptKey(STEP1_PROMPT_KEY),
            variables,
            HumanizationExtractResponse,
            STEP1_MODEL_KEY,
            effort="low",
            stream=False,
        )
    ) >> rethrow >> (lambda resp: pure(cast(GPTResponseTuple, resp)))


def _run_step2(variables: dict[str, str | None]) -> Run[GPTResponseTuple]:
    return (
        to_gpt_tuple
        & response_with_gpt_prompt(
            PromptKey(STEP2_PROMPT_KEY),
            variables,
            HumanizationDeidentifyResponse,
            STEP2_MODEL_KEY,
            effort="low",
            stream=False,
        )
    ) >> rethrow >> (lambda resp: pure(cast(GPTResponseTuple, resp)))


def _run_step3(variables: dict[str, str | None]) -> Run[GPTResponseTuple]:
    return (
        to_gpt_tuple
        & response_with_gpt_prompt(
            PromptKey(STEP3_PROMPT_KEY),
            variables,
            HumanizationDecisionResponse,
            STEP3_MODEL_KEY,
            effort="medium",
            stream=False,
        )
    ) >> rethrow >> (lambda resp: pure(cast(GPTResponseTuple, resp)))


def _upsert_stage_cache(candidate_row, step1: GPTResponseTuple, step2: GPTResponseTuple, step3: GPTResponseTuple) -> Run[Unit]:
    out1 = cast(HumanizationExtractResponse, step1.parsed.output)
    out2 = cast(HumanizationDeidentifyResponse, step2.parsed.output)
    out3 = cast(HumanizationDecisionResponse, step3.parsed.output)
    u1 = step1.parsed.usage
    u2 = step2.parsed.usage
    u3 = step3.parsed.usage
    return sql_exec(
        SQL(
            """
            INSERT INTO humanization_stage_cache (
              incident_cache_key,
              incident_date_norm,
              incident_summary_norm,
              victim_name_norm,
              victim_age_norm,

              step1_prompt_id, step1_prompt_version, step1_model, step1_status,
              step1_output_text, step1_output_json, step1_reasoning,
              step1_input_tokens, step1_cached_tokens, step1_output_tokens, step1_reasoning_tokens, step1_cost,

              step2_prompt_id, step2_prompt_version, step2_model, step2_status,
              step2_output_text, step2_output_json, step2_reasoning,
              step2_input_tokens, step2_cached_tokens, step2_output_tokens, step2_reasoning_tokens, step2_cost,

              step3_prompt_id, step3_prompt_version, step3_model, step3_status,
              step3_decision_label, step3_output_json, step3_reasoning,
              step3_input_tokens, step3_cached_tokens, step3_output_tokens, step3_reasoning_tokens, step3_cost,

              updated_at
            ) VALUES (
              ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
              NOW()
            )
            ON CONFLICT(incident_cache_key) DO UPDATE SET
              incident_date_norm = EXCLUDED.incident_date_norm,
              incident_summary_norm = EXCLUDED.incident_summary_norm,
              victim_name_norm = EXCLUDED.victim_name_norm,
              victim_age_norm = EXCLUDED.victim_age_norm,

              step1_prompt_id = EXCLUDED.step1_prompt_id,
              step1_prompt_version = EXCLUDED.step1_prompt_version,
              step1_model = EXCLUDED.step1_model,
              step1_status = EXCLUDED.step1_status,
              step1_output_text = EXCLUDED.step1_output_text,
              step1_output_json = EXCLUDED.step1_output_json,
              step1_reasoning = EXCLUDED.step1_reasoning,
              step1_input_tokens = EXCLUDED.step1_input_tokens,
              step1_cached_tokens = EXCLUDED.step1_cached_tokens,
              step1_output_tokens = EXCLUDED.step1_output_tokens,
              step1_reasoning_tokens = EXCLUDED.step1_reasoning_tokens,
              step1_cost = EXCLUDED.step1_cost,

              step2_prompt_id = EXCLUDED.step2_prompt_id,
              step2_prompt_version = EXCLUDED.step2_prompt_version,
              step2_model = EXCLUDED.step2_model,
              step2_status = EXCLUDED.step2_status,
              step2_output_text = EXCLUDED.step2_output_text,
              step2_output_json = EXCLUDED.step2_output_json,
              step2_reasoning = EXCLUDED.step2_reasoning,
              step2_input_tokens = EXCLUDED.step2_input_tokens,
              step2_cached_tokens = EXCLUDED.step2_cached_tokens,
              step2_output_tokens = EXCLUDED.step2_output_tokens,
              step2_reasoning_tokens = EXCLUDED.step2_reasoning_tokens,
              step2_cost = EXCLUDED.step2_cost,

              step3_prompt_id = EXCLUDED.step3_prompt_id,
              step3_prompt_version = EXCLUDED.step3_prompt_version,
              step3_model = EXCLUDED.step3_model,
              step3_status = EXCLUDED.step3_status,
              step3_decision_label = EXCLUDED.step3_decision_label,
              step3_output_json = EXCLUDED.step3_output_json,
              step3_reasoning = EXCLUDED.step3_reasoning,
              step3_input_tokens = EXCLUDED.step3_input_tokens,
              step3_cached_tokens = EXCLUDED.step3_cached_tokens,
              step3_output_tokens = EXCLUDED.step3_output_tokens,
              step3_reasoning_tokens = EXCLUDED.step3_reasoning_tokens,
              step3_cost = EXCLUDED.step3_cost,
              updated_at = NOW();
            """
        ),
        SQLParams(
            (
                String(str(_row_value(candidate_row, "incident_cache_key", ""))),
                String(str(_row_value(candidate_row, "incident_date_norm", ""))),
                String(str(_row_value(candidate_row, "incident_summary_norm", ""))),
                String(str(_row_value(candidate_row, "victim_name_norm2", ""))),
                String(str(_row_value(candidate_row, "victim_age_norm", "NULL"))),

                String(STEP1_PROMPT_ID),
                String(STEP1_PROMPT_VERSION) if STEP1_PROMPT_VERSION is not None else None,
                String(STEP1_MODEL.value),
                String("success"),
                String(out1.incident_excerpt),
                String(out1.model_dump_json(indent=2)),
                String(str(step1.parsed.reasoning)),
                u1.input_tokens,
                u1.cached_tokens,
                u1.output_tokens,
                u1.reasoning_tokens,
                u1.cost(),

                String(STEP2_PROMPT_ID),
                String(STEP2_PROMPT_VERSION) if STEP2_PROMPT_VERSION is not None else None,
                String(STEP2_MODEL.value),
                String("success"),
                String(out2.deidentified_excerpt),
                String(out2.model_dump_json(indent=2)),
                String(str(step2.parsed.reasoning)),
                u2.input_tokens,
                u2.cached_tokens,
                u2.output_tokens,
                u2.reasoning_tokens,
                u2.cost(),

                String(STEP3_PROMPT_ID),
                String(STEP3_PROMPT_VERSION) if STEP3_PROMPT_VERSION is not None else None,
                String(STEP3_MODEL.value),
                String("success"),
                String(out3.humanization_classification.value),
                String(out3.model_dump_json(indent=2)),
                String(str(step3.parsed.reasoning)),
                u3.input_tokens,
                u3.cached_tokens,
                u3.output_tokens,
                u3.reasoning_tokens,
                u3.cost(),
            )
        ),
    )


def _write_article_result(
    run_id: str,
    shr_uid: str,
    entity_uid: str,
    candidate_row,
    used_cache: bool,
    decision_label: str | None,
    decision_binary: int | None,
    status: str,
    error_text: str | None,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    est_cost: float,
) -> Run[Unit]:
    return sql_exec(
        SQL(
            """
            INSERT INTO humanization_article_result (
              run_id, shr_uid, entity_uid, article_id, incident_idx, victim_idx,
              incident_cache_key, specificity_score, priority_rank,
              used_cache, decision_label, decision_binary, status, error_text,
              input_tokens, cached_tokens, output_tokens, reasoning_tokens, est_cost,
              processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW());
            """
        ),
        SQLParams(
            (
                String(run_id),
                String(shr_uid),
                String(entity_uid),
                int(_row_value(candidate_row, "article_id", 0)),
                int(_row_value(candidate_row, "incident_idx", 0)),
                int(_row_value(candidate_row, "victim_idx", 0)),
                String(str(_row_value(candidate_row, "incident_cache_key", ""))),
                float(_row_value(candidate_row, "specificity_score", 0.0)),
                int(_row_value(candidate_row, "priority_rank", 0)),
                1 if used_cache else 0,
                String(decision_label) if decision_label is not None else None,
                decision_binary,
                String(status),
                String(error_text) if error_text is not None else None,
                input_tokens,
                cached_tokens,
                output_tokens,
                reasoning_tokens,
                est_cost,
            )
        ),
    )


def _resolve_candidate(candidate_row) -> Run[CandidateDecision]:
    cache_key = str(_row_value(candidate_row, "incident_cache_key", ""))

    def _from_cache(rows: Array) -> Run[CandidateDecision]:
        if len(rows) > 0:
            row = rows[0]
            label = str(_row_value(row, "step3_decision_label", HumanizationClass.NOT_HUMANIZING.value))
            binary = 1 if label == HumanizationClass.HUMANIZING.value else 0
            return pure(
                CandidateDecision(
                    article_id=int(_row_value(candidate_row, "article_id", 0)),
                    decision_binary=binary,
                    decision_label=label,
                    used_cache=True,
                    input_tokens=0,
                    cached_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    est_cost=0.0,
                )
            )

        article_id = int(_row_value(candidate_row, "article_id", 0))
        step1_variables = _step1_variables(candidate_row)

        def _after_step3(
            s1: GPTResponseTuple,
            s2: GPTResponseTuple,
            s3: GPTResponseTuple,
            step3_variables: dict[str, str | None],
        ) -> Run[CandidateDecision]:
            step3_output = cast(HumanizationDecisionResponse, s3.parsed.output)
            return (
                _print_step_result("Step 3", s3)
                ^ save_humanization_step_gpt_result(
                    article_id=article_id,
                    step_prompt_key=STEP3_PROMPT_KEY,
                    variables=step3_variables,
                    resp_t=s3,
                    format_type=HumanizationDecisionResponse.__name__,
                )
                ^ _upsert_stage_cache(candidate_row, s1, s2, s3)
                ^ pure(
                    CandidateDecision(
                        article_id=article_id,
                        decision_binary=(
                            1
                            if step3_output.humanization_classification == HumanizationClass.HUMANIZING
                            else 0
                        ),
                        decision_label=step3_output.humanization_classification.value,
                        used_cache=False,
                        input_tokens=s1.parsed.usage.input_tokens + s2.parsed.usage.input_tokens + s3.parsed.usage.input_tokens,
                        cached_tokens=s1.parsed.usage.cached_tokens + s2.parsed.usage.cached_tokens + s3.parsed.usage.cached_tokens,
                        output_tokens=s1.parsed.usage.output_tokens + s2.parsed.usage.output_tokens + s3.parsed.usage.output_tokens,
                        reasoning_tokens=s1.parsed.usage.reasoning_tokens + s2.parsed.usage.reasoning_tokens + s3.parsed.usage.reasoning_tokens,
                        est_cost=s1.parsed.usage.cost() + s2.parsed.usage.cost() + s3.parsed.usage.cost(),
                    )
                )
            )

        def _after_step2(
            s1: GPTResponseTuple,
            s2: GPTResponseTuple,
            step2_variables: dict[str, str | None],
        ) -> Run[CandidateDecision]:
            step2_output = cast(HumanizationDeidentifyResponse, s2.parsed.output)
            step3_variables = _step3_variables(step2_output.deidentified_excerpt)
            return (
                _print_step_result("Step 2", s2)
                ^ save_humanization_step_gpt_result(
                    article_id=article_id,
                    step_prompt_key=STEP2_PROMPT_KEY,
                    variables=step2_variables,
                    resp_t=s2,
                    format_type=HumanizationDeidentifyResponse.__name__,
                )
                ^ _run_step3(step3_variables)
                >> (lambda s3: _after_step3(s1, s2, s3, step3_variables))
            )

        def _after_step1(s1: GPTResponseTuple) -> Run[CandidateDecision]:
            step1_output = cast(HumanizationExtractResponse, s1.parsed.output)
            step2_variables = _step2_variables(step1_output.incident_excerpt)
            return (
                _print_step_result("Step 1", s1)
                ^ save_humanization_step_gpt_result(
                    article_id=article_id,
                    step_prompt_key=STEP1_PROMPT_KEY,
                    variables=step1_variables,
                    resp_t=s1,
                    format_type=HumanizationExtractResponse.__name__,
                )
                ^ _run_step2(step2_variables)
                >> (lambda s2: _after_step2(s1, s2, step2_variables))
            )

        return (
            _display_article_for_analysis(article_id)
            ^ _display_latest_gpt_for_article(article_id)
            ^ _run_step1(step1_variables)
            >> _after_step1
        )

    return _lookup_cache(cache_key) >> _from_cache


def _retrieve_candidates_for_shr(shr_uid: str) -> Run[Array]:
    return sql_query(
        SQL(
            """
            SELECT
              c.*,
              ROW_NUMBER() OVER (
                ORDER BY c.specificity_score DESC, c.article_id ASC, c.incident_idx ASC, c.victim_idx ASC
              ) AS priority_rank
            FROM humanization_candidates_current c
            WHERE c.shr_uid = ?
            ORDER BY c.specificity_score DESC, c.article_id ASC, c.incident_idx ASC, c.victim_idx ASC;
            """
        ),
        SQLParams((String(shr_uid),)),
    )


def _persist_shr_result(
    run_id: str,
    shr_row,
    candidate_set_hash: str,
    humanizing_binary: int | None,
    run_status: str,
    trigger_article_id: int | None,
    articles_total: int,
    articles_attempted: int,
    articles_successful: int,
    articles_failed: int,
) -> Run[Unit]:
    return sql_exec(
        SQL(
            """
            INSERT INTO humanization_shr_result (
              shr_uid, run_id, shr_year, entity_uid, candidate_set_hash,
              humanizing_binary, run_status, trigger_article_id,
              articles_total, articles_attempted, articles_successful, articles_failed,
              processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
            ON CONFLICT(shr_uid) DO UPDATE SET
              run_id = EXCLUDED.run_id,
              shr_year = EXCLUDED.shr_year,
              entity_uid = EXCLUDED.entity_uid,
              candidate_set_hash = EXCLUDED.candidate_set_hash,
              humanizing_binary = EXCLUDED.humanizing_binary,
              run_status = EXCLUDED.run_status,
              trigger_article_id = EXCLUDED.trigger_article_id,
              articles_total = EXCLUDED.articles_total,
              articles_attempted = EXCLUDED.articles_attempted,
              articles_successful = EXCLUDED.articles_successful,
              articles_failed = EXCLUDED.articles_failed,
              processed_at = NOW();
            """
        ),
        SQLParams(
            (
                String(str(_row_value(shr_row, "shr_uid", ""))),
                String(run_id),
                int(_row_value(shr_row, "shr_year", 0)) if _row_value(shr_row, "shr_year", None) is not None else None,
                String(str(_row_value(shr_row, "entity_uid", ""))),
                String(candidate_set_hash),
                humanizing_binary,
                String(run_status),
                trigger_article_id,
                articles_total,
                articles_attempted,
                articles_successful,
                articles_failed,
            )
        ),
    )


def _process_single_shr(run_id: str, shr_row) -> Run[ShrProcessResult]:
    shr_uid = str(_row_value(shr_row, "shr_uid", ""))
    entity_uid = str(_row_value(shr_row, "entity_uid", ""))
    candidate_set_hash = str(_row_value(shr_row, "candidate_set_hash", ""))

    def _loop(
        idx: int,
        candidates: list,
        attempted: int,
        successful: int,
        failed: int,
        cache_hits: int,
        cache_misses: int,
        in_tokens: int,
        c_tokens: int,
        out_tokens: int,
        r_tokens: int,
        est_cost: float,
        found_positive: bool,
        trigger_article_id: int | None,
    ) -> Run[ShrProcessResult]:
        if found_positive or idx >= len(candidates):
            humanizing_binary: int | None
            run_status: str
            if found_positive:
                humanizing_binary = 1
                run_status = "complete"
            elif failed > 0:
                humanizing_binary = None
                run_status = "pending_retry"
            else:
                humanizing_binary = 0
                run_status = "complete"

            return (
                _persist_shr_result(
                    run_id,
                    shr_row,
                    candidate_set_hash,
                    humanizing_binary,
                    run_status,
                    trigger_article_id,
                    len(candidates),
                    attempted,
                    successful,
                    failed,
                )
                ^ pure(
                    ShrProcessResult(
                        shr_uid=shr_uid,
                        run_status=run_status,
                        humanizing_binary=humanizing_binary,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        input_tokens=in_tokens,
                        cached_tokens=c_tokens,
                        output_tokens=out_tokens,
                        reasoning_tokens=r_tokens,
                        est_cost=est_cost,
                    )
                )
            )

        candidate = candidates[idx]
        return (
            sql_query(SQL("SELECT 1;"))
            >> (
                lambda _noop: (
                    _resolve_candidate(candidate)
                    >> (
                        lambda dec: _write_article_result(
                            run_id,
                            shr_uid,
                            entity_uid,
                            candidate,
                            dec.used_cache,
                            dec.decision_label,
                            dec.decision_binary,
                            "success",
                            None,
                            dec.input_tokens,
                            dec.cached_tokens,
                            dec.output_tokens,
                            dec.reasoning_tokens,
                            dec.est_cost,
                        )
                        ^ _loop(
                            idx + 1,
                            candidates,
                            attempted + 1,
                            successful + 1,
                            failed,
                            cache_hits + (1 if dec.used_cache else 0),
                            cache_misses + (0 if dec.used_cache else 1),
                            in_tokens + dec.input_tokens,
                            c_tokens + dec.cached_tokens,
                            out_tokens + dec.output_tokens,
                            r_tokens + dec.reasoning_tokens,
                            est_cost + dec.est_cost,
                            dec.decision_binary == 1,
                            dec.article_id if dec.decision_binary == 1 else trigger_article_id,
                        )
                    )
                )
            )
        )

    def _handle_candidates(rows: Array) -> Run[ShrProcessResult]:
        candidates = list(rows)

        def _run_one() -> Run[ShrProcessResult]:
            return _loop(
                0,
                candidates,
                attempted=0,
                successful=0,
                failed=0,
                cache_hits=0,
                cache_misses=0,
                in_tokens=0,
                c_tokens=0,
                out_tokens=0,
                r_tokens=0,
                est_cost=0.0,
                found_positive=False,
                trigger_article_id=None,
            )

        return _run_one()

    return _retrieve_candidates_for_shr(shr_uid) >> _handle_candidates


def _process_single_shr_safe(run_id: str, shr_row) -> Run[ShrProcessResult]:
    shr_uid = str(_row_value(shr_row, "shr_uid", ""))
    entity_uid = str(_row_value(shr_row, "entity_uid", ""))

    return (
        _retrieve_candidates_for_shr(shr_uid)
        >> (
            lambda rows: _process_candidates_with_error_capture(
                run_id,
                shr_row,
                entity_uid,
                list(rows),
            )
        )
    )


def _process_candidates_with_error_capture(
    run_id: str,
    shr_row,
    entity_uid: str,
    candidates: list,
) -> Run[ShrProcessResult]:
    shr_uid = str(_row_value(shr_row, "shr_uid", ""))
    candidate_set_hash = str(_row_value(shr_row, "candidate_set_hash", ""))

    def _loop(
        idx: int,
        attempted: int,
        successful: int,
        failed: int,
        cache_hits: int,
        cache_misses: int,
        in_tokens: int,
        c_tokens: int,
        out_tokens: int,
        r_tokens: int,
        est_cost: float,
        found_positive: bool,
        trigger_article_id: int | None,
    ) -> Run[ShrProcessResult]:
        if found_positive or idx >= len(candidates):
            humanizing_binary: int | None
            run_status: str
            if found_positive:
                humanizing_binary = 1
                run_status = "complete"
            elif failed > 0:
                humanizing_binary = None
                run_status = "pending_retry"
            else:
                humanizing_binary = 0
                run_status = "complete"
            return (
                _persist_shr_result(
                    run_id,
                    shr_row,
                    candidate_set_hash,
                    humanizing_binary,
                    run_status,
                    trigger_article_id,
                    len(candidates),
                    attempted,
                    successful,
                    failed,
                )
                ^ pure(
                    ShrProcessResult(
                        shr_uid=shr_uid,
                        run_status=run_status,
                        humanizing_binary=humanizing_binary,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        input_tokens=in_tokens,
                        cached_tokens=c_tokens,
                        output_tokens=out_tokens,
                        reasoning_tokens=r_tokens,
                        est_cost=est_cost,
                    )
                )
            )

        cand = candidates[idx]
        return (
            _resolve_candidate(cand)
            >> (
                lambda dec: _write_article_result(
                    run_id,
                    shr_uid,
                    entity_uid,
                    cand,
                    dec.used_cache,
                    dec.decision_label,
                    dec.decision_binary,
                    "success",
                    None,
                    dec.input_tokens,
                    dec.cached_tokens,
                    dec.output_tokens,
                    dec.reasoning_tokens,
                    dec.est_cost,
                )
                ^ _loop(
                    idx + 1,
                    attempted + 1,
                    successful + 1,
                    failed,
                    cache_hits + (1 if dec.used_cache else 0),
                    cache_misses + (0 if dec.used_cache else 1),
                    in_tokens + dec.input_tokens,
                    c_tokens + dec.cached_tokens,
                    out_tokens + dec.output_tokens,
                    r_tokens + dec.reasoning_tokens,
                    est_cost + dec.est_cost,
                    dec.decision_binary == 1,
                    dec.article_id if dec.decision_binary == 1 else trigger_article_id,
                )
            )
        )

    def _handle_error(idx: int, err: ErrorPayload, state_vals: tuple[int, int, int, int, int, int, int, int, int, float, bool, int | None]) -> Run[ShrProcessResult]:
        attempted, successful, failed, cache_hits, cache_misses, in_tokens, c_tokens, out_tokens, r_tokens, est_cost, found_positive, trigger_article_id = state_vals
        if isinstance(err.app, UserAbort):
            return throw(err)
        cand = candidates[idx]
        return (
            _write_article_result(
                run_id,
                shr_uid,
                entity_uid,
                cand,
                False,
                None,
                None,
                "error",
                str(err),
                0,
                0,
                0,
                0,
                0.0,
            )
            ^ _loop(
                idx + 1,
                attempted + 1,
                successful,
                failed + 1,
                cache_hits,
                cache_misses + 1,
                in_tokens,
                c_tokens,
                out_tokens,
                r_tokens,
                est_cost,
                found_positive,
                trigger_article_id,
            )
        )

    def _step(
        idx: int,
        attempted: int,
        successful: int,
        failed: int,
        cache_hits: int,
        cache_misses: int,
        in_tokens: int,
        c_tokens: int,
        out_tokens: int,
        r_tokens: int,
        est_cost: float,
        found_positive: bool,
        trigger_article_id: int | None,
    ) -> Run[ShrProcessResult]:
        if found_positive or idx >= len(candidates):
            return _loop(
                idx,
                attempted,
                successful,
                failed,
                cache_hits,
                cache_misses,
                in_tokens,
                c_tokens,
                out_tokens,
                r_tokens,
                est_cost,
                found_positive,
                trigger_article_id,
            )
        return (
            ask()
            >> (
                lambda _env: _resolve_candidate(candidates[idx])
                >> (
                    lambda dec: _write_article_result(
                        run_id,
                        shr_uid,
                        entity_uid,
                        candidates[idx],
                        dec.used_cache,
                        dec.decision_label,
                        dec.decision_binary,
                        "success",
                        None,
                        dec.input_tokens,
                        dec.cached_tokens,
                        dec.output_tokens,
                        dec.reasoning_tokens,
                        dec.est_cost,
                    )
                    ^ _step(
                        idx + 1,
                        attempted + 1,
                        successful + 1,
                        failed,
                        cache_hits + (1 if dec.used_cache else 0),
                        cache_misses + (0 if dec.used_cache else 1),
                        in_tokens + dec.input_tokens,
                        c_tokens + dec.cached_tokens,
                        out_tokens + dec.output_tokens,
                        r_tokens + dec.reasoning_tokens,
                        est_cost + dec.est_cost,
                        dec.decision_binary == 1,
                        dec.article_id if dec.decision_binary == 1 else trigger_article_id,
                    )
                )
            )
        )

    # Lightweight explicit error capture around each candidate run.
    def _go(
        idx: int,
        attempted: int,
        successful: int,
        failed: int,
        cache_hits: int,
        cache_misses: int,
        in_tokens: int,
        c_tokens: int,
        out_tokens: int,
        r_tokens: int,
        est_cost: float,
        found_positive: bool,
        trigger_article_id: int | None,
    ) -> Run[ShrProcessResult]:
        if found_positive or idx >= len(candidates):
            return _loop(
                idx,
                attempted,
                successful,
                failed,
                cache_hits,
                cache_misses,
                in_tokens,
                c_tokens,
                out_tokens,
                r_tokens,
                est_cost,
                found_positive,
                trigger_article_id,
            )

        from pymonad import run_except

        return run_except(_resolve_candidate(candidates[idx])) >> (
            lambda ei: (
                _handle_error(
                    idx,
                    ei.l,
                    (attempted, successful, failed, cache_hits, cache_misses, in_tokens, c_tokens, out_tokens, r_tokens, est_cost, found_positive, trigger_article_id),
                )
                if isinstance(ei, Left)
                else _write_article_result(
                    run_id,
                    shr_uid,
                    entity_uid,
                    candidates[idx],
                    ei.r.used_cache,
                    ei.r.decision_label,
                    ei.r.decision_binary,
                    "success",
                    None,
                    ei.r.input_tokens,
                    ei.r.cached_tokens,
                    ei.r.output_tokens,
                    ei.r.reasoning_tokens,
                    ei.r.est_cost,
                )
                ^ _go(
                    idx + 1,
                    attempted + 1,
                    successful + 1,
                    failed,
                    cache_hits + (1 if ei.r.used_cache else 0),
                    cache_misses + (0 if ei.r.used_cache else 1),
                    in_tokens + ei.r.input_tokens,
                    c_tokens + ei.r.cached_tokens,
                    out_tokens + ei.r.output_tokens,
                    r_tokens + ei.r.reasoning_tokens,
                    est_cost + ei.r.est_cost,
                    ei.r.decision_binary == 1,
                    ei.r.article_id if ei.r.decision_binary == 1 else trigger_article_id,
                )
            )
        )

    return _go(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, False, None)


def _process_shr_row(run_id: str, row) -> Run[ShrProcessResult]:
    return _process_single_shr_safe(run_id, row)


def _sum_result_int(results: list[ShrProcessResult], attr: str) -> int:
    return sum(int(getattr(r, attr)) for r in results)


def _sum_result_float(results: list[ShrProcessResult], attr: str) -> float:
    return sum(float(getattr(r, attr)) for r in results)


def _write_run_history(
    run_id: str,
    requested_count: int,
    processed_count: int,
    humanizing_count: int,
    not_humanizing_count: int,
    pending_retry_count: int,
    cache_hits: int,
    cache_misses: int,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    est_cost: float,
    elapsed_display: str,
    export_path: str,
) -> Run[Unit]:
    return sql_exec(
        SQL(
            """
            INSERT INTO humanization_run_history (
              run_id, requested_count, processed_count,
              humanizing_count, not_humanizing_count, pending_retry_count,
              cache_hits, cache_misses,
              input_tokens, cached_tokens, output_tokens, reasoning_tokens, est_cost,
              elapsed_display, export_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
            ON CONFLICT(run_id) DO UPDATE SET
              requested_count = EXCLUDED.requested_count,
              processed_count = EXCLUDED.processed_count,
              humanizing_count = EXCLUDED.humanizing_count,
              not_humanizing_count = EXCLUDED.not_humanizing_count,
              pending_retry_count = EXCLUDED.pending_retry_count,
              cache_hits = EXCLUDED.cache_hits,
              cache_misses = EXCLUDED.cache_misses,
              input_tokens = EXCLUDED.input_tokens,
              cached_tokens = EXCLUDED.cached_tokens,
              output_tokens = EXCLUDED.output_tokens,
              reasoning_tokens = EXCLUDED.reasoning_tokens,
              est_cost = EXCLUDED.est_cost,
              elapsed_display = EXCLUDED.elapsed_display,
              export_path = EXCLUDED.export_path,
              created_at = NOW();
            """
        ),
        SQLParams(
            (
                String(run_id),
                requested_count,
                processed_count,
                humanizing_count,
                not_humanizing_count,
                pending_retry_count,
                cache_hits,
                cache_misses,
                input_tokens,
                cached_tokens,
                output_tokens,
                reasoning_tokens,
                est_cost,
                String(elapsed_display),
                String(export_path),
            )
        ),
    )


def _export_results(run_id: str) -> Run[str]:
    filename = f"out/humanization_shr_export_{run_id}.csv"
    return (
        sql_export(
            SQL(
                """
                SELECT
                  r.shr_uid,
                  r.shr_year,
                  r.entity_uid,
                  r.humanizing_binary,
                  r.run_status,
                  r.trigger_article_id,
                  r.articles_total,
                  r.articles_attempted,
                  r.articles_successful,
                  r.articles_failed,
                  r.candidate_set_hash,
                  r.run_id,
                  r.processed_at,
                  sc.victim_age,
                  sc.victim_sex,
                  sc.victim_race,
                  sc.victim_ethnicity
                FROM humanization_shr_result r
                LEFT JOIN shr_cached sc
                  ON CAST(sc.unique_id AS VARCHAR) = r.shr_uid
                ORDER BY r.shr_year, r.shr_uid;
                """
            ),
            filename,
        )
        ^ pure(filename)
    )


def _render_summary(
    run_id: str,
    requested_count: int,
    process_result: Either[StopRun[Any, ShrProcessResult], ProcessAcc[Any, ShrProcessResult]],
    elapsed_display: str,
    export_path: str,
    export_rows: int,
) -> Run[NextStep]:
    match process_result:
        case Left(stop):
            results = list(stop.acc.results)
            processed = stop.acc.processed
            humanizing_count = sum(1 for r in results if r.humanizing_binary == 1)
            not_humanizing_count = sum(1 for r in results if r.humanizing_binary == 0)
            pending_retry_count = sum(1 for r in results if r.run_status == "pending_retry")
            cache_hits = _sum_result_int(results, "cache_hits")
            cache_misses = _sum_result_int(results, "cache_misses")
            input_tokens = _sum_result_int(results, "input_tokens")
            cached_tokens = _sum_result_int(results, "cached_tokens")
            output_tokens = _sum_result_int(results, "output_tokens")
            reasoning_tokens = _sum_result_int(results, "reasoning_tokens")
            est_cost = _sum_result_float(results, "est_cost")
            return (
                _write_run_history(
                    run_id=run_id,
                    requested_count=requested_count,
                    processed_count=processed,
                    humanizing_count=humanizing_count,
                    not_humanizing_count=not_humanizing_count,
                    pending_retry_count=pending_retry_count,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    input_tokens=input_tokens,
                    cached_tokens=cached_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    est_cost=est_cost,
                    elapsed_display=elapsed_display,
                    export_path=export_path,
                )
                ^ put_line(
                    "Humanization run summary (stopped early)\n"
                    f"Requested SHR rows: {requested_count}\n"
                    f"Processed SHR rows: {processed}\n"
                    f"humanizing=1: {humanizing_count}\n"
                    f"humanizing=0: {not_humanizing_count}\n"
                    f"pending_retry: {pending_retry_count}\n"
                    f"cache hits: {cache_hits}\n"
                    f"cache misses: {cache_misses}\n"
                    f"input tokens: {input_tokens}\n"
                    f"cached tokens: {cached_tokens}\n"
                    f"output tokens: {output_tokens}\n"
                    f"reasoning tokens: {reasoning_tokens}\n"
                    f"estimated cost: ${est_cost:.4f}\n"
                    f"{elapsed_display}\n"
                    f"Export: {export_path} (rows={export_rows})\n"
                )
                ^ pure(NextStep.CONTINUE)
            )
        case Right(acc):
            results = list(acc.results)
            processed = acc.processed
            humanizing_count = sum(1 for r in results if r.humanizing_binary == 1)
            not_humanizing_count = sum(1 for r in results if r.humanizing_binary == 0)
            pending_retry_count = sum(1 for r in results if r.run_status == "pending_retry")
            cache_hits = _sum_result_int(results, "cache_hits")
            cache_misses = _sum_result_int(results, "cache_misses")
            input_tokens = _sum_result_int(results, "input_tokens")
            cached_tokens = _sum_result_int(results, "cached_tokens")
            output_tokens = _sum_result_int(results, "output_tokens")
            reasoning_tokens = _sum_result_int(results, "reasoning_tokens")
            est_cost = _sum_result_float(results, "est_cost")
            return (
                _write_run_history(
                    run_id=run_id,
                    requested_count=requested_count,
                    processed_count=processed,
                    humanizing_count=humanizing_count,
                    not_humanizing_count=not_humanizing_count,
                    pending_retry_count=pending_retry_count,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    input_tokens=input_tokens,
                    cached_tokens=cached_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    est_cost=est_cost,
                    elapsed_display=elapsed_display,
                    export_path=export_path,
                )
                ^ put_line(
                    "Humanization run summary\n"
                    f"Requested SHR rows: {requested_count}\n"
                    f"Processed SHR rows: {processed}\n"
                    f"humanizing=1: {humanizing_count}\n"
                    f"humanizing=0: {not_humanizing_count}\n"
                    f"pending_retry: {pending_retry_count}\n"
                    f"cache hits: {cache_hits}\n"
                    f"cache misses: {cache_misses}\n"
                    f"input tokens: {input_tokens}\n"
                    f"cached tokens: {cached_tokens}\n"
                    f"output tokens: {output_tokens}\n"
                    f"reasoning tokens: {reasoning_tokens}\n"
                    f"estimated cost: ${est_cost:.4f}\n"
                    f"{elapsed_display}\n"
                    f"Export: {export_path} (rows={export_rows})\n"
                )
                ^ pure(NextStep.CONTINUE)
            )
    raise RuntimeError("Unreachable Either branch")


def _process_queue_rows(run_id: str, rows: Array, requested_count: int) -> Run[NextStep]:
    return process_items(
        render=render_as_failure,
        happy=lambda row: _process_shr_row(run_id, row),
        items=rows,
    ) >> (
        lambda process_result: read_elapsed_display(RUN_TIMER_NAME)
        >> (
            lambda elapsed_maybe: _export_results(run_id)
            >> (
                lambda export_path: sql_query(
                    SQL("SELECT COUNT(*) AS n FROM humanization_shr_result;")
                ) >> (
                    lambda count_rows: _render_summary(
                        run_id,
                        requested_count,
                        process_result,
                        str(elapsed_maybe.a) if isinstance(elapsed_maybe, Just) else "Elapsed: n/a",
                        export_path,
                        int(_row_value(count_rows[0], "n", 0)) if len(count_rows) > 0 else 0,
                    )
                )
            )
        )
    )


def _run_pipeline() -> Run[NextStep]:
    run_id = datetime.now(timezone.utc).strftime("humanize_%Y%m%dT%H%M%SZ")
    return (
        start_run_timer(RUN_TIMER_NAME)
        ^ put_line(f"Step 1 JSON schema:\n{to_json(HumanizationExtractResponse)}")
        ^ put_line(f"Step 2 JSON schema:\n{to_json(HumanizationDeidentifyResponse)}")
        ^ put_line(f"Step 3 JSON schema:\n{to_json(HumanizationDecisionResponse)}")
        ^ _ensure_tables()
        ^ _build_current_candidates()
        ^ _build_reprocess_queue()
        ^ _display_queue_counts()
        ^ _input_number_to_process()
        >> (
            lambda requested: (
                _retrieve_shr_queue(requested)
                >> (
                    lambda rows: put_line(
                        f"Selected {len(rows)} SHR rows for processing.\n"
                    )
                    ^ _process_queue_rows(run_id, rows, requested)
                )
            )
        )
    )


def humanization_shr() -> Run[NextStep]:
    """
    Run the SHR-level humanization controller in the humanization namespace.
    """
    return with_models(
        MODELS,
        with_namespace(
            Namespace("humanization"),
            to_prompts(PROMPTS),
            with_duckdb(_run_pipeline()),
        ),
    )
