"""
Compute cosine similarity between summaries from two articles' incidents.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Mapping, cast

from calculations import cosine_similarity, sbert_average_vector
from menuprompts import NextStep
from pymonad import (
    Environment,
    Namespace,
    PromptKey,
    Run,
    SQL,
    SQLParams,
    String,
    Unit,
    ask,
    input_number,
    input_with_prompt,
    pure,
    put_line,
    sql_exec,
    sql_query,
    to_prompts,
    with_duckdb,
    with_namespace,
)

VEC_SIM_PROMPTS: dict[str, str | tuple[str, str] | tuple[str]] = {
    "article_left": "Enter the first article RecordId (or entity id): ",
    "article_right": "Enter the second article RecordId (or entity id): ",
    "incident_left": "Select incident number for the first article: ",
    "incident_right": "Select incident number for the second article: ",
}


def _normalize_text(text: str) -> str:
    clean = re.sub(r"[\x00-\x1f\x7f]+", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _vec_to_array_sql(vec: Iterable[float]) -> str:
    return f"ARRAY[{','.join(format(x, '.17g') for x in vec)}]"


def _coerce_vec(raw) -> tuple[float, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, tuple):
        return tuple(float(x) for x in raw)
    if isinstance(raw, list):
        return tuple(float(x) for x in raw)
    try:
        return tuple(float(x) for x in list(raw))
    except TypeError:
        return tuple()


def _ensure_sbert_cache_table() -> Run[Unit]:
    return sql_exec(
        SQL(
            """
        CREATE TABLE IF NOT EXISTS sbert_cache (
            input_text VARCHAR,
            vec DOUBLE[384]
        );
        """
        )
    )


def _get_or_create_vec(env: Environment, text: str) -> Run[tuple[float, ...]]:
    key = _normalize_text(text)
    if key == "":
        return put_line("[W] Empty text provided; skipping vector.") ^ pure(tuple())

    def _cache_miss() -> Run[tuple[float, ...]]:
        vec_vals = sbert_average_vector(env["fasttext_model"].model, key)
        return (
            put_line(f"[I] Cache miss for text: {key[:60]}")
            ^ sql_exec(
                SQL(
                    f"INSERT INTO sbert_cache (input_text, vec) VALUES (?, {_vec_to_array_sql(vec_vals)});"
                ),
                SQLParams((String(key),)),
            )
            ^ pure(tuple(vec_vals))
        )

    return with_duckdb(
        _ensure_sbert_cache_table()
        ^ sql_query(
            SQL("SELECT vec FROM sbert_cache WHERE input_text = ?;"),
            SQLParams((String(key),)),
        )
        >> (
            lambda rows: (
                put_line(f"[I] Using cached vector for text: {key[:60]}")
                ^ pure(_coerce_vec(rows[0].get("vec")))
                if len(rows) > 0
                else _cache_miss()
            )
        )
    )


def _normalize_record_id(record_id: int) -> int:
    if 0 < record_id < 100000000:
        return record_id + 100000000
    return record_id


def _retrieve_article_row(record_id: int) -> Run[dict[str, object]]:
    def _validate(rows) -> Run[dict[str, object]]:
        if len(rows) == 0:
            return (
                put_line(f"[E] No article found for RecordId {record_id}.")
                ^ pure({})
            )
        if len(rows) > 1:
            return (
                put_line(f"[E] Multiple articles found for RecordId {record_id}.")
                ^ pure({})
            )
        return pure(dict(rows[0]))

    return sql_query(
        SQL(
            """
        SELECT RecordId, Title, gptVictimJson
        FROM articles
        WHERE RecordId = ?;
        """
        ),
        SQLParams((record_id,)),
    ) >> _validate


def _row_value(row: Mapping, key: str):
    if not row:
        return None
    try:
        return row[key]
    except Exception:  # noqa: BLE001
        return None


def _extract_incident_summaries(raw_json: str) -> list[str]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []
    incidents: list[object] = []
    if isinstance(data, dict) and "incidents" in data:
        incidents = data.get("incidents") or []
    elif isinstance(data, list):
        incidents = data
    if not isinstance(incidents, list):
        return []
    summaries = []
    for item in incidents:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if isinstance(summary, str) and summary.strip():
            summaries.append(summary.strip())
    return summaries


def _looks_like_entity_id(raw: str) -> bool:
    return ":" in raw


def _get_entity_vector(entity_id: str) -> Run[tuple[float, ...]]:
    def _validate(rows) -> Run[tuple[float, ...]]:
        if len(rows) == 0:
            return put_line(
                f"[E] No entity found for id {entity_id}."
            ) ^ pure(tuple())
        row = rows[0]
        vec = _coerce_vec(row.get("summary_vec"))
        if not vec:
            return put_line(
                f"[E] Entity {entity_id} has no summary vector."
            ) ^ pure(tuple())
        article_ids_csv = row.get("article_ids_csv")
        if isinstance(article_ids_csv, str) and article_ids_csv.strip():
            return put_line(
                f"[I] Entity articles: {article_ids_csv}"
            ) ^ pure(vec)
        return pure(vec)

    return with_duckdb(
        sql_query(
            SQL(
                """
            SELECT summary_vec, article_ids_csv
            FROM victim_entity_reps
            WHERE victim_entity_id = ?;
            """
            ),
            SQLParams((String(entity_id),)),
        )
        >> _validate
    )


def _prompt_for_entity_id(
    record_prompt: PromptKey,
    label: str,
) -> Run[tuple[float, ...] | None]:
    def _loop() -> Run[tuple[float, ...] | None]:
        return (
            input_with_prompt(record_prompt)
            >> (lambda raw_input: pure(str(raw_input).strip()))
            >> (lambda raw: (
                pure(cast(tuple[float, ...] | None, None))
                if raw.upper() == "Q"
                else (
                    put_line(f"[E] {label} id is empty.")
                    >> (lambda _: _loop())
                    if raw == ""
                    else (
                        _get_entity_vector(raw)
                        >> (
                            lambda vec: _loop()
                            if not vec
                            else pure(cast(tuple[float, ...] | None, vec))
                        )
                        if _looks_like_entity_id(raw)
                        else put_line(
                            f"[E] {label} id must be a number or entity id."
                        )
                        >> (lambda _: _loop())
                    )
                )
            ))
        )
    return _loop()


def _select_incident_summary(
    summaries: list[str],
    prompt_key: PromptKey,
    label: str,
) -> Run[str]:
    if len(summaries) == 1:
        return put_line(f"[I] {label} incident summary:\n{summaries[0]}") ^ pure(
            summaries[0]
        )

    listing = "\n".join(
        f"  {idx}. {summary}" for idx, summary in enumerate(summaries, start=1)
    )
    return (
        put_line(f"[I] {label} incidents:")
        ^ put_line(listing)
        ^ input_number(prompt_key, min_value=1, max_value=len(summaries))
        >> (lambda choice: put_line(f"[I] Selected {label} incident #{choice}.")
            ^ put_line(summaries[choice - 1])
            ^ pure(summaries[choice - 1]))
    )


def _get_article_summary_for_id(
    record_id: int,
    incident_prompt: PromptKey,
    label: str,
) -> Run[str]:
    if record_id <= 0:
        return put_line(f"[E] {label} RecordId must be positive.") ^ pure("")

    return (
        pure(_normalize_record_id(record_id))
        >> _retrieve_article_row
        >> (
            lambda row: (
                put_line(f"[E] {label} article is missing gptVictimJson.")
                ^ pure("")
                if not row or not _row_value(row, "gptVictimJson")
                else (
                    pure(str(_row_value(row, "gptVictimJson")))
                    >> (lambda raw_json: (
                        lambda summaries: (
                            put_line(
                                f"[E] {label} article has no incidents or summaries."
                            )
                            ^ pure("")
                            if len(summaries) == 0
                            else _select_incident_summary(
                                summaries,
                                incident_prompt,
                                label,
                            )
                        )
                    )(_extract_incident_summaries(raw_json)))
                )
            )
        )
    )


def _resolve_vector_for_input(
    record_prompt: PromptKey,
    incident_prompt: PromptKey,
    label: str,
    env: Environment,
) -> Run[tuple[float, ...] | None]:
    def _loop() -> Run[tuple[float, ...] | None]:
        return (
            input_with_prompt(record_prompt)
            >> (lambda raw_input: pure(str(raw_input).strip()))
            >> (
                lambda raw: (
                    pure(cast(tuple[float, ...] | None, None))
                    if raw.upper() == "Q"
                    else (
                        put_line(f"[E] {label} id is empty.")
                        >> (lambda _: _loop())
                        if raw == ""
                        else (
                            _prompt_for_entity_id(record_prompt, label)
                            if _looks_like_entity_id(raw)
                            else (
                                _get_article_summary_for_id(
                                    int(raw),
                                    incident_prompt,
                                    label,
                                )
                                >> (lambda summary: (
                                    _loop()
                                    if summary == ""
                                    else _get_or_create_vec(env, summary)
                                    >> (
                                        lambda vec:
                                        pure(cast(tuple[float, ...] | None, vec))
                                    )
                                ))
                                if raw.isdigit()
                                else (
                                    put_line(
                                        f"[E] {label} id must be a number or entity id."
                                    )
                                    >> (lambda _: _loop())
                                )
                            )
                        )
                    )
                )
            )
        )
    return _loop()


def vector_similarity() -> Run[NextStep]:
    """
    Prompt for two article IDs, compute cosine similarity, and return to main menu.
    """
    def _run() -> Run[NextStep]:
        return ask() >> (
            lambda env:
            _resolve_vector_for_input(
                PromptKey("article_left"),
                PromptKey("incident_left"),
                "First",
                env,
            )
            >> (lambda vec_a:
                pure(NextStep.CONTINUE)
                if vec_a is None
                else (
                    _resolve_vector_for_input(
                        PromptKey("article_right"),
                        PromptKey("incident_right"),
                        "Second",
                        env,
                    )
                    >> (lambda vec_b:
                        pure(NextStep.CONTINUE)
                        if vec_b is None
                        else (
                            (
                                put_line(
                                    "[W] Unable to compute similarity due to missing vector."
                                )
                                if not vec_a or not vec_b
                                else put_line(
                                    f"[I] Cosine similarity: {cosine_similarity(vec_a, vec_b):.6f}"
                                )
                            )
                            ^ pure(NextStep.CONTINUE)
                        )
                    )
                )
            )
        )

    return with_namespace(
        Namespace("vector_similarity"),
        to_prompts(VEC_SIM_PROMPTS),
        _run(),
    )
