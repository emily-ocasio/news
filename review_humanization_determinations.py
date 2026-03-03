"""
Controller to review and rerun SHR-member humanization determinations.
"""

from __future__ import annotations

from typing import Any

from menuprompts import NextStep
from pymonad import (
    DbBackend,
    Left,
    Namespace,
    PromptKey,
    Run,
    SQL,
    SQLParams,
    String,
    input_with_prompt,
    local,
    pure,
    put_line,
    run_except,
    sql_query,
    to_prompts,
    with_duckdb,
    with_models,
    with_namespace,
)
from humanization_pipeline import (
    MODELS,
    PROMPTS,
    build_humanization_candidates_current,
    ensure_humanization_tables,
    resolve_candidate,
)

REVIEW_PROMPTS: dict[str, str | tuple[str, str] | tuple[str,]] = {
    "review_shr_index": (
        "Enter SHR index number (0=[M]ain menu, [M]ain menu, [Q]uit): "
    ),
    "review_member_choice": "Enter line number, [R] new SHR, [M]ain menu, [Q]uit: ",
    "review_detail_choice": "Choose [Z] rerun, [B]ack, [R] new SHR, [M]ain menu, [Q]uit: ",
    "review_unlinked_choice": "Choose [R] new SHR, [M]ain menu, [Q]uit: ",
}

HUMANIZATION_KEYS = (
    "humanize_extract_incident",
    "humanize_deidentify",
    "humanize_classify",
)


def _with_sqlite(subprog: Run[Any]) -> Run[Any]:
    return local(lambda env: {**env, "current_backend": DbBackend.SQLITE}, subprog)


def _table_ready(table_name: str) -> Run[bool]:
    return run_except(sql_query(SQL(f"SELECT 1 FROM {table_name} LIMIT 1;"))) >> (
        lambda result: pure(not isinstance(result, Left))
    )


def _validate_prerequisites() -> Run[bool]:
    required = (
        "shr_cached",
        "shr_max_weight_matches",
        "victim_entity_members",
    )

    def _loop(idx: int) -> Run[bool]:
        if idx >= len(required):
            return pure(True)
        table = required[idx]
        return _table_ready(table) >> (
            lambda ok: _loop(idx + 1)
            if ok
            else put_line(
                "[R] Required linkage table is missing or unavailable: "
                f"{table}. Run [L] (and required upstream steps) first."
            )
            ^ pure(False)
        )

    return _loop(0)


def _shr_exists(shr_uid: str) -> Run[bool]:
    return sql_query(
        SQL(
            """
            SELECT 1
            FROM shr_cached
            WHERE CAST(unique_id AS VARCHAR) = ?
            LIMIT 1;
            """
        ),
        SQLParams((String(shr_uid),)),
    ) >> (lambda rows: pure(len(rows) > 0))


def _linked_entity_for_shr(shr_uid: str) -> Run[str | None]:
    return sql_query(
        SQL(
            """
            SELECT CAST(unique_id_l AS VARCHAR) AS entity_uid
            FROM shr_max_weight_matches
            WHERE CAST(unique_id_r AS VARCHAR) = ?
            LIMIT 1;
            """
        ),
        SQLParams((String(shr_uid),)),
    ) >> (
        lambda rows: pure(str(rows[0]["entity_uid"]) if len(rows) > 0 else None)
    )


def _rows_for_shr_entity(shr_uid: str, entity_uid: str):
    return sql_query(
        SQL(
            """
            WITH base AS (
              SELECT
                c.*,
                m.victim_row_id,
                CASE
                  WHEN h.step3_decision_label = 'humanizing' THEN 'humanizing'
                  WHEN h.step3_decision_label IN ('not humanizing', 'not_humanizing')
                    THEN 'not humanizing'
                  ELSE 'not tested'
                END AS row_determination
              FROM humanization_candidates_current c
              LEFT JOIN victim_entity_members m
                ON CAST(m.victim_entity_id AS VARCHAR) = c.entity_uid
               AND m.article_id = c.article_id
               AND m.incident_idx = c.incident_idx
               AND m.victim_idx = c.victim_idx
              LEFT JOIN humanization_stage_cache h
                ON h.incident_cache_key = c.incident_cache_key
               AND h.step1_status = 'success'
               AND h.step2_status = 'success'
               AND h.step3_status = 'success'
               AND h.step3_decision_label IN ('humanizing', 'not humanizing', 'not_humanizing')
              WHERE c.shr_uid = ?
                AND c.entity_uid = ?
            ),
            representative AS (
              SELECT
                b.*,
                ROW_NUMBER() OVER (
                  PARTITION BY b.article_id
                  ORDER BY b.specificity_score DESC, b.incident_idx ASC, b.victim_idx ASC
                ) AS rn
              FROM base b
            ),
            article_rollup AS (
              SELECT
                article_id,
                SUM(CASE WHEN row_determination = 'humanizing' THEN 1 ELSE 0 END) AS n_humanizing,
                SUM(CASE WHEN row_determination = 'not humanizing' THEN 1 ELSE 0 END) AS n_not_humanizing,
                SUM(CASE WHEN row_determination = 'not tested' THEN 1 ELSE 0 END) AS n_not_tested
              FROM base
              GROUP BY article_id
            )
            SELECT
              r.*,
              ar.n_humanizing,
              ar.n_not_humanizing,
              ar.n_not_tested,
              CASE
                WHEN ar.n_humanizing > 0 THEN 'humanizing'
                WHEN ar.n_not_tested > 0 THEN 'not tested'
                ELSE 'not humanizing'
              END AS determination
            FROM representative r
            JOIN article_rollup ar USING (article_id)
            WHERE r.rn = 1
            ORDER BY r.article_id ASC;
            """
        ),
        SQLParams((String(shr_uid), String(entity_uid))),
    )


def _display_summary(shr_uid: str, entity_uid: str, rows) -> Run[None]:
    if len(rows) == 0:
        return (
            put_line(
                f"SHR {shr_uid} is linked to entity {entity_uid}, but no canonical member candidates were found."
            )
            ^ pure(None)
        )

    top = rows[0]
    article_determinations = [str(row["determination"]) for row in rows]
    if any(d == "humanizing" for d in article_determinations):
        final_determination = "humanizing"
    elif all(d == "not humanizing" for d in article_determinations):
        final_determination = "not humanizing"
    else:
        final_determination = "not fully tested"
    header = (
        f"SHR {shr_uid} linked entity: {entity_uid}\n"
        f"Victim id: {entity_uid}\n"
        f"Canonical incident date: {top['incident_date_norm']}\n"
        f"Incident summary: {top['incident_summary_norm']}\n"
        f"Victim name: {top['victim_name_norm2']}\n"
        f"Victim age: {top['victim_age']}"
    )
    lines = "\n".join(
        f"{idx}. article_id={row['article_id']} determination={row['determination']}"
        for idx, row in enumerate(rows, start=1)
    )
    return (
        put_line(header)
        ^ put_line("Articles:")
        ^ put_line(lines)
        ^ put_line(f"Final victim determination: {final_determination}")
        ^ pure(None)
    )


def _show_latest_cache(selected_row) -> Run[None]:
    cache_key = str(selected_row["incident_cache_key"])
    return sql_query(
        SQL(
            """
            SELECT *
            FROM humanization_stage_cache
            WHERE incident_cache_key = ?
            LIMIT 1;
            """
        ),
        SQLParams((String(cache_key),)),
    ) >> (
        lambda rows: (
            put_line("No cached humanization stage data found for this member.")
            if len(rows) == 0
            else (lambda r: (
                (lambda step1_cost, step2_cost, step3_cost, total_cost:
                    put_line(
                        "Humanization cache:\n"
                        f"Step 1 status: {r['step1_status']}\n"
                        f"Step 1 output: {r['step1_output_text']}\n"
                        f"Step 1 reasoning: {r['step1_reasoning']}\n"
                        f"Step 1 estimated cost (per 1000 responses): ${step1_cost:.6f}\n"
                        "\n"
                        f"Step 2 status: {r['step2_status']}\n"
                        f"Step 2 output: {r['step2_output_text']}\n"
                        f"Step 2 reasoning: {r['step2_reasoning']}\n"
                        f"Step 2 estimated cost (per 1000 responses): ${step2_cost:.6f}\n"
                        "\n"
                        f"Step 3 status: {r['step3_status']}\n"
                        f"Step 3 decision: {r['step3_decision_label']}\n"
                        f"Step 3 reasoning: {r['step3_reasoning']}\n"
                        f"Step 3 estimated cost (per 1000 responses): ${step3_cost:.6f}\n"
                        "\n"
                        f"Combined estimated cost (per 1000 responses): ${total_cost:.6f}"
                    )
                )(
                    float(r["step1_cost"] or 0.0),
                    float(r["step2_cost"] or 0.0),
                    float(r["step3_cost"] or 0.0),
                    float(r["step1_cost"] or 0.0)
                    + float(r["step2_cost"] or 0.0)
                    + float(r["step3_cost"] or 0.0),
                )
            ))(rows[0])
        )
        ^ pure(None)
    )


def _show_latest_humanization_gpt_result(selected_row) -> Run[None]:
    article_id = int(selected_row["article_id"])
    return _with_sqlite(
        sql_query(
            SQL(
                """
                SELECT *
                FROM gptResults
                WHERE RecordId = ?
                  AND PromptKey IN (?, ?, ?)
                ORDER BY TimeStamp DESC, ResultId DESC
                LIMIT 1;
                """
            ),
            SQLParams(
                (
                    article_id,
                    String(HUMANIZATION_KEYS[0]),
                    String(HUMANIZATION_KEYS[1]),
                    String(HUMANIZATION_KEYS[2]),
                )
            ),
        )
    ) >> (
        lambda rows: (
            put_line("No humanization gptResults reasoning found for this member.")
            if len(rows) == 0
            else put_line(
                "Latest humanization gptResults:\n"
                f"PromptKey: {rows[0]['PromptKey']}\n"
                f"Timestamp: {rows[0]['TimeStamp']}\n"
                f"Reasoning: {rows[0]['Reasoning']}"
            )
        )
        ^ pure(None)
    )


def _detail_prompt(shr_uid: str, selected_row) -> Run[NextStep]:
    return (
        _show_latest_cache(selected_row)
        ^ _show_latest_humanization_gpt_result(selected_row)
        ^ input_with_prompt(PromptKey("review_detail_choice"))
        >> (lambda raw: _handle_detail_choice(shr_uid, selected_row, str(raw).strip().upper()))
    )


def _handle_detail_choice(shr_uid: str, selected_row, choice: str) -> Run[NextStep]:
    match choice:
        case "Q":
            return pure(NextStep.QUIT)
        case "M":
            return pure(NextStep.CONTINUE)
        case "R":
            return _select_shr()
        case "B":
            return _review_shr(shr_uid)
        case "Z":
            return (
                put_line("Re-running 3-step humanization for selected member...")
                ^ resolve_candidate(selected_row, force_refresh=True)
                >> (
                    lambda dec: put_line(
                        "Rerun complete. "
                        f"decision={dec.decision_label}, article_id={dec.article_id}"
                    )
                    ^ _detail_prompt(shr_uid, selected_row)
                )
            )
        case _:
            return put_line("Invalid choice. Enter Z, B, R, M, or Q.") ^ _detail_prompt(shr_uid, selected_row)


def _handle_member_choice(shr_uid: str, rows, choice: str) -> Run[NextStep]:
    upper = choice.upper()
    if upper == "Q":
        return pure(NextStep.QUIT)
    if upper == "M":
        return pure(NextStep.CONTINUE)
    if upper == "R":
        return _select_shr()
    if not choice.isdigit():
        return put_line("Please enter a valid line number, R, M, or Q.") ^ _member_prompt(shr_uid, rows)

    idx = int(choice)
    if idx <= 0 or idx > len(rows):
        return put_line("Line number out of range.") ^ _member_prompt(shr_uid, rows)

    selected_row = rows[idx - 1]
    return _detail_prompt(shr_uid, selected_row)


def _member_prompt(shr_uid: str, rows) -> Run[NextStep]:
    return input_with_prompt(PromptKey("review_member_choice")) >> (
        lambda raw: _handle_member_choice(shr_uid, rows, str(raw).strip())
    )


def _handle_unlinked_choice(choice: str) -> Run[NextStep]:
    upper = choice.upper()
    if upper == "Q":
        return pure(NextStep.QUIT)
    if upper == "M":
        return pure(NextStep.CONTINUE)
    if upper == "R":
        return _select_shr()
    return put_line("Invalid choice. Enter R, M, or Q.") ^ _unlinked_prompt()


def _unlinked_prompt() -> Run[NextStep]:
    return input_with_prompt(PromptKey("review_unlinked_choice")) >> (
        lambda raw: _handle_unlinked_choice(str(raw).strip())
    )


def _review_shr(shr_uid: str) -> Run[NextStep]:
    return (
        build_humanization_candidates_current()
        ^ _linked_entity_for_shr(shr_uid)
        >> (
            lambda entity_uid: (
                put_line(f"SHR {shr_uid} is not linked to any entity.") ^ _unlinked_prompt()
                if entity_uid is None
                else _rows_for_shr_entity(shr_uid, entity_uid)
                >> (
                    lambda rows: _display_summary(shr_uid, entity_uid, rows)
                    ^ _member_prompt(shr_uid, rows)
                )
            )
        )
    )


def _handle_shr_input(raw: str) -> Run[NextStep]:
    choice = raw.strip()
    upper = choice.upper()
    if upper == "Q":
        return pure(NextStep.QUIT)
    if upper == "M":
        return pure(NextStep.CONTINUE)
    if choice == "":
        return put_line("Please enter an SHR index number, M, or Q.") ^ _select_shr()
    if not choice.lstrip("-").isdigit():
        return put_line("Invalid input. Enter a numeric SHR index, M, or Q.") ^ _select_shr()

    shr_num = int(choice)
    if shr_num == 0:
        return pure(NextStep.CONTINUE)
    if shr_num < 0:
        return put_line("SHR index must be 0 or positive.") ^ _select_shr()

    shr_uid = str(shr_num)
    return _shr_exists(shr_uid) >> (
        lambda exists: _review_shr(shr_uid)
        if exists
        else put_line(
            f"No SHR record found for index {shr_uid}. Please try again."
        )
        ^ _select_shr()
    )


def _select_shr() -> Run[NextStep]:
    return input_with_prompt(PromptKey("review_shr_index")) >> (
        lambda raw: _handle_shr_input(str(raw))
    )


def _run_review() -> Run[NextStep]:
    return (
        ensure_humanization_tables()
        ^ _validate_prerequisites()
        >> (
            lambda ok: _select_shr()
            if ok
            else pure(NextStep.CONTINUE)
        )
    )


def review_humanization_determinations() -> Run[NextStep]:
    """
    Review and rerun SHR-member humanization determinations.
    """
    return with_models(
        MODELS,
        with_namespace(
            Namespace("humanization_review"),
            to_prompts(PROMPTS | REVIEW_PROMPTS),
            with_duckdb(_run_review()),
        ),
    )
