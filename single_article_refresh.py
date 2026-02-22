"""
Single-article post-[G] refresh for DuckDB incident/geocode tables and
adjudication invalidation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from geocode_incidents import (
    ALTER_CACHE_SQL,
    CACHE_GET_SQL,
    CREATE_ADDR_MAP_SQL,
    CREATE_CACHE_SQL,
    CREATE_GEOCODED_VIEW_SQL,
    INSERT_ADDR_MAP_SQL,
    INSERT_CACHE_SQL,
)
from incidents_setup import (
    VICTIMS_CACHED_ENH_SELECT_SQL,
    VICTIMS_CACHED_SELECT_SQL,
    _row_update_run,
)
from pymonad import (
    Array,
    Environment,
    Run,
    SQL,
    SQLParams,
    String,
    Unit,
    ask,
    geocode_address,
    mar_result_score,
    mar_result_type_with_input,
    pure,
    put_line,
    sql_exec,
    sql_query,
    sql_script,
    run_except,
    Left,
    Right,
    unit,
    with_duckdb,
)
from pymonad.traverse import array_traverse_run


def _refresh_sqlite_subset_row(record_id: int) -> Run[Unit]:
    def _run_core() -> Run[Unit]:
        return (
            sql_exec(
                SQL("DELETE FROM articles_wp_subset WHERE RecordId = ?;"),
                SQLParams((record_id,)),
            )
            ^ sql_exec(
                SQL(
                    """
                    INSERT INTO articles_wp_subset (
                        RecordId, Publication, PubDate, Title, FullText, gptVictimJson, LastUpdated
                    )
                    SELECT RecordId, Publication, PubDate, Title, FullText, gptVictimJson, LastUpdated
                    FROM articles
                    WHERE RecordId = ?
                      AND Dataset='CLASS_WP'
                      AND gptClass='M';
                    """
                ),
                SQLParams((record_id,)),
            )
            ^ sql_query(
                SQL("SELECT COUNT(*) AS n FROM articles_wp_subset WHERE RecordId = ?;"),
                SQLParams((record_id,)),
            )
            >> (
                lambda rows: put_line(
                    "[F] Refreshed articles_wp_subset row "
                    f"for {record_id}: present={rows[0]['n']}"
                )
                ^ pure(unit)
            )
        )

    return run_except(_run_core()) >> (
        lambda res: (
            (
                put_line("[F] Skipped subset refresh: articles_wp_subset does not exist.")
                if isinstance(res, Left)
                else pure(None)
            )
            ^ pure(unit)
        )
    )


def _is_article_in_scope_for_incidents(record_id: int) -> Run[bool]:
    return (
        sql_query(
            SQL(
                """
                SELECT COUNT(*) AS n
                FROM articles
                WHERE RecordId = ?
                  AND Dataset='CLASS_WP'
                  AND gptClass='M';
                """
            ),
            SQLParams((record_id,)),
        )
        >> (lambda rows: pure(rows[0]["n"] > 0))
    )


def _refresh_duckdb_incidents_for_article(env: Environment, record_id: int) -> Run[Unit]:
    return (
        put_line(f"[F] incidents refresh entry for article {record_id}.")
        ^ put_line(f"[F] incidents refresh step: ensure index on incidents_cached.article_id ({record_id})")
        ^ sql_exec(
            SQL(
                """
                CREATE INDEX IF NOT EXISTS idx_incidents_cached_article_id
                ON incidents_cached(article_id);
                """
            )
        )
        ^ put_line(f"[F] incidents refresh step: alter incidents_cached summary_vec ({record_id})")
        ^ sql_exec(
            SQL(
                """
                ALTER TABLE incidents_cached
                ADD COLUMN IF NOT EXISTS summary_vec DOUBLE[384];
                """
            )
        )
        ^ put_line(f"[F] incidents refresh step: delete old incidents row(s) ({record_id})")
        ^ sql_exec(
            SQL("DELETE FROM incidents_cached WHERE article_id = ?;"),
            SQLParams((record_id,)),
        )
        ^ put_line(f"[F] incidents refresh step: insert refreshed incidents row(s) ({record_id})")
        ^ sql_exec(
            SQL(
                """
                INSERT INTO incidents_cached (
                  article_id, city_id, incident_idx, incident_json,
                  publish_date, article_title, article_text,
                  year, month, day,
                  incident_date, event_start_day, event_end_day, midpoint_day,
                  date_precision, has_incident_date, interval_span_days,
                  location_raw, circumstance, weapon,
                  offender_count, offender_name, offender_age,
                  offender_sex, offender_race, offender_ethnicity,
                  victim_count, summary, summary_norm, summary_vec
                )
                SELECT
                  article_id, city_id, incident_idx, incident_json,
                  publish_date, article_title, article_text,
                  year, month, day,
                  incident_date, event_start_day, event_end_day, midpoint_day,
                  date_precision, has_incident_date, interval_span_days,
                  location_raw, circumstance, weapon,
                  offender_count, offender_name, offender_age,
                  offender_sex, offender_race, offender_ethnicity,
                  victim_count, summary, summary_norm,
                  CAST(NULL AS DOUBLE[384]) AS summary_vec
                FROM (
                  WITH base AS (
                        SELECT
                          a.RecordId AS article_id,
                          a.Publication AS city_id,
                          a.PubDate AS publish_date,
                          a.Title AS article_title,
                          a.FullText AS article_text,
                          CAST(i.key AS INTEGER) AS incident_idx,
                          i.value AS incident_json,
                          try_cast(json_extract_string(i.value,'$.year')  AS INTEGER) AS year_raw,
                          try_cast(json_extract_string(i.value,'$.month') AS INTEGER) AS month_raw,
                          try_cast(json_extract_string(i.value,'$.day')   AS INTEGER) AS day_raw,
                          json_extract_string(i.value,'$.location')       AS location_raw,
                          json_extract_string(i.value,'$.circumstance')   AS circumstance,
                          CASE
                            WHEN COALESCE(
                              json_extract_string(i.value,'$.killing_method'),
                              json_extract_string(i.value,'$.weapon')
                            ) = 'beating' THEN 'personal weapon'
                            ELSE COALESCE(
                              json_extract_string(i.value,'$.killing_method'),
                              json_extract_string(i.value,'$.weapon')
                            )
                          END AS weapon,
                          try_cast(json_extract_string(i.value,'$.offender_count') AS INTEGER) AS offender_count,
                          json_extract_string(i.value,'$.offender_name')  AS offender_name,
                          try_cast(json_extract_string(i.value,'$.offender_age')  AS INTEGER) AS offender_age,
                          json_extract_string(i.value,'$.offender_sex')   AS offender_sex,
                          json_extract_string(i.value,'$.offender_race')  AS offender_race,
                          json_extract_string(i.value,'$.offender_ethnicity') AS offender_ethnicity,
                          try_cast(json_extract_string(i.value,'$.victim_count') AS INTEGER) AS victim_count,
                          json_extract_string(i.value,'$.summary')        AS summary
                        FROM sqldb.articles_wp_subset a
                        CROSS JOIN json_each(
                          CASE
                            WHEN a.gptVictimJson IS NOT NULL
                                 AND a.gptVictimJson <> ''
                                 AND json_valid(a.gptVictimJson)
                              THEN a.gptVictimJson
                            ELSE '[]'
                          END
                        ) AS i
                        WHERE a.RecordId = ?
                      ),
                      dates AS (
                        SELECT
                          *,
                          CASE WHEN month_raw BETWEEN 1 AND 12 THEN month_raw END AS month_sane,
                          CASE WHEN day_raw BETWEEN 1 AND 31 THEN day_raw END AS day_sane
                        FROM base
                      ),
                      norm_dates AS (
                        SELECT
                          *,
                          CASE
                            WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL THEN 'day'
                            WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL THEN 'month'
                            WHEN year_raw IS NOT NULL THEN 'year'
                          END AS date_precision,
                          CASE
                            WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
                              THEN make_date(year_raw, month_sane, day_sane)
                          END AS incident_date,
                          CASE
                            WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
                              THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, day_sane))
                            WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NULL
                              THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, 1))
                            WHEN year_raw IS NOT NULL AND month_sane IS NULL
                              THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, 1, 1))
                          END AS event_start_day,
                          CASE
                            WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NOT NULL
                              THEN date_diff('day', DATE '1970-01-01', make_date(year_raw, month_sane, day_sane))
                            WHEN year_raw IS NOT NULL AND month_sane IS NOT NULL AND day_sane IS NULL
                              THEN date_diff(
                                'day',
                                DATE '1970-01-01',
                                (make_date(year_raw, month_sane, 1) + INTERVAL 1 MONTH - INTERVAL 1 DAY)
                              )
                            WHEN year_raw IS NOT NULL AND month_sane IS NULL
                              THEN date_diff(
                                'day',
                                DATE '1970-01-01',
                                (make_date(year_raw, 1, 1) + INTERVAL 1 YEAR - INTERVAL 1 DAY)
                              )
                          END AS event_end_day,
                          CASE
                            WHEN event_start_day IS NOT NULL AND event_end_day IS NOT NULL
                              THEN floor((event_start_day + event_end_day) / 2)
                          END AS midpoint_day
                        FROM dates
                      ),
                      norm AS (
                        SELECT
                          *,
                          ((event_start_day IS NOT NULL) AND (event_end_day IS NOT NULL)) AS has_incident_date,
                          (event_end_day - event_start_day) AS interval_span_days,
                          trim(
                            regexp_replace(
                              regexp_replace(coalesce(summary, ''), '[[:cntrl:]]+', '', 'g'),
                              '[[:space:]]+',
                              ' ',
                              'g'
                            )
                          ) AS summary_norm
                        FROM norm_dates
                      )
                      SELECT
                        article_id, city_id, incident_idx,
                        CAST(incident_json AS JSON) AS incident_json,
                        publish_date, article_title, article_text,
                        year_raw AS year, month_sane AS month, day_sane AS day,
                        incident_date, event_start_day, event_end_day, midpoint_day,
                        date_precision, has_incident_date, interval_span_days,
                        location_raw,
                        circumstance, weapon,
                        offender_count, offender_name, offender_age,
                        offender_sex, offender_race, offender_ethnicity,
                        victim_count, summary, summary_norm
                      FROM norm
                  WHERE year_raw >= 1977
                ) s;
                """
            ),
            SQLParams((record_id,)),
        )
        ^ put_line(f"[F] incidents refresh step: select refreshed incidents for vectorization ({record_id})")
        ^ sql_query(
            SQL(
                """
                SELECT article_id, incident_idx, summary_norm
                FROM incidents_cached
                WHERE article_id = ?;
                """
            ),
            SQLParams((record_id,)),
        )
        >> (
            lambda rows: put_line(
                f"[F] incidents_cached rows refreshed for {record_id}: {len(rows)}"
            )
            ^ array_traverse_run(
                Array.make(tuple(rows)),
                lambda r: _row_update_run(env, r),
            )
            ^ put_line(f"[F] Updated summary_vec for article {record_id}.")
        )
        ^ pure(unit)
    )


def _refresh_duckdb_victims_for_article(record_id: int) -> Run[Unit]:
    return (
        put_line(f"[F] victims refresh entry for article {record_id}.")
        ^ put_line(f"[F] victims refresh step: ensure index on victims_cached.article_id ({record_id})")
        ^ sql_exec(
            SQL(
                """
                ALTER TABLE victims_cached
                ADD COLUMN IF NOT EXISTS victim_relationship VARCHAR;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """
                ALTER TABLE victims_cached
                ADD COLUMN IF NOT EXISTS relationship VARCHAR;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """
                CREATE INDEX IF NOT EXISTS idx_victims_cached_article_id
                ON victims_cached(article_id);
                """
            )
        )
        ^ put_line(f"[F] victims refresh step: delete old victim row(s) ({record_id})")
        ^ sql_exec(
            SQL("DELETE FROM victims_cached WHERE article_id = ?;"),
            SQLParams((record_id,)),
        )
        ^ put_line(f"[F] victims refresh step: insert refreshed victim row(s) ({record_id})")
        ^ sql_exec(
            SQL(
                f"""
                INSERT INTO victims_cached (
                  article_id, city_id, publish_date, year, month, day,
                  incident_idx, summary_vec, victim_idx,
                  victim_name_raw, victim_age_raw, victim_sex, victim_race, victim_ethnicity,
                  victim_relationship, relationship,
                  incident_date, event_start_day, event_end_day,
                  weapon, circumstance,
                  offender_name, offender_count,
                  victim_count, victim_row_id
                )
                SELECT *
                FROM ({VICTIMS_CACHED_SELECT_SQL}) vc
                WHERE vc.article_id = ?;
                """
            ),
            SQLParams((record_id,)),
        )
        ^ sql_query(
            SQL("SELECT COUNT(*) AS n FROM victims_cached WHERE article_id = ?;"),
            SQLParams((record_id,)),
        )
        >> (
            lambda rows: put_line(
                "[F] victims_cached rows refreshed "
                f"for {record_id}: {rows[0]['n']}"
            )
            ^ pure(unit)
        )
    )


def _upsert_geocode_cache_for_addr(env: Environment, addr_raw: str) -> Run[Unit]:
    addr_key = (addr_raw or "").strip().upper()
    if addr_key == "":
        return pure(unit)

    def _handle_cached(rows) -> Run[Unit]:
        if len(rows) > 0:
            return pure(unit)

        return (
            geocode_address(addr_key, env["mar_key"])
            >> (lambda g: _insert_geocode_result(addr_key, g))
        )

    return (
        sql_exec(
            INSERT_ADDR_MAP_SQL,
            SQLParams((String(addr_raw), String(addr_key))),
        )
        ^ sql_query(CACHE_GET_SQL, SQLParams((String(addr_key),)))
        >> _handle_cached
    )


def _insert_geocode_result(addr_key: str, g) -> Run[Unit]:
    result_type = mar_result_type_with_input(addr_key, g.raw_json).value
    score_value = mar_result_score(g.raw_json)
    if g.ok:
        return (
            sql_exec(
                INSERT_CACHE_SQL,
                SQLParams(
                    (
                        String(addr_key),
                        String(g.matched_address),
                        g.x_lon,
                        g.y_lat,
                        String(json.dumps(g.raw_json)),
                        String(result_type),
                        score_value,
                    )
                ),
            )
            ^ pure(unit)
        )
    return (
        sql_exec(
            INSERT_CACHE_SQL,
            SQLParams(
                (
                    String(addr_key),
                    String(""),
                    None,
                    None,
                    String(json.dumps(g.raw_json)),
                    String(result_type),
                    score_value,
                )
            ),
        )
        ^ pure(unit)
    )


def _refresh_duckdb_geocode_for_article(env: Environment, record_id: int) -> Run[Unit]:
    def _run_core() -> Run[Unit]:
        return (
            sql_exec(CREATE_CACHE_SQL)
            ^ sql_exec(CREATE_ADDR_MAP_SQL)
            ^ sql_exec(ALTER_CACHE_SQL)
            ^ sql_query(
                SQL(
                    """
                    SELECT DISTINCT trim(coalesce(location_raw, '')) AS addr_raw
                    FROM incidents_cached
                    WHERE article_id = ?
                      AND trim(coalesce(location_raw, '')) <> '';
                    """
                ),
                SQLParams((record_id,)),
            )
            >> (
                lambda rows: array_traverse_run(
                    Array.make(tuple(rows)),
                    lambda r: _upsert_geocode_cache_for_addr(env, str(r["addr_raw"])),
                )
            )
            ^ sql_exec(CREATE_GEOCODED_VIEW_SQL)
            ^ sql_exec(
                SQL(
                    """
                    UPDATE victims_cached vc
                    SET geo_address_norm = g.geo_address_norm,
                        lon = CASE WHEN g.lon = 0 THEN NULL ELSE g.lon END,
                        lat = CASE WHEN g.lat = 0 THEN NULL ELSE g.lat END,
                        address_type = g.address_type,
                        geo_score = g.geo_score,
                        geo_address_short = g.geo_address_short,
                        geo_address_short_2 = g.geo_address_short_2
                    FROM stg_article_incidents_geo g
                    WHERE vc.article_id = ?
                      AND g.article_id = ?
                      AND vc.article_id = g.article_id
                      AND vc.incident_idx = g.incident_idx;
                    """
                ),
                SQLParams((record_id, record_id)),
            )
            ^ put_line(f"[F] Geocode refresh applied for article {record_id}.")
            ^ pure(unit)
        )

    return run_except(_run_core()) >> (
        lambda res: (
            (
                put_line("[F] Skipped geocode refresh: victims_cached is missing.")
                if isinstance(res, Left)
                else pure(None)
            )
            ^ pure(unit)
        )
    )


def _refresh_duckdb_victims_enh_for_article(record_id: int) -> Run[Unit]:
    return (
        put_line(f"[F] victims_enh refresh entry for article {record_id}.")
        ^ put_line(f"[F] victims_enh refresh step: ensure index on victims_cached_enh.article_id ({record_id})")
        ^ sql_exec(
            SQL(
                """
                ALTER TABLE victims_cached_enh
                ADD COLUMN IF NOT EXISTS victim_relationship VARCHAR;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """
                ALTER TABLE victims_cached_enh
                ADD COLUMN IF NOT EXISTS relationship VARCHAR;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """
                CREATE INDEX IF NOT EXISTS idx_victims_cached_enh_article_id
                ON victims_cached_enh(article_id);
                """
            )
        )
        ^ put_line(f"[F] victims_enh refresh step: delete old enhanced row(s) ({record_id})")
        ^ sql_exec(
            SQL("DELETE FROM victims_cached_enh WHERE article_id = ?;"),
            SQLParams((record_id,)),
        )
        ^ put_line(f"[F] victims_enh refresh step: insert refreshed enhanced row(s) ({record_id})")
        ^ sql_exec(
            SQL(
                f"""
                INSERT INTO victims_cached_enh
                SELECT *
                FROM ({VICTIMS_CACHED_ENH_SELECT_SQL}) v
                WHERE v.article_id = ?;
                """
            ),
            SQLParams((record_id,)),
        )
        ^ sql_query(
            SQL("SELECT COUNT(*) AS n FROM victims_cached_enh WHERE article_id = ?;"),
            SQLParams((record_id,)),
        )
        >> (
            lambda rows: put_line(
                "[F] victims_cached_enh rows refreshed "
                f"for {record_id}: {rows[0]['n']}"
            )
            ^ pure(unit)
        )
    )


def _delete_article_rows_if_exists(table_name: str, record_id: int) -> Run[Unit]:
    return run_except(
        sql_exec(
            SQL(f"DELETE FROM {table_name} WHERE article_id = ?;"),
            SQLParams((record_id,)),
        )
    ) >> (
        lambda res: (
            put_line(f"[F] Purged article {record_id} from {table_name}.")
            if isinstance(res, Right)
            else put_line(f"[F] Skipped purge for {table_name}: table unavailable.")
        )
        ^ pure(unit)
    )


def _purge_duckdb_for_non_m_article(record_id: int, run_id: str) -> Run[Unit]:
    return (
        put_line(
            f"[F] Article {record_id} no longer in CLASS_WP/M scope; "
            "purging cached incident/geocode rows and invalidating adjudications."
        )
        ^ _delete_article_rows_if_exists("incidents_cached", record_id)
        ^ _delete_article_rows_if_exists("victims_cached", record_id)
        ^ _delete_article_rows_if_exists("victims_cached_enh", record_id)
        ^ _invalidate_adjudications_for_article(record_id, run_id)
        ^ pure(unit)
    )


def _ensure_splink_udfs_loaded() -> Run[None]:
    return run_except(sql_script(SQL("LOAD splink_udfs;"))) >> (
        lambda res: (
            sql_script(
                SQL(
                    """
                    INSTALL splink_udfs FROM community;
                    LOAD splink_udfs;
                    """
                )
            )
            if isinstance(res, Left)
            else pure(None)
        )
    )


def _invalidate_adjudications_for_article(record_id: int, run_id: str) -> Run[Unit]:
    rec_id_s = str(record_id)
    rec_prefix = String(f"{rec_id_s}:%")

    def _run_core() -> Run[Unit]:
        return (
            put_line(f"[F] invalidation step: ensure history table ({record_id})")
            ^ sql_exec(
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
                )
            )
            ^ put_line(f"[F] invalidation step: collect candidates ({record_id})")
            ^ sql_exec(
                SQL(
                    """
                    CREATE OR REPLACE TEMP TABLE _invalidate_fixg AS
                    SELECT
                      o.*,
                      CASE
                        WHEN o.orphan_id LIKE ? THEN 'fixarticle_G_reextract_orphan_anchor'
                        WHEN o.resolved_entity_id IS NOT NULL
                             AND o.resolved_entity_id LIKE ?
                          THEN 'fixarticle_G_reextract_entity_anchor'
                        ELSE NULL
                      END AS invalidate_reason
                    FROM orphan_adjudication_overrides o
                    WHERE o.orphan_id LIKE ?
                       OR (
                           o.resolved_entity_id IS NOT NULL
                           AND o.resolved_entity_id LIKE ?
                       );
                    """
                ),
                SQLParams((rec_prefix, rec_prefix, rec_prefix, rec_prefix)),
            )
            ^ put_line(f"[F] invalidation step: count candidates ({record_id})")
            ^ sql_query(
                SQL(
                    """
                    SELECT
                      COUNT(*) AS n_all,
                      COUNT(*) FILTER (WHERE invalidate_reason = 'fixarticle_G_reextract_orphan_anchor') AS n_orphan_anchor,
                      COUNT(*) FILTER (WHERE invalidate_reason = 'fixarticle_G_reextract_entity_anchor') AS n_entity_anchor
                    FROM _invalidate_fixg;
                    """
                )
            )
            >> (
                lambda rows: (
                    (
                        put_line(f"[F] invalidation step: append history ({record_id})")
                        ^ sql_exec(
                            SQL(
                                f"""
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
                                )
                                SELECT
                                  NULL,
                                  '{run_id}',
                                  orphan_id,
                                  resolution_label,
                                  resolved_entity_id,
                                  confidence,
                                  'invalidated',
                                  NULL,
                                  NULL,
                                  invalidate_reason,
                                  evidence_json,
                                  'system_invalidation',
                                  NOW()
                                FROM _invalidate_fixg;
                                """
                            )
                        )
                        ^ put_line(f"[F] invalidation step: delete active overrides ({record_id})")
                        ^ sql_exec(
                            SQL(
                                """
                                DELETE FROM orphan_adjudication_overrides
                                WHERE orphan_id IN (SELECT orphan_id FROM _invalidate_fixg);
                                """
                            )
                        )
                        ^ put_line(
                            "[F] Invalidated adjudications for article "
                            f"{record_id}: total={rows[0]['n_all']}, "
                            f"orphan_anchor={rows[0]['n_orphan_anchor']}, "
                            f"entity_anchor={rows[0]['n_entity_anchor']}"
                        )
                    )
                    if rows[0]["n_all"] > 0
                    else put_line(f"[F] No adjudications invalidated for article {record_id}.")
                )
                ^ pure(unit)
            )
        )

    return (
        put_line(f"[F] invalidation entry for article {record_id}.")
        ^ run_except(_run_core())
        >> (
            lambda res: (
                (
                    put_line("[F] Skipped adjudication invalidation: orphan_adjudication_overrides is missing or unavailable.")
                    if isinstance(res, Left)
                    else pure(None)
                )
                ^ pure(unit)
            )
        )
    )


def _refresh_duckdb_for_article(env: Environment, record_id: int, run_id: str) -> Run[Unit]:
    def _run_core() -> Run[Unit]:
        return (
            put_line(f"[F] Entered DuckDB refresh core for article {record_id}.")
            ^ _refresh_duckdb_incidents_for_article(env, record_id)
            ^ _refresh_duckdb_victims_for_article(record_id)
            ^ _refresh_duckdb_geocode_for_article(env, record_id)
            ^ _ensure_splink_udfs_loaded()
            ^ _refresh_duckdb_victims_enh_for_article(record_id)
            ^ _invalidate_adjudications_for_article(record_id, run_id)
            ^ pure(unit)
        )

    return (
        put_line(f"[F] DuckDB refresh probe start for article {record_id}.")
        ^ run_except(_run_core())
        >> (
            lambda res: (
                (
                    put_line("[F] Skipped DuckDB refresh: required base tables are missing.")
                    if isinstance(res, Left)
                    else pure(None)
                )
                ^ pure(unit)
            )
        )
    )


def refresh_single_article_after_extract(record_id: int) -> Run[Unit]:
    run_id = datetime.now(timezone.utc).strftime(f"fixg_{record_id}_%Y%m%dT%H%M%SZ")
    def _after_scope(scope_ok: bool) -> Run[Unit]:
        return with_duckdb(
            ask()
            >> (
                lambda env: (
                    _refresh_duckdb_for_article(env, record_id, run_id)
                    if scope_ok
                    else _purge_duckdb_for_non_m_article(record_id, run_id)
                )
            )
        )

    return (
        put_line(f"[F] Starting single-article post-[G] refresh for {record_id} (run_id={run_id})...")
        ^ _refresh_sqlite_subset_row(record_id)
        ^ put_line(f"[F] Entering DuckDB single-article refresh for {record_id}...")
        ^ _is_article_in_scope_for_incidents(record_id)
        >> _after_scope
        ^ put_line(f"[F] Completed single-article post-[G] refresh for {record_id}.")
        ^ pure(unit)
    )
