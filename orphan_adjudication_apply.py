"""
Apply persisted orphan adjudications on top of baseline orphan-linkage outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from menuprompts import NextStep
from pymonad import (
    Run,
    SQL,
    Unit,
    pure,
    put_line,
    sql_exec,
    sql_export,
    sql_query,
    unit,
    with_duckdb,
)


def _build_adjudication_candidates() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE adjudicated_orphan_candidates AS
                SELECT
                  orphan_id,
                  resolution_label,
                  resolved_entity_id,
                  confidence,
                  reason_summary,
                  evidence_json,
                  analyst_mode,
                  created_at,
                  updated_at
                FROM orphan_adjudication_overrides
                WHERE resolution_label = 'likely_missed_match'
                  AND resolved_entity_id IS NOT NULL;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE adjudicated_orphan_apply_report AS
                WITH c AS (
                  SELECT * FROM adjudicated_orphan_candidates
                ),
                dup_orphans AS (
                  SELECT orphan_id, COUNT(*) AS n_rows
                  FROM c
                  GROUP BY orphan_id
                  HAVING COUNT(*) > 1
                ),
                checks AS (
                  SELECT
                    c.orphan_id,
                    c.resolution_label,
                    c.resolved_entity_id,
                    c.confidence,
                    c.reason_summary,
                    c.evidence_json,
                    c.analyst_mode,
                    c.created_at,
                    c.updated_at,
                    o.unique_id AS orphan_exists,
                    o.article_id AS orphan_article_id,
                    e.victim_entity_id AS entity_exists,
                    fm.entity_uid AS machine_entity_uid,
                    d.n_rows AS dup_rows
                  FROM c
                  LEFT JOIN orphan_link_input o
                    ON o.unique_id = c.orphan_id
                  LEFT JOIN victim_entity_reps_new e
                    ON e.victim_entity_id = c.resolved_entity_id
                  LEFT JOIN final_orphan_matches fm
                    ON fm.orphan_uid = c.orphan_id
                  LEFT JOIN dup_orphans d
                    ON d.orphan_id = c.orphan_id
                )
                SELECT
                  orphan_id,
                  resolution_label,
                  resolved_entity_id,
                  confidence,
                  reason_summary,
                  evidence_json,
                  analyst_mode,
                  orphan_article_id,
                  machine_entity_uid,
                  CASE
                    WHEN dup_rows IS NOT NULL THEN 'skip_duplicate_adjudication_orphan'
                    WHEN orphan_exists IS NULL THEN 'skip_orphan_not_found'
                    WHEN entity_exists IS NULL THEN 'skip_entity_not_found'
                    WHEN machine_entity_uid IS NOT NULL
                         AND machine_entity_uid <> resolved_entity_id
                      THEN 'skip_conflict_entity_mismatch'
                    WHEN machine_entity_uid IS NOT NULL THEN 'skip_already_machine_matched'
                    ELSE 'applied'
                  END AS apply_status,
                  CASE
                    WHEN dup_rows IS NOT NULL THEN 'multiple adjudication rows found for orphan_id'
                    WHEN orphan_exists IS NULL THEN 'orphan_id missing from orphan_link_input'
                    WHEN entity_exists IS NULL THEN 'resolved_entity_id missing from victim_entity_reps_new'
                    WHEN machine_entity_uid IS NOT NULL
                         AND machine_entity_uid <> resolved_entity_id
                      THEN 'orphan already machine-matched to a different entity'
                    WHEN machine_entity_uid IS NOT NULL THEN 'orphan already matched by baseline Splink linkage'
                    ELSE 'eligible and applied'
                  END AS reason_detail
                FROM checks
                ORDER BY orphan_id;
                """
            )
        )
        ^ pure(unit)
    )


def _build_postadj_matches() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TEMP TABLE _applied_adjudication_rows AS
                SELECT
                  r.resolved_entity_id AS entity_uid,
                  r.orphan_id AS orphan_uid,
                  r.orphan_article_id AS article_id,
                  CAST(NULL AS DOUBLE) AS match_probability
                FROM adjudicated_orphan_apply_report r
                WHERE r.apply_status = 'applied';
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE final_orphan_matches_postadj AS
                WITH combined AS (
                  SELECT
                    0 AS src,
                    entity_uid,
                    orphan_uid,
                    article_id,
                    CAST(match_probability AS DOUBLE) AS match_probability
                  FROM final_orphan_matches

                  UNION ALL

                  SELECT
                    1 AS src,
                    entity_uid,
                    orphan_uid,
                    article_id,
                    match_probability
                  FROM _applied_adjudication_rows
                ),
                ranked AS (
                  SELECT
                    *,
                    ROW_NUMBER() OVER (
                      PARTITION BY orphan_uid
                      ORDER BY src, entity_uid
                    ) AS rn
                  FROM combined
                )
                SELECT
                  entity_uid,
                  orphan_uid,
                  article_id,
                  match_probability
                FROM ranked
                WHERE rn = 1;
                """
            )
        )
        ^ pure(unit)
    )


def _prune_stale_override_rows(run_id: str) -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
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
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TEMP TABLE _stale_override_rows AS
                SELECT
                  orphan_id,
                  apply_status
                FROM adjudicated_orphan_apply_report
                WHERE apply_status IN ('skip_orphan_not_found', 'skip_entity_not_found');
                """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM _stale_override_rows;"))
        >> (
            lambda rows: (
                (
                    sql_exec(
                        SQL(
                            f"""--sql
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
                              NULL AS history_id,
                              '{run_id}' AS run_id,
                              o.orphan_id,
                              o.resolution_label AS prior_resolution_label,
                              o.resolved_entity_id AS prior_resolved_entity_id,
                              o.confidence AS prior_confidence,
                              'invalidated' AS new_resolution_label,
                              NULL AS new_resolved_entity_id,
                              NULL AS new_confidence,
                              'apply_skip_stale_reference' AS reason_summary,
                              o.evidence_json,
                              'system_invalidation' AS analyst_mode,
                              NOW() AS changed_at
                            FROM orphan_adjudication_overrides o
                            JOIN _stale_override_rows s
                              ON s.orphan_id = o.orphan_id;
                            """
                        )
                    )
                    ^ sql_exec(
                        SQL(
                            """--sql
                            DELETE FROM orphan_adjudication_overrides
                            WHERE orphan_id IN (SELECT orphan_id FROM _stale_override_rows);
                            """
                        )
                    )
                    ^ put_line(
                        "[J] Invalidated stale adjudication overrides from apply skips: "
                        f"{rows[0]['n']}"
                    )
                )
                if rows[0]["n"] > 0
                else put_line("[J] No stale adjudication overrides to invalidate.")
            )
            ^ pure(unit)
        )
    )


def _build_postadj_entities() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE victim_entity_reps_postadj AS
                SELECT * FROM victim_entity_reps_new;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                WITH matched_orphans AS (
                  SELECT
                    aar.entity_uid,
                    vce.lat,
                    vce.lon,
                    vce.midpoint_day,
                    vce.article_id,
                    vce.victim_count,
                    vce.offender_count
                  FROM _applied_adjudication_rows aar
                  JOIN victims_cached_enh vce
                    ON vce.victim_row_id = CAST(aar.orphan_uid AS VARCHAR)
                ),
                matched_updates AS (
                  SELECT
                    entity_uid,
                    COUNT(*) AS num_orphans,
                    MIN(midpoint_day) AS min_orphan_mid,
                    MAX(midpoint_day) AS max_orphan_mid,
                    STRING_AGG(DISTINCT CAST(article_id AS VARCHAR), ',') AS orphan_article_ids,
                    MAX(victim_count) FILTER (WHERE victim_count IS NOT NULL) AS max_orphan_victim_count,
                    MAX(offender_count) FILTER (WHERE offender_count IS NOT NULL) AS max_orphan_offender_count
                  FROM matched_orphans
                  GROUP BY entity_uid
                )
                UPDATE victim_entity_reps_postadj
                SET
                  cluster_size = cluster_size + COALESCE(mu.num_orphans, 0),
                  min_event_day = CASE
                    WHEN mu.num_orphans IS NOT NULL THEN LEAST(min_event_day, mu.min_orphan_mid)
                    ELSE min_event_day
                  END,
                  max_event_day = CASE
                    WHEN mu.num_orphans IS NOT NULL THEN GREATEST(max_event_day, mu.max_orphan_mid)
                    ELSE max_event_day
                  END,
                  article_ids_csv = CASE
                    WHEN COALESCE(mu.orphan_article_ids, '') = '' THEN article_ids_csv
                    WHEN COALESCE(article_ids_csv, '') = '' THEN mu.orphan_article_ids
                    ELSE article_ids_csv || ',' || mu.orphan_article_ids
                  END,
                  canonical_victim_count = CASE
                    WHEN mu.max_orphan_victim_count IS NOT NULL AND canonical_victim_count IS NOT NULL
                      THEN GREATEST(canonical_victim_count, mu.max_orphan_victim_count)
                    WHEN mu.max_orphan_victim_count IS NOT NULL
                      THEN mu.max_orphan_victim_count
                    ELSE canonical_victim_count
                  END,
                  canonical_offender_count = CASE
                    WHEN mu.max_orphan_offender_count IS NOT NULL AND canonical_offender_count IS NOT NULL
                      THEN GREATEST(canonical_offender_count, mu.max_orphan_offender_count)
                    WHEN mu.max_orphan_offender_count IS NOT NULL
                      THEN mu.max_orphan_offender_count
                    ELSE canonical_offender_count
                  END
                FROM matched_updates mu
                WHERE victim_entity_reps_postadj.victim_entity_id = mu.entity_uid;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                DELETE FROM victim_entity_reps_postadj
                WHERE victim_entity_id IN (
                  SELECT orphan_uid
                  FROM _applied_adjudication_rows
                );
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TEMP TABLE _affected_entities_postadj AS
                SELECT DISTINCT entity_uid AS victim_entity_id
                FROM _applied_adjudication_rows;
                """
            )
        )
        ^ sql_query(
            SQL(
                """--sql
                SELECT COUNT(*) AS n_affected
                FROM _affected_entities_postadj;
                """
            )
        )
        >> (
            lambda rows: (
                (
                    sql_exec(
                        SQL(
                            """--sql
                            CREATE OR REPLACE TEMP TABLE all_members_postadj_temp AS
                            SELECT
                              m.victim_entity_id,
                              m.date_precision,
                              m.incident_date,
                              m.midpoint_day,
                              m.offender_count,
                              m.victim_relationship AS relationship,
                              m.geo_address_norm,
                              m.geo_address_short,
                              m.geo_address_short_2,
                              m.geo_score,
                              m.address_type,
                              m.lat,
                              m.lon
                            FROM victim_entity_members m
                            JOIN _affected_entities_postadj ae
                              ON m.victim_entity_id = ae.victim_entity_id

                            UNION ALL

                            SELECT
                              fom.entity_uid AS victim_entity_id,
                              vce.date_precision,
                              vce.incident_date,
                              vce.midpoint_day,
                              vce.offender_count,
                              vce.victim_relationship AS relationship,
                              vce.geo_address_norm,
                              vce.geo_address_short,
                              vce.geo_address_short_2,
                              vce.geo_score,
                              vce.address_type,
                              vce.lat,
                              vce.lon
                            FROM final_orphan_matches fom
                            JOIN victims_cached_enh vce
                              ON vce.victim_row_id = CAST(fom.orphan_uid AS VARCHAR)
                            JOIN _affected_entities_postadj ae
                              ON fom.entity_uid = ae.victim_entity_id

                            UNION ALL

                            SELECT
                              aar.entity_uid AS victim_entity_id,
                              vce.date_precision,
                              vce.incident_date,
                              vce.midpoint_day,
                              vce.offender_count,
                              vce.victim_relationship AS relationship,
                              vce.geo_address_norm,
                              vce.geo_address_short,
                              vce.geo_address_short_2,
                              vce.geo_score,
                              vce.address_type,
                              vce.lat,
                              vce.lon
                            FROM _applied_adjudication_rows aar
                            JOIN victims_cached_enh vce
                              ON vce.victim_row_id = CAST(aar.orphan_uid AS VARCHAR)
                            JOIN _affected_entities_postadj ae
                              ON aar.entity_uid = ae.victim_entity_id;
                            """
                        )
                    )
                    ^ sql_exec(
                        SQL(
                            """--sql
                            CREATE OR REPLACE TEMP TABLE recomputed_postadj_agg AS
                            WITH agg AS (
                              SELECT
                                victim_entity_id,
                                count_if(date_precision = 'day') AS n_day,
                                count_if(date_precision = 'month') AS n_month,
                                count_if(date_precision = 'year') AS n_year,
                                mode(incident_date) FILTER (
                                  WHERE date_precision = 'day'
                                    AND incident_date IS NOT NULL
                                ) AS mode_day_date,
                                mode(midpoint_day) FILTER (WHERE date_precision = 'month') AS mode_month_mid,
                                mode(midpoint_day) FILTER (WHERE date_precision = 'year') AS mode_year_mid,
                                MAX(offender_count) FILTER (WHERE offender_count IS NOT NULL) AS max_offender_count
                              FROM all_members_postadj_temp
                              GROUP BY victim_entity_id
                            ),
                            relationship_counts AS (
                              SELECT
                                victim_entity_id,
                                relationship,
                                COUNT(*) AS rel_cnt,
                                CASE
                                  WHEN relationship IS NULL OR trim(relationship) = '' THEN 0
                                  WHEN lower(trim(relationship)) IN ('relationship not determined', 'unknown relationship') THEN 1
                                  ELSE 2
                                END AS specificity_rank
                              FROM all_members_postadj_temp
                              GROUP BY victim_entity_id, relationship
                            ),
                            relationship_ranked AS (
                              SELECT
                                victim_entity_id,
                                relationship,
                                specificity_rank,
                                rel_cnt,
                                ROW_NUMBER() OVER (
                                  PARTITION BY victim_entity_id
                                  ORDER BY
                                    specificity_rank DESC,
                                    rel_cnt DESC,
                                    relationship ASC NULLS LAST
                                ) AS rn
                              FROM relationship_counts
                            ),
                            relationship_best AS (
                              SELECT
                                victim_entity_id,
                                CASE
                                  WHEN specificity_rank = 0 THEN NULL
                                  ELSE relationship
                                END AS canonical_relationship
                              FROM relationship_ranked
                              WHERE rn = 1
                            ),
                            location_base AS (
                              SELECT
                                victim_entity_id,
                                geo_address_norm,
                                geo_address_short,
                                geo_address_short_2,
                                geo_score,
                                address_type,
                                lat,
                                lon,
                                COUNT(*) OVER (
                                  PARTITION BY victim_entity_id, geo_address_norm
                                ) AS geo_norm_count,
                                MAX(geo_score) OVER (
                                  PARTITION BY victim_entity_id, geo_address_norm
                                ) AS geo_norm_max_score,
                                MAX(
                                  CASE
                                    WHEN upper(address_type) = 'NAMED_PLACE' THEN 5
                                    WHEN upper(address_type) = 'ADDRESS' THEN 4
                                    WHEN upper(address_type) = 'INTERSECTION' THEN 3
                                    WHEN upper(address_type) = 'BLOCK' THEN 2
                                    ELSE 1
                                  END
                                ) OVER (
                                  PARTITION BY victim_entity_id, geo_address_norm
                                ) AS geo_norm_best_addr_rank,
                                CASE
                                  WHEN upper(address_type) = 'NAMED_PLACE' THEN 5
                                  WHEN upper(address_type) = 'ADDRESS' THEN 4
                                  WHEN upper(address_type) = 'INTERSECTION' THEN 3
                                  WHEN upper(address_type) = 'BLOCK' THEN 2
                                  ELSE 1
                                END AS addr_rank
                              FROM all_members_postadj_temp
                              WHERE geo_address_norm IS NOT NULL
                                AND geo_address_norm <> ''
                                AND geo_address_norm <> 'UNKNOWN'
                            ),
                            location_ranked AS (
                              SELECT
                                *,
                                ROW_NUMBER() OVER (
                                  PARTITION BY victim_entity_id
                                  ORDER BY
                                    geo_norm_max_score DESC NULLS LAST,
                                    geo_norm_count DESC,
                                    geo_norm_best_addr_rank DESC,
                                    addr_rank DESC,
                                    geo_address_norm ASC,
                                    geo_score DESC NULLS LAST,
                                    lat ASC NULLS LAST,
                                    lon ASC NULLS LAST,
                                    geo_address_short ASC NULLS LAST,
                                    geo_address_short_2 ASC NULLS LAST
                                ) AS rn
                              FROM location_base
                            ),
                            location_best AS (
                              SELECT
                                victim_entity_id,
                                geo_address_norm AS canonical_geo_address_norm,
                                geo_address_short AS canonical_geo_address_short,
                                geo_address_short_2 AS canonical_geo_address_short_2,
                                geo_score AS canonical_geo_score,
                                address_type AS canonical_address_type,
                                lat AS canonical_lat,
                                lon AS canonical_lon
                              FROM location_ranked
                              WHERE rn = 1
                            )
                            SELECT
                              agg.victim_entity_id,
                              CASE
                                WHEN n_day > 0 THEN 'day'
                                WHEN n_month > 0 THEN 'month'
                                ELSE 'year'
                              END AS entity_date_precision,
                              CASE
                                WHEN n_day > 0 THEN mode_day_date
                                ELSE NULL
                              END AS incident_date,
                              CAST(
                                CASE
                                  WHEN n_day > 0 THEN date_diff('day', DATE '1970-01-01', mode_day_date)
                                  WHEN n_month > 0 THEN mode_month_mid
                                  ELSE mode_year_mid
                                END AS INTEGER
                              ) AS entity_midpoint_day,
                              max_offender_count,
                              rb.canonical_relationship,
                              lb.canonical_geo_address_norm,
                              lb.canonical_geo_address_short,
                              lb.canonical_geo_address_short_2,
                              lb.canonical_geo_score,
                              lb.canonical_address_type,
                              lb.canonical_lat,
                              lb.canonical_lon
                            FROM agg
                            LEFT JOIN relationship_best rb
                              ON agg.victim_entity_id = rb.victim_entity_id
                            LEFT JOIN location_best lb
                              ON agg.victim_entity_id = lb.victim_entity_id;
                            """
                        )
                    )
                    ^ sql_exec(
                        SQL(
                            """--sql
                            UPDATE victim_entity_reps_postadj
                            SET
                              entity_date_precision = ra.entity_date_precision,
                              incident_date = ra.incident_date,
                              entity_midpoint_day = ra.entity_midpoint_day,
                              canonical_relationship = ra.canonical_relationship,
                              canonical_geo_address_norm = ra.canonical_geo_address_norm,
                              canonical_geo_address_short = ra.canonical_geo_address_short,
                              canonical_geo_address_short_2 = ra.canonical_geo_address_short_2,
                              canonical_geo_score = ra.canonical_geo_score,
                              canonical_address_type = ra.canonical_address_type,
                              canonical_lat = ra.canonical_lat,
                              canonical_lon = ra.canonical_lon,
                              canonical_offender_count = CASE
                                WHEN ra.max_offender_count IS NOT NULL
                                     AND canonical_offender_count IS NOT NULL
                                  THEN GREATEST(canonical_offender_count, ra.max_offender_count)
                                WHEN ra.max_offender_count IS NOT NULL
                                  THEN ra.max_offender_count
                                ELSE canonical_offender_count
                              END
                            FROM recomputed_postadj_agg ra
                            WHERE victim_entity_reps_postadj.victim_entity_id = ra.victim_entity_id;
                            """
                        )
                    )
                    ^ put_line("[J] Recomputed date/location fields for post-adjudication affected entities.")
                )
                if rows[0]["n_affected"] > 0
                else put_line("[J] No post-adjudication entity updates needed.")
            )
        )
        ^ pure(unit)
    )


def _build_postadj_entity_origin() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE victim_entity_reps_postadj_origin AS
                WITH applied_entities AS (
                  SELECT DISTINCT entity_uid
                  FROM _applied_adjudication_rows
                ),
                machine_entities AS (
                  SELECT DISTINCT entity_uid
                  FROM final_orphan_matches
                )
                SELECT
                  vep.victim_entity_id,
                  CASE
                    WHEN vep.victim_entity_id IN (
                      SELECT victim_entity_id
                      FROM victim_entity_reps
                    ) THEN
                      CASE
                        WHEN vep.victim_entity_id IN (
                          SELECT entity_uid FROM applied_entities
                        ) THEN 'adjudication_matched_entity'
                        WHEN vep.victim_entity_id IN (
                          SELECT entity_uid FROM machine_entities
                        ) THEN 'machine_matched_entity'
                        ELSE 'baseline_entity'
                      END
                    ELSE 'singleton_orphan_unresolved'
                  END AS entity_origin_type
                FROM victim_entity_reps_postadj vep;
                """
            )
        )
        ^ pure(unit)
    )


def _build_orphan_matches_postadj_current() -> Run[Unit]:
    orphan_matches_postadj_select = SQL(
        """--sql
        WITH baseline_unmatched_orphans AS (
          SELECT o.*
          FROM orphan_link_input o
          LEFT JOIN final_orphan_matches fm0
            ON fm0.orphan_uid = o.unique_id
          WHERE fm0.orphan_uid IS NULL
        ),
        adjudication_lookup AS (
          SELECT
            orphan_id,
            resolution_label,
            resolved_entity_id,
            confidence,
            reason_summary
          FROM orphan_adjudication_overrides
        ),
        orphan_choice_display AS (
          SELECT
            o.*,
            fm.entity_uid,
            al.resolution_label AS adjudication_label,
            al.resolved_entity_id AS adjudicated_entity_id,
            al.confidence AS adjudication_confidence,
            al.reason_summary AS adjudication_reason_summary
          FROM baseline_unmatched_orphans o
          LEFT JOIN final_orphan_matches_postadj fm
            ON o.unique_id = fm.orphan_uid
          LEFT JOIN adjudication_lookup al
            ON al.orphan_id = o.unique_id
        ),
        display_matched_entities AS (
          SELECT DISTINCT entity_uid
          FROM orphan_choice_display
          WHERE entity_uid IS NOT NULL
        ),
        entity_with_match AS (
          SELECT
            e.*,
            CASE WHEN e.unique_id IN (
              SELECT entity_uid
              FROM display_matched_entities
            )
              THEN CONCAT('match_', e.unique_id)
              ELSE CONCAT('entity_', e.unique_id)
            END AS match_id,
            CASE WHEN e.unique_id IN (
              SELECT entity_uid
              FROM display_matched_entities
            )
              THEN 'adjudication_matched_entity'
              ELSE 'untouched_entity'
            END AS display_category,
            CASE WHEN e.unique_id IN (
              SELECT entity_uid
              FROM display_matched_entities
            )
              THEN 2
              ELSE 0
            END AS display_band_key,
            CASE WHEN e.unique_id IN (
              SELECT entity_uid
              FROM display_matched_entities
            )
              THEN 'adjudication_applied'
              ELSE 'none'
            END AS adjudication_flag
          FROM entity_link_input e
        ),
        combined AS (
          SELECT
            'entity' AS rec_type,
            e.unique_id AS uid,
            e.match_id,
            e.midpoint_day AS group_midpoint_day,
            e.midpoint_day,
            e.incident_date,
            e.date_precision,
            CAST(NULL AS BIGINT) AS article_id,
            e.article_ids_csv,
            e.city_id,
            e.year,
            e.month,
            e.geo_address_norm,
            e.geo_address_short,
            e.geo_address_short_2,
            e.geo_score,
            e.address_type,
            e.lat,
            e.lon,
            e.victim_age,
            e.victim_sex,
            e.victim_race,
            e.victim_ethnicity,
            e.relationship,
            e.victim_fullname_norm,
            e.weapon,
            e.circumstance,
            e.offender_forename_norm,
            e.offender_surname_norm,
            e.victim_count,
            e.offender_count,
            CAST(NULL AS DOUBLE) AS confidence,
            e.display_category,
            e.display_band_key,
            CAST(NULL AS VARCHAR) AS adjudication_label,
            e.adjudication_flag,
            CAST(NULL AS VARCHAR) AS reason_summary
          FROM entity_with_match e

          UNION ALL

          SELECT
            'orphan' AS rec_type,
            o.unique_id AS uid,
            CASE
              WHEN o.entity_uid IS NOT NULL THEN CONCAT('match_', o.entity_uid)
              ELSE CONCAT('orphan_', o.unique_id)
            END AS match_id,
            COALESCE(e.midpoint_day, o.midpoint_day) AS group_midpoint_day,
            o.midpoint_day,
            o.incident_date,
            o.date_precision,
            o.article_id,
            o.article_ids_csv,
            o.city_id,
            o.year,
            o.month,
            o.geo_address_norm,
            o.geo_address_short,
            o.geo_address_short_2,
            o.geo_score,
            o.address_type,
            o.lat,
            o.lon,
            o.victim_age,
            o.victim_sex,
            o.victim_race,
            o.victim_ethnicity,
            o.relationship,
            o.victim_fullname_norm,
            o.weapon,
            o.circumstance,
            o.offender_forename_norm,
            o.offender_surname_norm,
            o.victim_count,
            o.offender_count,
            o.adjudication_confidence AS confidence,
            CASE
              WHEN o.entity_uid IS NOT NULL THEN 'adjudication_matched_orphan'
              WHEN o.adjudication_label = 'unlikely' THEN 'left_behind_unlikely'
              ELSE 'left_behind_other'
            END AS display_category,
            CASE
              WHEN o.entity_uid IS NOT NULL THEN 2
              WHEN o.adjudication_label = 'unlikely' THEN 3
              ELSE 1
            END AS display_band_key,
            o.adjudication_label,
            CASE
              WHEN o.entity_uid IS NOT NULL THEN 'adjudication_applied'
              WHEN o.adjudication_label = 'unlikely' THEN 'adjudication_unlikely'
              ELSE 'none'
            END AS adjudication_flag,
            o.adjudication_reason_summary AS reason_summary
          FROM orphan_choice_display o
          LEFT JOIN entity_link_input e
            ON e.unique_id = o.entity_uid
        )
        SELECT
          rec_type,
          match_id,
          display_band_key AS band_key,
          midpoint_day,
          uid,
          article_id,
          city_id,
          year,
          month,
          date_precision,
          incident_date,
          geo_address_norm,
          geo_address_short,
          geo_address_short_2,
          geo_score,
          address_type,
          lat,
          lon,
          victim_age,
          victim_sex,
          victim_race,
          victim_ethnicity,
          relationship,
          victim_fullname_norm,
          weapon,
          circumstance,
          offender_forename_norm,
          offender_surname_norm,
          victim_count,
          offender_count,
          confidence,
          article_ids_csv,
          display_category,
          display_band_key,
          adjudication_label,
          adjudication_flag,
          reason_summary
        FROM combined
        ORDER BY
          group_midpoint_day NULLS LAST,
          match_id,
          CASE rec_type WHEN 'entity' THEN 0 ELSE 1 END,
          uid
        """
    )

    return (
        sql_exec(
            SQL(
                f"""--sql
                CREATE OR REPLACE TABLE orphan_matches_postadj_current AS
                {orphan_matches_postadj_select}
                """
            )
        )
        ^ sql_export(
            orphan_matches_postadj_select,
            "orphan_matches_postadj.xlsx",
            "Matches",
            band_by_group_col="display_band_key",
            band_wrap=4,
        )
        ^ put_line("[J] Wrote orphan_matches_postadj.xlsx.")
        ^ pure(unit)
    )


def _export_final_victim_entities_postadj() -> Run[Unit]:
    final_victim_entities_postadj_select = SQL(
        """--sql
        WITH applied_entities AS (
          SELECT DISTINCT entity_uid
          FROM _applied_adjudication_rows
        ),
        machine_entities AS (
          SELECT DISTINCT entity_uid
          FROM final_orphan_matches
        ),
        categorized_entities AS (
          SELECT
            vep.*,
            CASE
              WHEN vep.victim_entity_id IN (SELECT victim_entity_id FROM victim_entity_reps) THEN
                CASE
                  WHEN vep.victim_entity_id IN (SELECT entity_uid FROM applied_entities) THEN 'adjudication_matched_entity'
                  WHEN vep.victim_entity_id IN (SELECT entity_uid FROM machine_entities) THEN 'machine_matched_entity'
                  ELSE 'baseline_entity'
                END
              ELSE 'remaining_singleton_orphan'
            END AS display_category,
            CASE
              WHEN vep.victim_entity_id IN (SELECT victim_entity_id FROM victim_entity_reps) THEN
                CASE
                  WHEN vep.victim_entity_id IN (SELECT entity_uid FROM applied_entities) THEN 2
                  WHEN vep.victim_entity_id IN (SELECT entity_uid FROM machine_entities) THEN 1
                  ELSE 0
                END
              ELSE 3
            END AS display_band_key
          FROM victim_entity_reps_postadj vep
        )
        SELECT
          'entity' AS rec_type,
          CONCAT('entity_', c.victim_entity_id) AS match_id,
          c.display_band_key AS band_key,
          c.entity_midpoint_day AS midpoint_day,
          c.victim_entity_id AS uid,
          CAST(NULL AS BIGINT) AS article_id,
          c.city_id,
          EXTRACT(
            YEAR FROM COALESCE(
              c.incident_date,
              date_add(DATE '1970-01-01', INTERVAL (CAST(c.entity_midpoint_day AS INTEGER)) DAY)
            )
          ) AS year,
          EXTRACT(
            MONTH FROM COALESCE(
              c.incident_date,
              date_add(DATE '1970-01-01', INTERVAL (CAST(c.entity_midpoint_day AS INTEGER)) DAY)
            )
          ) AS month,
          c.entity_date_precision AS date_precision,
          c.incident_date,
          c.canonical_geo_address_norm AS geo_address_norm,
          c.canonical_geo_address_short AS geo_address_short,
          c.canonical_geo_address_short_2 AS geo_address_short_2,
          c.canonical_geo_score AS geo_score,
          c.canonical_address_type AS address_type,
          c.canonical_lat AS lat,
          c.canonical_lon AS lon,
          c.canonical_age AS victim_age,
          c.canonical_sex AS victim_sex,
          c.canonical_race AS victim_race,
          c.canonical_ethnicity AS victim_ethnicity,
          c.canonical_relationship AS relationship,
          CAST(c.canonical_fullname AS VARCHAR) AS victim_fullname_norm,
          c.mode_weapon AS weapon,
          c.mode_circumstance AS circumstance,
          CAST(NULL AS VARCHAR) AS offender_forename_norm,
          CAST(NULL AS VARCHAR) AS offender_surname_norm,
          c.canonical_victim_count AS victim_count,
          c.canonical_offender_count AS offender_count,
          CAST(NULL AS DOUBLE) AS confidence,
          c.article_ids_csv,
          c.display_category,
          c.display_band_key,
          CAST(NULL AS VARCHAR) AS adjudication_label,
          CASE
            WHEN c.display_category = 'adjudication_matched_entity' THEN 'adjudication_applied'
            ELSE 'none'
          END AS adjudication_flag,
          CAST(NULL AS VARCHAR) AS reason_summary
        FROM categorized_entities c
        ORDER BY c.entity_midpoint_day
        """
    )

    return (
        sql_export(
            final_victim_entities_postadj_select,
            "final_victim_entities_postadj.xlsx",
            "Entities",
            band_by_group_col="display_band_key",
            band_wrap=4,
        )
        ^ put_line("[J] Wrote final_victim_entities_postadj.xlsx.")
        ^ pure(unit)
    )


def _append_apply_history(run_id: str) -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE TABLE IF NOT EXISTS orphan_adjudication_apply_history (
                  run_id VARCHAR,
                  run_at TIMESTAMP,
                  candidate_count BIGINT,
                  applied_count BIGINT,
                  skip_orphan_not_found BIGINT,
                  skip_entity_not_found BIGINT,
                  skip_already_machine_matched BIGINT,
                  skip_duplicate_adjudication_orphan BIGINT,
                  skip_conflict_entity_mismatch BIGINT
                );
                """
            )
        )
        ^ sql_exec(
            SQL(
                f"""--sql
                INSERT INTO orphan_adjudication_apply_history
                SELECT
                  '{run_id}' AS run_id,
                  NOW() AS run_at,
                  COUNT(*) AS candidate_count,
                  COUNT(*) FILTER (WHERE apply_status = 'applied') AS applied_count,
                  COUNT(*) FILTER (WHERE apply_status = 'skip_orphan_not_found') AS skip_orphan_not_found,
                  COUNT(*) FILTER (WHERE apply_status = 'skip_entity_not_found') AS skip_entity_not_found,
                  COUNT(*) FILTER (WHERE apply_status = 'skip_already_machine_matched') AS skip_already_machine_matched,
                  COUNT(*) FILTER (WHERE apply_status = 'skip_duplicate_adjudication_orphan') AS skip_duplicate_adjudication_orphan,
                  COUNT(*) FILTER (WHERE apply_status = 'skip_conflict_entity_mismatch') AS skip_conflict_entity_mismatch
                FROM adjudicated_orphan_apply_report;
                """
            )
        )
        ^ pure(unit)
    )


def _print_summary() -> Run[Unit]:
    return (
        sql_query(
            SQL(
                """--sql
                SELECT
                  COUNT(*) AS candidate_count,
                  COUNT(*) FILTER (WHERE apply_status = 'applied') AS applied_count,
                  COUNT(*) FILTER (WHERE apply_status <> 'applied') AS skipped_count
                FROM adjudicated_orphan_apply_report;
                """
            )
        )
        >> (
            lambda rows: put_line(
                "[J] Post-adjudication apply summary: "
                f"candidates={rows[0]['candidate_count']}, "
                f"applied={rows[0]['applied_count']}, "
                f"skipped={rows[0]['skipped_count']}"
            )
        )
        ^ pure(unit)
    )


def _run_apply() -> Run[NextStep]:
    run_id = datetime.now(timezone.utc).strftime("postadj_%Y%m%dT%H%M%SZ")
    return (
        put_line(f"[J] Applying orphan adjudications (run_id={run_id})...")
        ^ _build_adjudication_candidates()
        ^ _prune_stale_override_rows(run_id)
        ^ _build_postadj_matches()
        ^ _build_postadj_entities()
        ^ _build_postadj_entity_origin()
        ^ _build_orphan_matches_postadj_current()
        ^ _export_final_victim_entities_postadj()
        ^ _append_apply_history(run_id)
        ^ _print_summary()
        ^ pure(NextStep.CONTINUE)
    )


def apply_orphan_adjudications() -> Run[NextStep]:
    """
    Entry point for post-adjudication orphan integration.
    """
    return with_duckdb(_run_apply())
