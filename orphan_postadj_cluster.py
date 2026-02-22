"""
Cluster unresolved post-adjudication singleton orphan entities into unnamed entities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from blocking import ORPHAN_DETERMINISTIC_BLOCKS, ORPHAN_VICTIM_BLOCKS
from menuprompts import NextStep
from splink_types import SplinkType
from pymonad import (
    BlockedPairsTableName,
    ClustersTableName,
    DoNotLinkTableName,
    ErrorPayload,
    PairsTableName,
    PredictionInputTableName,
    Run,
    SQL,
    Unit,
    pure,
    put_line,
    splink_dedupe_job,
    sql_exec,
    sql_export,
    sql_query,
    throw,
    unit,
    with_duckdb,
    file_exists,
)


def _dedupe_model_path() -> Path:
    key_str = str(SplinkType.DEDUP).replace("/", "_")
    return Path("splink_models") / f"splink_model_{key_str}.json"


def _load_dedupe_model_settings() -> Run[dict]:
    model_path = _dedupe_model_path()

    def _load(_: bool) -> Run[dict]:
        try:
            with open(model_path, "r", encoding="utf-8") as handle:
                settings = json.load(handle)
        except Exception as exc:  # pylint: disable=W0718
            return throw(
                ErrorPayload(f"Failed to load dedupe model settings: {exc}")
            )
        if not isinstance(settings, dict) or "comparisons" not in settings:
            return throw(
                ErrorPayload(
                    "Dedupe model settings missing comparisons. Re-run DEDUP first."
                )
            )
        return pure(settings)

    return file_exists(str(model_path)) >> (
        lambda exists: _load(exists)
        if exists
        else throw(
            ErrorPayload(
                "Dedupe Splink model not found. Run DEDUP before post-adj orphan clustering."
            )
        )
    )


def _settings_for_postadj_orphan_reuse(dedupe_settings: dict) -> dict:
    settings = dict(dedupe_settings)
    settings.update(
        {
            "link_type": "dedupe_only",
            "unique_id_column_name": "unique_id",
            "blocking_rules_to_generate_predictions": ORPHAN_VICTIM_BLOCKS,
        }
    )
    return settings


def _assert_postadj_inputs() -> Run[Unit]:
    return (
        sql_query(
            SQL(
                """--sql
                WITH checks AS (
                  SELECT
                    'victim_entity_reps_postadj' AS name,
                    COUNT(*) AS n_cols
                  FROM pragma_table_info('victim_entity_reps_postadj')
                  UNION ALL
                  SELECT
                    'victim_entity_reps_postadj_origin' AS name,
                    COUNT(*) AS n_cols
                  FROM pragma_table_info('victim_entity_reps_postadj_origin')
                )
                SELECT name
                FROM checks
                WHERE n_cols = 0
                ORDER BY name;
                """
            )
        )
        >> (
            lambda rows: (
                throw(
                    ErrorPayload(
                        "[O] Missing required post-adjudication tables: "
                        + ", ".join(str(row["name"]) for row in rows)
                        + ". Run [J] Apply orphan adjudication first."
                    )
                )
                if rows
                else pure(unit)
            )
        )
    )


def _build_postadj_orphan_cluster_input() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE postadj_orphan_cluster_input AS
                SELECT
                  vep.victim_entity_id AS unique_id,
                  'postadj_orphan' AS source_dataset,
                  vep.city_id,
                  vep.entity_midpoint_day AS midpoint_day,
                  vep.incident_date,
                  vep.entity_date_precision AS date_precision,
                  EXTRACT(
                    YEAR FROM COALESCE(
                      vep.incident_date,
                      date_add(DATE '1970-01-01', INTERVAL (CAST(vep.entity_midpoint_day AS INTEGER)) DAY)
                    )
                  ) AS year,
                  EXTRACT(
                    MONTH FROM COALESCE(
                      vep.incident_date,
                      date_add(DATE '1970-01-01', INTERVAL (CAST(vep.entity_midpoint_day AS INTEGER)) DAY)
                    )
                  ) AS month,
                  vep.canonical_geo_address_norm AS geo_address_norm,
                  vep.canonical_geo_address_short AS geo_address_short,
                  vep.canonical_geo_address_short_2 AS geo_address_short_2,
                  vep.canonical_geo_score AS geo_score,
                  vep.canonical_address_type AS address_type,
                  vep.canonical_lat AS lat,
                  vep.canonical_lon AS lon,
                  vep.canonical_age AS victim_age,
                  vep.canonical_victim_count AS victim_count,
                  vep.canonical_sex AS victim_sex,
                  vep.canonical_race AS victim_race,
                  vep.canonical_ethnicity AS victim_ethnicity,
                  vep.canonical_relationship AS relationship,
                  CAST(NULL AS VARCHAR) AS victim_fullname_norm,
                  CAST(NULL AS VARCHAR) AS victim_fullname_concat,
                  CAST(NULL AS VARCHAR) AS victim_forename_norm,
                  CAST(NULL AS VARCHAR) AS victim_surname_norm,
                  vep.mode_weapon AS weapon,
                  vep.mode_circumstance AS circumstance,
                  vep.offender_forename AS offender_forename_norm,
                  vep.offender_surname AS offender_surname_norm,
                  vep.offender_fullname AS offender_fullname_concat,
                  vep.canonical_offender_age AS offender_age,
                  vep.canonical_offender_sex AS offender_sex,
                  vep.canonical_offender_race AS offender_race,
                  vep.canonical_offender_ethnicity AS offender_ethnicity,
                  vep.canonical_offender_count AS offender_count,
                  vep.summary_vec,
                  vep.article_ids_csv AS source_article_ids_csv,
                  COALESCE(NULLIF(split_part(vep.article_ids_csv, ',', 1), ''), vep.victim_entity_id) AS exclusion_id,
                  []::VARCHAR[] AS exclusion_ids,
                  vep.cluster_size AS source_cluster_size
                FROM victim_entity_reps_postadj vep
                JOIN victim_entity_reps_postadj_origin o
                  ON o.victim_entity_id = vep.victim_entity_id
                WHERE o.entity_origin_type = 'singleton_orphan_unresolved';
                """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM postadj_orphan_cluster_input;"))
        >> (lambda rows: put_line(f"[O] postadj_orphan_cluster_input rows: {rows[0]['n']}"))
        ^ pure(unit)
    )


def _cluster_postadj_singletons() -> Run[Unit]:
    def _create_empty_cluster_outputs() -> Run[Unit]:
        return (
            sql_exec(
                SQL(
                    """--sql
                    CREATE OR REPLACE TABLE postadj_orphan_cluster_pairs (
                      unique_id_l VARCHAR,
                      unique_id_r VARCHAR,
                      match_probability DOUBLE
                    );
                    """
                )
            )
            ^ sql_exec(
                SQL(
                    """--sql
                    CREATE OR REPLACE TABLE postadj_orphan_clusters (
                      cluster_id VARCHAR,
                      unique_id VARCHAR
                    );
                    """
                )
            )
            ^ sql_exec(
                SQL(
                    """--sql
                    CREATE OR REPLACE TABLE postadj_orphan_cluster_exclusion (
                      id_l VARCHAR,
                      id_r VARCHAR
                    );
                    """
                )
            )
            ^ sql_exec(
                SQL(
                    """--sql
                    CREATE OR REPLACE TABLE postadj_orphan_cluster_blocked_edges (
                      id_l VARCHAR,
                      id_r VARCHAR,
                      match_probability DOUBLE,
                      shared_exclusion_ids VARCHAR
                    );
                    """
                )
            )
            ^ put_line("[O] No unresolved singleton orphans to cluster; wrote empty clustering outputs.")
            ^ pure(unit)
        )

    def _run(settings: dict) -> Run[Unit]:
        return splink_dedupe_job(
            input_table=PredictionInputTableName("postadj_orphan_cluster_input"),
            settings=settings,
            predict_threshold=0.25,
            cluster_threshold=0.0,
            pairs_out=PairsTableName("postadj_orphan_cluster_pairs"),
            clusters_out=ClustersTableName("postadj_orphan_clusters"),
            deterministic_rules=ORPHAN_DETERMINISTIC_BLOCKS,
            deterministic_recall=0.1,
            train_first=False,
            skip_u_estimation=True,
            training_blocking_rules=[],
            visualize=False,
            splink_key=SplinkType.POSTADJ_ORPHAN_CLUSTER,
            do_not_link_table=DoNotLinkTableName("postadj_orphan_cluster_exclusion"),
            blocked_pairs_out=BlockedPairsTableName("postadj_orphan_cluster_blocked_edges"),
            capture_blocked_edges=False,
        ) >> (
            lambda outnames: (
                put_line(f"[O] Wrote {outnames[1]} and {outnames[2]} in DuckDB.")
                ^ sql_exec(
                    SQL(
                        """--sql
                        CREATE TABLE IF NOT EXISTS postadj_orphan_cluster_exclusion (
                          id_l VARCHAR,
                          id_r VARCHAR
                        );
                        """
                    )
                )
                ^ sql_exec(
                    SQL(
                        """--sql
                        CREATE TABLE IF NOT EXISTS postadj_orphan_cluster_blocked_edges (
                          id_l VARCHAR,
                          id_r VARCHAR,
                          match_probability DOUBLE,
                          shared_exclusion_ids VARCHAR
                        );
                        """
                    )
                )
            )
        ) ^ pure(unit)

    return sql_query(SQL("SELECT COUNT(*) AS n FROM postadj_orphan_cluster_input;")) >> (
        lambda rows: (
            _create_empty_cluster_outputs()
            if rows[0]["n"] == 0
            else _load_dedupe_model_settings() >> (
                lambda settings: _run(_settings_for_postadj_orphan_reuse(settings))
            )
        )
    )


def _build_postadj_orphancluster_canonical() -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE postadj_orphan_cluster_members AS
                SELECT
                  c.cluster_id,
                  i.*
                FROM postadj_orphan_clusters c
                JOIN postadj_orphan_cluster_input i
                  ON i.unique_id = c.unique_id;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE postadj_orphan_cluster_sizes AS
                SELECT
                  cluster_id,
                  COUNT(*) AS member_count
                FROM postadj_orphan_cluster_members
                GROUP BY cluster_id;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TEMP TABLE _postadj_absorbed_singletons AS
                SELECT m.unique_id
                FROM postadj_orphan_cluster_members m
                JOIN postadj_orphan_cluster_sizes s
                  ON s.cluster_id = m.cluster_id
                WHERE s.member_count >= 2;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE postadj_orphan_cluster_entities AS
                WITH members AS (
                  SELECT m.*
                  FROM postadj_orphan_cluster_members m
                  JOIN postadj_orphan_cluster_sizes s
                    ON s.cluster_id = m.cluster_id
                  WHERE s.member_count >= 2
                ),
                date_agg AS (
                  SELECT
                    cluster_id,
                    count_if(date_precision = 'day') AS n_day,
                    count_if(date_precision = 'month') AS n_month,
                    mode(incident_date) FILTER (
                      WHERE date_precision = 'day'
                        AND incident_date IS NOT NULL
                    ) AS mode_day_date,
                    mode(midpoint_day) FILTER (WHERE date_precision = 'month') AS mode_month_mid,
                    mode(midpoint_day) FILTER (WHERE date_precision = 'year') AS mode_year_mid
                  FROM members
                  GROUP BY cluster_id
                ),
                location_base AS (
                  SELECT
                    cluster_id,
                    geo_address_norm,
                    geo_address_short,
                    geo_address_short_2,
                    geo_score,
                    address_type,
                    lat,
                    lon,
                    COUNT(*) OVER (
                      PARTITION BY cluster_id, geo_address_norm
                    ) AS geo_norm_count,
                    MAX(geo_score) OVER (
                      PARTITION BY cluster_id, geo_address_norm
                    ) AS geo_norm_max_score,
                    CASE
                      WHEN upper(address_type) = 'NAMED_PLACE' THEN 5
                      WHEN upper(address_type) = 'ADDRESS' THEN 4
                      WHEN upper(address_type) = 'INTERSECTION' THEN 3
                      WHEN upper(address_type) = 'BLOCK' THEN 2
                      ELSE 1
                    END AS addr_rank
                  FROM members
                  WHERE geo_address_norm IS NOT NULL
                    AND geo_address_norm <> ''
                    AND geo_address_norm <> 'UNKNOWN'
                ),
                location_ranked AS (
                  SELECT
                    *,
                    ROW_NUMBER() OVER (
                      PARTITION BY cluster_id
                      ORDER BY
                        geo_norm_max_score DESC NULLS LAST,
                        geo_norm_count DESC,
                        addr_rank DESC,
                        geo_address_norm,
                        geo_score DESC NULLS LAST
                    ) AS rn
                  FROM location_base
                ),
                location_best AS (
                  SELECT
                    cluster_id,
                    geo_address_norm AS canonical_geo_address_norm,
                    geo_address_short AS canonical_geo_address_short,
                    geo_address_short_2 AS canonical_geo_address_short_2,
                    geo_score AS canonical_geo_score,
                    address_type AS canonical_address_type,
                    lat AS canonical_lat,
                    lon AS canonical_lon
                  FROM location_ranked
                  WHERE rn = 1
                ),
                age_agg AS (
                  SELECT
                    cluster_id,
                    AVG(victim_age) FILTER (WHERE victim_age IS NOT NULL) AS avg_age,
                    MIN(victim_age) FILTER (WHERE victim_age IS NOT NULL) AS min_age,
                    MAX(victim_age) FILTER (WHERE victim_age IS NOT NULL) AS max_age
                  FROM members
                  GROUP BY cluster_id
                ),
                relationship_counts AS (
                  SELECT
                    cluster_id,
                    relationship,
                    COUNT(*) AS rel_cnt,
                    CASE
                      WHEN relationship IS NULL OR trim(relationship) = '' THEN 0
                      WHEN lower(trim(relationship)) IN ('relationship not determined', 'unknown relationship') THEN 1
                      ELSE 2
                    END AS specificity_rank
                  FROM members
                  GROUP BY cluster_id, relationship
                ),
                relationship_ranked AS (
                  SELECT
                    cluster_id,
                    relationship,
                    specificity_rank,
                    rel_cnt,
                    ROW_NUMBER() OVER (
                      PARTITION BY cluster_id
                      ORDER BY
                        specificity_rank DESC,
                        rel_cnt DESC,
                        relationship ASC NULLS LAST
                    ) AS rn
                  FROM relationship_counts
                ),
                relationship_best AS (
                  SELECT
                    cluster_id,
                    CASE
                      WHEN specificity_rank = 0 THEN NULL
                      ELSE relationship
                    END AS canonical_relationship
                  FROM relationship_ranked
                  WHERE rn = 1
                )
                SELECT
                  MIN(unique_id) AS victim_entity_id,
                  mode(city_id) FILTER (WHERE city_id IS NOT NULL) AS city_id,
                  MIN(midpoint_day) AS min_event_day,
                  MAX(midpoint_day) AS max_event_day,
                  CAST(NULL AS DOUBLE[384]) AS summary_vec,
                  CASE
                    WHEN da.n_day > 0 THEN 'day'
                    WHEN da.n_month > 0 THEN 'month'
                    ELSE 'year'
                  END AS entity_date_precision,
                  CASE
                    WHEN da.n_day > 0 THEN da.mode_day_date
                    ELSE NULL
                  END AS incident_date,
                  CAST(
                    CASE
                      WHEN da.n_day > 0 THEN date_diff('day', DATE '1970-01-01', da.mode_day_date)
                      WHEN da.n_month > 0 THEN da.mode_month_mid
                      ELSE da.mode_year_mid
                    END AS INTEGER
                  ) AS entity_midpoint_day,
                  CAST(NULL AS VARCHAR) AS canonical_fullname,
                  mode(victim_sex) FILTER (WHERE victim_sex IS NOT NULL) AS canonical_sex,
                  mode(victim_race) FILTER (WHERE victim_race IS NOT NULL) AS canonical_race,
                  mode(victim_ethnicity) FILTER (WHERE victim_ethnicity IS NOT NULL) AS canonical_ethnicity,
                  rb.canonical_relationship,
                  mode(offender_age) FILTER (WHERE offender_age IS NOT NULL) AS canonical_offender_age,
                  mode(offender_sex) FILTER (WHERE offender_sex IS NOT NULL) AS canonical_offender_sex,
                  mode(offender_race) FILTER (WHERE offender_race IS NOT NULL) AS canonical_offender_race,
                  mode(offender_ethnicity) FILTER (WHERE offender_ethnicity IS NOT NULL) AS canonical_offender_ethnicity,
                  MAX(offender_count) FILTER (WHERE offender_count IS NOT NULL) AS canonical_offender_count,
                  lb.canonical_geo_address_norm,
                  lb.canonical_geo_address_short,
                  lb.canonical_geo_address_short_2,
                  lb.canonical_geo_score,
                  lb.canonical_address_type,
                  lb.canonical_lat,
                  lb.canonical_lon,
                  mode(victim_age) FILTER (WHERE victim_age IS NOT NULL) AS canonical_age,
                  MAX(victim_count) FILTER (WHERE victim_count IS NOT NULL) AS canonical_victim_count,
                  aa.avg_age,
                  aa.min_age,
                  aa.max_age,
                  mode(weapon) FILTER (WHERE weapon IS NOT NULL) AS mode_weapon,
                  mode(circumstance) FILTER (
                    WHERE circumstance IS NOT NULL
                      AND lower(circumstance) <> 'undetermined'
                  ) AS mode_circumstance,
                  mode(offender_fullname_concat) FILTER (
                    WHERE offender_fullname_concat IS NOT NULL
                  ) AS offender_fullname,
                  mode(offender_forename_norm) FILTER (
                    WHERE offender_forename_norm IS NOT NULL
                  ) AS offender_forename,
                  mode(offender_surname_norm) FILTER (
                    WHERE offender_surname_norm IS NOT NULL
                  ) AS offender_surname,
                  COUNT(*) AS cluster_size,
                  string_agg(DISTINCT source_article_ids_csv, ',') AS article_ids_csv
                FROM members m
                JOIN date_agg da
                  ON da.cluster_id = m.cluster_id
                LEFT JOIN location_best lb
                  ON lb.cluster_id = m.cluster_id
                LEFT JOIN age_agg aa
                  ON aa.cluster_id = m.cluster_id
                LEFT JOIN relationship_best rb
                  ON rb.cluster_id = m.cluster_id
                GROUP BY m.cluster_id, da.n_day, da.n_month, da.mode_day_date, da.mode_month_mid, da.mode_year_mid,
                         lb.canonical_geo_address_norm, lb.canonical_geo_address_short, lb.canonical_geo_address_short_2,
                         lb.canonical_geo_score, lb.canonical_address_type, lb.canonical_lat, lb.canonical_lon, rb.canonical_relationship,
                         aa.avg_age, aa.min_age, aa.max_age;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE victim_entity_reps_postadj_orphancluster AS
                SELECT
                  victim_entity_id,
                  city_id,
                  min_event_day,
                  max_event_day,
                  summary_vec,
                  entity_date_precision,
                  incident_date,
                  entity_midpoint_day,
                  canonical_fullname,
                  canonical_sex,
                  canonical_race,
                  canonical_ethnicity,
                  canonical_relationship,
                  canonical_offender_age,
                  canonical_offender_sex,
                  canonical_offender_race,
                  canonical_offender_ethnicity,
                  canonical_offender_count,
                  canonical_geo_address_norm,
                  canonical_geo_address_short,
                  canonical_geo_address_short_2,
                  canonical_geo_score,
                  canonical_address_type,
                  canonical_lat,
                  canonical_lon,
                  canonical_age,
                  canonical_victim_count,
                  avg_age,
                  min_age,
                  max_age,
                  mode_weapon,
                  mode_circumstance,
                  offender_fullname,
                  offender_forename,
                  offender_surname,
                  cluster_size,
                  article_ids_csv
                FROM victim_entity_reps_postadj
                WHERE victim_entity_id NOT IN (SELECT unique_id FROM _postadj_absorbed_singletons)

                UNION ALL

                SELECT
                  victim_entity_id,
                  city_id,
                  min_event_day,
                  max_event_day,
                  summary_vec,
                  entity_date_precision,
                  incident_date,
                  entity_midpoint_day,
                  canonical_fullname,
                  canonical_sex,
                  canonical_race,
                  canonical_ethnicity,
                  canonical_relationship,
                  canonical_offender_age,
                  canonical_offender_sex,
                  canonical_offender_race,
                  canonical_offender_ethnicity,
                  canonical_offender_count,
                  canonical_geo_address_norm,
                  canonical_geo_address_short,
                  canonical_geo_address_short_2,
                  canonical_geo_score,
                  canonical_address_type,
                  canonical_lat,
                  canonical_lon,
                  canonical_age,
                  canonical_victim_count,
                  avg_age,
                  min_age,
                  max_age,
                  mode_weapon,
                  mode_circumstance,
                  offender_fullname,
                  offender_forename,
                  offender_surname,
                  cluster_size,
                  article_ids_csv
                FROM postadj_orphan_cluster_entities;
                """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE victim_entity_reps_postadj_orphancluster_origin AS
                SELECT
                  o.victim_entity_id,
                  o.entity_origin_type
                FROM victim_entity_reps_postadj_origin o
                WHERE o.victim_entity_id NOT IN (SELECT unique_id FROM _postadj_absorbed_singletons)

                UNION ALL

                SELECT
                  e.victim_entity_id,
                  'clustered_singleton_orphans' AS entity_origin_type
                FROM postadj_orphan_cluster_entities e;
                """
            )
        )
        ^ pure(unit)
    )


def _export_postadj_orphan_clusters_excel() -> Run[Unit]:
    review_select = SQL(
        """--sql
        WITH multi_clusters AS (
          SELECT cluster_id
          FROM postadj_orphan_cluster_sizes
          WHERE member_count >= 2
        ),
        adjudication_lookup AS (
          SELECT
            orphan_id,
            reason_summary
          FROM orphan_adjudication_overrides
        ),
        named_entities AS (
          SELECT
            'named_entity' AS rec_type,
            v.victim_entity_id AS uid,
            v.victim_entity_id AS entity_uid,
            v.entity_midpoint_day AS group_midpoint_day,
            v.entity_midpoint_day AS midpoint_day,
            v.entity_date_precision AS date_precision,
            v.incident_date,
            v.canonical_fullname,
            v.canonical_sex,
            v.canonical_race,
            v.canonical_ethnicity,
            v.canonical_relationship AS relationship,
            v.canonical_age,
            v.canonical_victim_count AS victim_count,
            v.canonical_offender_count AS offender_count,
            v.mode_weapon AS weapon,
            v.mode_circumstance AS circumstance,
            v.canonical_geo_address_norm AS geo_address_norm,
            v.canonical_geo_address_short AS geo_address_short,
            v.canonical_geo_address_short_2 AS geo_address_short_2,
            v.canonical_address_type AS address_type,
            v.offender_fullname,
            v.article_ids_csv,
            o.entity_origin_type,
            CAST(NULL AS VARCHAR) AS adjudication_reason_summary,
            0 AS band_key,
            v.victim_entity_id AS order_group
          FROM victim_entity_reps_postadj_orphancluster v
          JOIN victim_entity_reps_postadj_orphancluster_origin o
            ON o.victim_entity_id = v.victim_entity_id
          WHERE o.entity_origin_type IN (
            'baseline_entity',
            'machine_matched_entity',
            'adjudication_matched_entity'
          )
        ),
        cluster_leaders AS (
          SELECT
            m.cluster_id,
            MIN(m.unique_id) AS leader_uid
          FROM postadj_orphan_cluster_members m
          JOIN multi_clusters mc
            ON mc.cluster_id = m.cluster_id
          GROUP BY m.cluster_id
        ),
        cluster_leader_rank AS (
          SELECT
            cluster_id,
            leader_uid,
            DENSE_RANK() OVER (ORDER BY leader_uid) AS leader_rank
          FROM cluster_leaders
        ),
        new_cluster_members AS (
          SELECT
            'new_unnamed_cluster_member' AS rec_type,
            m.unique_id AS uid,
            clr.leader_uid AS entity_uid,
            MIN(m.midpoint_day) OVER (PARTITION BY m.cluster_id) AS group_midpoint_day,
            m.midpoint_day,
            m.date_precision,
            m.incident_date,
            CAST(NULL AS VARCHAR) AS canonical_fullname,
            m.victim_sex AS canonical_sex,
            m.victim_race AS canonical_race,
            m.victim_ethnicity AS canonical_ethnicity,
            m.relationship,
            m.victim_age AS canonical_age,
            m.victim_count,
            m.offender_count,
            m.weapon,
            m.circumstance,
            m.geo_address_norm,
            m.geo_address_short,
            m.geo_address_short_2,
            m.address_type,
            m.offender_fullname_concat AS offender_fullname,
            m.source_article_ids_csv AS article_ids_csv,
            'clustered_singleton_orphans' AS entity_origin_type,
            al.reason_summary AS adjudication_reason_summary,
            CASE
              WHEN MOD(clr.leader_rank, 2) = 1
                THEN 1
              ELSE 2
            END AS band_key,
            clr.leader_uid AS order_group
          FROM postadj_orphan_cluster_members m
          JOIN multi_clusters mc
            ON mc.cluster_id = m.cluster_id
          JOIN cluster_leader_rank clr
            ON clr.cluster_id = m.cluster_id
          LEFT JOIN adjudication_lookup al
            ON al.orphan_id = m.unique_id
        ),
        unchanged_singletons AS (
          SELECT
            'unchanged_singleton_orphan' AS rec_type,
            i.unique_id AS uid,
            i.unique_id AS entity_uid,
            i.midpoint_day AS group_midpoint_day,
            i.midpoint_day,
            i.date_precision,
            i.incident_date,
            CAST(NULL AS VARCHAR) AS canonical_fullname,
            i.victim_sex AS canonical_sex,
            i.victim_race AS canonical_race,
            i.victim_ethnicity AS canonical_ethnicity,
            i.relationship,
            i.victim_age AS canonical_age,
            i.victim_count,
            i.offender_count,
            i.weapon,
            i.circumstance,
            i.geo_address_norm,
            i.geo_address_short,
            i.geo_address_short_2,
            i.address_type,
            i.offender_fullname_concat AS offender_fullname,
            i.source_article_ids_csv AS article_ids_csv,
            'singleton_orphan_unresolved' AS entity_origin_type,
            al.reason_summary AS adjudication_reason_summary,
            3 AS band_key,
            i.unique_id AS order_group
          FROM postadj_orphan_cluster_input i
          LEFT JOIN adjudication_lookup al
            ON al.orphan_id = i.unique_id
          WHERE i.unique_id NOT IN (SELECT unique_id FROM _postadj_absorbed_singletons)
        ),
        combined AS (
          SELECT * FROM named_entities
          UNION ALL
          SELECT * FROM new_cluster_members
          UNION ALL
          SELECT * FROM unchanged_singletons
        )
        SELECT
          rec_type,
          uid,
          entity_uid,
          midpoint_day,
          date_precision,
          incident_date,
          canonical_fullname,
          canonical_sex,
          canonical_race,
          canonical_ethnicity,
          relationship,
          canonical_age,
          victim_count,
          offender_count,
          weapon,
          circumstance,
          geo_address_norm,
          geo_address_short,
          geo_address_short_2,
          address_type,
          offender_fullname,
          article_ids_csv,
          entity_origin_type,
          adjudication_reason_summary,
          band_key
        FROM combined
        ORDER BY
          group_midpoint_day NULLS LAST,
          order_group,
          CASE rec_type
            WHEN 'named_entity' THEN 0
            WHEN 'new_unnamed_cluster_member' THEN 1
            ELSE 2
          END,
          midpoint_day NULLS LAST,
          CASE rec_type
            WHEN 'named_entity' THEN 0
            WHEN 'new_unnamed_cluster_member' THEN 1
            ELSE 2
          END,
          uid;
        """
    )
    return (
        sql_export(
            review_select,
            "postadj_orphan_clusters.xlsx",
            "PostAdjOrphanClusters",
            band_by_group_col="band_key",
            band_wrap=4,
        )
        ^ put_line("[O] Wrote postadj_orphan_clusters.xlsx.")
        ^ pure(unit)
    )


def _append_postadj_orphan_cluster_history(run_id: str) -> Run[Unit]:
    return (
        sql_exec(
            SQL(
                """--sql
                CREATE TABLE IF NOT EXISTS postadj_orphan_cluster_history (
                  run_id VARCHAR,
                  run_at TIMESTAMP,
                  input_singleton_count BIGINT,
                  cluster_pair_count BIGINT,
                  multi_member_cluster_count BIGINT,
                  absorbed_singleton_count BIGINT,
                  new_cluster_entity_count BIGINT,
                  unchanged_singleton_count BIGINT,
                  final_entity_count BIGINT
                );
                """
            )
        )
        ^ sql_exec(
            SQL(
                f"""--sql
                INSERT INTO postadj_orphan_cluster_history
                SELECT
                  '{run_id}' AS run_id,
                  NOW() AS run_at,
                  (SELECT COUNT(*) FROM postadj_orphan_cluster_input) AS input_singleton_count,
                  (SELECT COUNT(*) FROM postadj_orphan_cluster_pairs) AS cluster_pair_count,
                  (SELECT COUNT(*) FROM postadj_orphan_cluster_sizes WHERE member_count >= 2) AS multi_member_cluster_count,
                  (SELECT COUNT(*) FROM _postadj_absorbed_singletons) AS absorbed_singleton_count,
                  (SELECT COUNT(*) FROM postadj_orphan_cluster_entities) AS new_cluster_entity_count,
                  (SELECT COUNT(*) FROM postadj_orphan_cluster_input WHERE unique_id NOT IN (SELECT unique_id FROM _postadj_absorbed_singletons)) AS unchanged_singleton_count,
                  (SELECT COUNT(*) FROM victim_entity_reps_postadj_orphancluster) AS final_entity_count;
                """
            )
        )
        ^ pure(unit)
    )


def _log_step(label: str) -> Run[Unit]:
    return (
        sql_query(
            SQL(
                f"""--sql
                SELECT strftime(NOW(), '%Y-%m-%d %H:%M:%S') AS ts, '{label}' AS step;
                """
            )
        )
        >> (lambda rows: put_line(f"[O] {rows[0]['ts']} {rows[0]['step']}"))
        ^ pure(unit)
    )


def _run_postadj_orphan_clustering() -> Run[NextStep]:
    run_id = datetime.now(timezone.utc).strftime("postadj_orphan_cluster_%Y%m%dT%H%M%SZ")
    return (
        put_line(f"[O] Running post-adjudication orphan clustering (run_id={run_id})...")
        ^ _log_step("start")
        ^ _log_step("assert_postadj_inputs")
        ^ _assert_postadj_inputs()
        ^ _log_step("build_postadj_orphan_cluster_input")
        ^ _build_postadj_orphan_cluster_input()
        ^ _log_step("cluster_postadj_singletons")
        ^ _cluster_postadj_singletons()
        ^ _log_step("build_postadj_orphancluster_canonical")
        ^ _build_postadj_orphancluster_canonical()
        ^ _log_step("export_postadj_orphan_clusters_excel")
        ^ _export_postadj_orphan_clusters_excel()
        ^ _log_step("append_postadj_orphan_cluster_history")
        ^ _append_postadj_orphan_cluster_history(run_id)
        ^ _log_step("done")
        ^ put_line("[O] Built victim_entity_reps_postadj_orphancluster (authoritative post-[O] canonical table).")
        ^ pure(NextStep.CONTINUE)
    )


def cluster_postadj_orphans() -> Run[NextStep]:
    """
    Entry point for post-adjudication orphan clustering controller.
    """
    return with_duckdb(_run_postadj_orphan_clustering())
