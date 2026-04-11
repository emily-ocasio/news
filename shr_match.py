"""
Match article victim entities to SHR victims using Splink linkage.
"""
from splink_types import SplinkType
from pymonad import (
    ErrorPayload,
    Run,
    SQL,
    put_line,
    pure,
    PredictionInputTableNames,
    PairsTableName,
    UniquePairsTableName,
    splink_dedupe_job,
    sql_exec,
    sql_export,
    sql_import,
    sql_query,
    throw,
    with_duckdb,
    file_exists,
    rename_file,
    Unit, unit,
)
from menuprompts import NextStep
from comparison import (
    SHR_POST_TRAIN_RATIO_COPY_COMPARISONS,
    SHR_COMPARISONS,
)
from blocking import (
    SHR_OVERALL_BLOCKS,
    SHR_DETERMINISTIC_BLOCKS,
    SHR_TRAINING_BLOCKS,
)


# Define linkage settings for SHR matching
shr_linkage_settings = {
    "link_type": "link_only",
    "comparisons": SHR_COMPARISONS,
    "blocking_rules_to_generate_predictions": SHR_OVERALL_BLOCKS,
    "unique_id_column_name": "unique_id",  # for both tables
}


def _assert_postadj_orphancluster_canonical_exists() -> Run[Unit]:
    return (
        sql_query(
            SQL(
                """--sql
                SELECT COUNT(*) AS n
                FROM pragma_table_info('victim_entity_reps_postadj_orphancluster');
                """
            )
        )
        >> (
            lambda rows: pure(unit)
            if rows[0]["n"] > 0
            else throw(
                ErrorPayload(
                    "[L] Missing victim_entity_reps_postadj_orphancluster. "
                    "Run [O] post-adjudication orphan clustering first."
                )
            )
        )
    )


def _assert_postadj_orphancluster_months_available() -> Run[Unit]:
    return (
        sql_query(
            SQL(
                """--sql
                SELECT COUNT(*) AS n
                FROM victim_entity_reps_postadj_orphancluster
                WHERE entity_midpoint_day IS NOT NULL;
                """
            )
        )
        >> (
            lambda rows: pure(unit)
            if rows[0]["n"] > 0
            else throw(
                ErrorPayload(
                    "[L] victim_entity_reps_postadj_orphancluster has no non-null "
                    "entity_midpoint_day values. Run [O] again or inspect "
                    "upstream entity dates."
                )
            )
        )
    )


def _export_shr_final_matches_excel() -> Run[Unit]:
    """
    Build a single worksheet that lists every article victim entity and every
    SHR record, using only the unique matches from shr_max_weight_matches.

    Matching logic:
      - Assumes unique_id_l is always from entities, unique_id_r from SHR.
      - For SHR linkage, assume one-to-one; no per-article uniqueness needed.
      - Only includes the single best match per entity/SHR combination from
        unique matching.

    Output:
      - Exactly one row per entity, one per SHR record.
      - match_id = 'match_<entity_uid>' for matched groups, otherwise
        'entity_<...>' or 'shr_<...>'.
      - band_key = 0 (unmatched entity), 1 (unmatched SHR), 2 (matched group).
      - Ordering uses the entity midpoint for matched groups; otherwise row's own
        midpoint.
    """

    def _shr_matches_final_select() -> SQL:
        return SQL(
            """--sql
        -- Extract pairs directly (assuming l=entity, r=SHR) - using unique matches only
        WITH pairs_raw AS (
          SELECT
            unique_id_l AS entity_uid,
            unique_id_r AS shr_uid,
            match_probability
          FROM shr_max_weight_matches
        ),
        combined AS (
          -- Matched pairs
          SELECT
            'match' AS rec_type,
            av.midpoint_day AS entity_midpoint_day,
            av.date_precision AS entity_date_precision,
            av.year AS entity_year, av.month AS entity_month,
            av.victim_age, av.victim_sex, av.victim_race, av.victim_ethnicity,
            av.relationship,
            av.victim_count,
            av.weapon, av.circumstance,
            av.offender_age, av.offender_sex, av.offender_race, av.offender_ethnicity,
            ver.article_ids_csv,
            ver.canonical_fullname AS canonical_victim_name,
            ver.offender_fullname AS canonical_offender_name,
            pr.match_probability,
            pr.entity_uid,
            pr.shr_uid,
            av.victim_age AS entity_victim_age,
            av.victim_count AS entity_victim_count,
            av.victim_sex AS entity_victim_sex,
            av.victim_race AS entity_victim_race,
            av.victim_ethnicity AS entity_victim_ethnicity,
            av.weapon AS entity_weapon,
            av.circumstance AS entity_circumstance,
            av.offender_age AS entity_offender_age,
            av.offender_sex AS entity_offender_sex,
            av.offender_race AS entity_offender_race,
            av.offender_ethnicity AS entity_offender_ethnicity,
            sc.victim_age AS shr_victim_age,
            sc.victim_count AS shr_victim_count,
            sc.victim_sex AS shr_victim_sex,
            sc.victim_race AS shr_victim_race,
            sc.victim_ethnicity AS shr_victim_ethnicity,
            sc.relationship AS shr_relationship,
            sc.weapon AS shr_weapon,
            sc.circumstance AS shr_circumstance,
            sc.offender_age AS shr_offender_age,
            sc.offender_sex AS shr_offender_sex,
            sc.offender_race AS shr_offender_race,
            sc.offender_ethnicity AS shr_offender_ethnicity,
            sc.year AS shr_year,
            sc.month AS shr_month,
            sc.date_precision AS shr_date_precision,
            sc.midpoint_day AS shr_midpoint_day
          FROM pairs_raw pr
          JOIN article_victims av ON pr.entity_uid = av.unique_id
          LEFT JOIN victim_entity_reps_postadj_orphancluster ver
            ON av.unique_id = ver.victim_entity_id
          JOIN shr_cached sc ON pr.shr_uid = sc.unique_id

          UNION ALL

          -- Unmatched entities
          SELECT
            'entity' AS rec_type,
            av.midpoint_day AS entity_midpoint_day,
            av.date_precision AS entity_date_precision,
            av.year AS entity_year, av.month AS entity_month,
            av.victim_age, av.victim_sex, av.victim_race, av.victim_ethnicity,
            av.relationship,
            av.victim_count,
            av.weapon, av.circumstance,
            av.offender_age, av.offender_sex, av.offender_race, av.offender_ethnicity,
            ver.article_ids_csv,
            ver.canonical_fullname AS canonical_victim_name,
            ver.offender_fullname AS canonical_offender_name,
            CAST(NULL AS DOUBLE) AS match_probability,
            av.unique_id AS entity_uid,
            CAST(NULL AS VARCHAR) AS shr_uid,
            av.victim_age AS entity_victim_age,
            av.victim_count AS entity_victim_count,
            av.victim_sex AS entity_victim_sex,
            av.victim_race AS entity_victim_race,
            av.victim_ethnicity AS entity_victim_ethnicity,
            av.weapon AS entity_weapon,
            av.circumstance AS entity_circumstance,
            av.offender_age AS entity_offender_age,
            av.offender_sex AS entity_offender_sex,
            av.offender_race AS entity_offender_race,
            av.offender_ethnicity AS entity_offender_ethnicity,
            CAST(NULL AS INTEGER) AS shr_victim_age,
            CAST(NULL AS INTEGER) AS shr_victim_count,
            CAST(NULL AS VARCHAR) AS shr_victim_sex,
            CAST(NULL AS VARCHAR) AS shr_victim_race,
            CAST(NULL AS VARCHAR) AS shr_victim_ethnicity,
            CAST(NULL AS VARCHAR) AS shr_relationship,
            CAST(NULL AS VARCHAR) AS shr_weapon,
            CAST(NULL AS VARCHAR) AS shr_circumstance,
            CAST(NULL AS INTEGER) AS shr_offender_age,
            CAST(NULL AS VARCHAR) AS shr_offender_sex,
            CAST(NULL AS VARCHAR) AS shr_offender_race,
            CAST(NULL AS VARCHAR) AS shr_offender_ethnicity,
            CAST(NULL AS INTEGER) AS shr_year,
            CAST(NULL AS INTEGER) AS shr_month,
            CAST(NULL AS VARCHAR) AS shr_date_precision,
            CAST(NULL AS DOUBLE) AS shr_midpoint_day
          FROM article_victims av
          LEFT JOIN victim_entity_reps_postadj_orphancluster ver
            ON av.unique_id = ver.victim_entity_id
          WHERE av.unique_id NOT IN (SELECT entity_uid FROM pairs_raw)

          UNION ALL

          -- Unmatched SHR
          SELECT
            'shr' AS rec_type,
            CAST(NULL AS DOUBLE) AS entity_midpoint_day,
            CAST(NULL AS VARCHAR) AS entity_date_precision,
            CAST(NULL AS INTEGER) AS entity_year, CAST(NULL AS INTEGER) AS entity_month,
            sc.victim_age, sc.victim_sex, sc.victim_race, sc.victim_ethnicity,
            CAST(NULL AS VARCHAR) AS relationship,
            sc.victim_count,
            sc.weapon, sc.circumstance,
            sc.offender_age, sc.offender_sex, sc.offender_race, sc.offender_ethnicity,
            CAST(NULL AS VARCHAR) AS article_ids_csv,
            CAST(NULL AS VARCHAR) AS canonical_victim_name,
            CAST(NULL AS VARCHAR) AS canonical_offender_name,
            CAST(NULL AS DOUBLE) AS match_probability,
            CAST(NULL AS VARCHAR) AS entity_uid,
            sc.unique_id AS shr_uid,
            CAST(NULL AS INTEGER) AS entity_victim_age,
            CAST(NULL AS INTEGER) AS entity_victim_count,
            CAST(NULL AS VARCHAR) AS entity_victim_sex,
            CAST(NULL AS VARCHAR) AS entity_victim_race,
            CAST(NULL AS VARCHAR) AS entity_victim_ethnicity,
            CAST(NULL AS VARCHAR) AS entity_weapon,
            CAST(NULL AS VARCHAR) AS entity_circumstance,
            CAST(NULL AS INTEGER) AS entity_offender_age,
            CAST(NULL AS VARCHAR) AS entity_offender_sex,
            CAST(NULL AS VARCHAR) AS entity_offender_race,
            CAST(NULL AS VARCHAR) AS entity_offender_ethnicity,
            sc.victim_age AS shr_victim_age,
            sc.victim_count AS shr_victim_count,
            sc.victim_sex AS shr_victim_sex,
            sc.victim_race AS shr_victim_race,
            sc.victim_ethnicity AS shr_victim_ethnicity,
            sc.relationship AS shr_relationship,
            sc.weapon AS shr_weapon,
            sc.circumstance AS shr_circumstance,
            sc.offender_age AS shr_offender_age,
            sc.offender_sex AS shr_offender_sex,
            sc.offender_race AS shr_offender_race,
            sc.offender_ethnicity AS shr_offender_ethnicity,
            sc.year AS shr_year,
            sc.month AS shr_month,
            sc.date_precision AS shr_date_precision,
            sc.midpoint_day AS shr_midpoint_day
          FROM shr_cached sc
          WHERE CAST(sc.unique_id AS VARCHAR) NOT IN (SELECT shr_uid FROM pairs_raw)
        )
        SELECT
          rec_type,
          CASE
            WHEN rec_type = 'match' THEN concat('match_', entity_uid)
            WHEN rec_type = 'entity' THEN concat('entity_', entity_uid)
            ELSE concat('shr_', shr_uid)
          END AS match_id,
          CASE
            WHEN rec_type = 'match'
              THEN concat('match::', entity_uid, '::', shr_uid)
            WHEN rec_type = 'entity'
              THEN concat('entity::', entity_uid)
            ELSE concat('shr::', shr_uid)
          END AS match_uid,
          entity_uid AS e_uid,
          shr_uid AS s_uid,
          canonical_victim_name,
          canonical_offender_name,
          match_probability AS prob,
          entity_year AS e_year,
          shr_year AS s_year,
          entity_month AS e_mon,
          shr_month AS s_mon,
          entity_date_precision AS e_prec,
          shr_date_precision AS s_prec,
          entity_midpoint_day AS e_mid,
          shr_midpoint_day AS s_mid,
          entity_victim_age AS e_vage,
          shr_victim_age AS s_vage,
          entity_victim_count AS e_vcount,
          shr_victim_count AS s_vcount,
          entity_victim_sex AS e_vsex,
          shr_victim_sex AS s_vsex,
          entity_victim_race AS e_vrac,
          shr_victim_race AS s_vrac,
          entity_victim_ethnicity AS e_veth,
          shr_victim_ethnicity AS s_veth,
          relationship AS e_rel,
          shr_relationship AS s_rel,
          entity_weapon AS e_weap,
          shr_weapon AS s_weap,
          entity_circumstance AS e_circ,
          shr_circumstance AS s_circ,
          entity_offender_age AS e_oage,
          shr_offender_age AS s_oage,
          entity_offender_sex AS e_osex,
          shr_offender_sex AS s_osex,
          entity_offender_race AS e_orac,
          shr_offender_race AS s_orac,
          entity_offender_ethnicity AS e_oeth,
          shr_offender_ethnicity AS s_oeth,
          article_ids_csv AS entity_article_ids,
          CASE WHEN rec_type = 'match' THEN 2
              WHEN rec_type = 'entity' THEN 0
              ELSE 1
          END AS band_key
        FROM combined
        ORDER BY
          COALESCE(entity_midpoint_day, shr_midpoint_day),
          CASE rec_type WHEN 'match' THEN 0 WHEN 'entity' THEN 1 ELSE 2 END,
          entity_uid,
          match_probability DESC,
          shr_uid
        """
        )

    def _normalize_prev_shr_matches_table(
        raw_table: str = "shr_matches_prev_raw",
        output_table: str = "shr_matches_prev",
    ) -> Run[Unit]:
        def _build_normalized_table(cols: set[str]) -> Run[Unit]:
            required_legacy_cols = {"rec_type", "e_uid", "s_uid"}
            if required_legacy_cols.issubset(cols):
                exclude_cols = ["rec_type"]
                if "match_id" in cols:
                    exclude_cols.append("match_id")
                if "match_uid" in cols:
                    exclude_cols.append("match_uid")
                exclude_sql = ", ".join(exclude_cols)
                normalized_shr_uid = """CASE
                  WHEN s_uid IS NULL THEN NULL
                  WHEN TRY_CAST(s_uid AS BIGINT) IS NOT NULL
                    THEN CAST(TRY_CAST(s_uid AS BIGINT) AS VARCHAR)
                  ELSE trim(CAST(s_uid AS VARCHAR))
                END"""
                had_keys = "match_id" in cols and "match_uid" in cols
                return (
                    sql_exec(
                        SQL(
                            f"""--sql
                    CREATE OR REPLACE TABLE {output_table} AS
                    SELECT
                      CAST(rec_type AS VARCHAR) AS rec_type,
                      CASE
                        WHEN rec_type = 'match'
                          THEN concat('match_', trim(CAST(e_uid AS VARCHAR)))
                        WHEN rec_type = 'entity'
                          THEN concat('entity_', trim(CAST(e_uid AS VARCHAR)))
                        ELSE concat('shr_', {normalized_shr_uid})
                      END AS match_id,
                      CASE
                        WHEN rec_type = 'match'
                          THEN concat(
                            'match::',
                            trim(CAST(e_uid AS VARCHAR)),
                            '::',
                            {normalized_shr_uid}
                          )
                        WHEN rec_type = 'entity'
                          THEN concat('entity::', trim(CAST(e_uid AS VARCHAR)))
                        ELSE concat('shr::', {normalized_shr_uid})
                      END AS match_uid,
                      * EXCLUDE ({exclude_sql})
                    FROM {raw_table}
                    """
                        )
                    )
                    ^ (
                        pure(unit)
                        if had_keys
                        else put_line(
                            "[L] Added derived diff key columns to legacy "
                            "shr_matches base workbook."
                        )
                        ^ pure(unit)
                    )
                )

            missing_legacy_cols = sorted(required_legacy_cols - cols)
            raise ValueError(
                "[L] Imported SHR base workbook is missing required columns "
                "for diff normalization: "
                f"{', '.join(missing_legacy_cols)}"
            )

        return sql_query(SQL(f"PRAGMA table_info('{raw_table}');")) >> (
            lambda rows: _build_normalized_table({row["name"] for row in rows})
        )

    def _register_previous_shr_matches_if_exists() -> Run[bool]:
        base_path = "shr_matches_base.xlsx"
        current_path = "shr_matches.xlsx"

        def _load_base() -> Run[bool]:
            return (
                sql_import(base_path, "shr_matches_prev_raw")
                ^ _normalize_prev_shr_matches_table()
                ^ put_line(
                    "[L] Loaded existing shr_matches_base.xlsx into shr_matches_prev."
                )
                ^ pure(True)
            )

        def _maybe_rename_and_load(has_current: bool) -> Run[bool]:
            if not has_current:
                return (
                    put_line(
                        "[L] No existing shr_matches_base.xlsx or "
                        "shr_matches.xlsx found."
                    )
                    ^ pure(False)
                )
            return (
                rename_file(current_path, base_path)
                ^ put_line(
                    "[L] Renamed shr_matches.xlsx to shr_matches_base.xlsx."
                )
                ^ _load_base()
            )

        return file_exists(base_path) >> (
            lambda has_base: _load_base()
            if has_base
            else file_exists(current_path) >> _maybe_rename_and_load
        )

    def _maybe_export_shr_match_diffs(has_prev: bool) -> Run[Unit]:
        if not has_prev:
            return pure(unit)
        return (
            sql_exec(
                SQL(
                    """--sql
            CREATE OR REPLACE TABLE shr_matches_diffs AS
            WITH prev_match AS (
              SELECT *
              FROM shr_matches_prev
              WHERE rec_type = 'match'
            ),
            current_match AS (
              SELECT *
              FROM shr_matches_current
              WHERE rec_type = 'match'
            ),
            gone_matches AS (
              SELECT
                1 AS source,
                'gone_match' AS diff_type,
                0 AS diff_band,
                p.*
              FROM prev_match p
              LEFT JOIN current_match c
                ON c.match_uid = p.match_uid
              WHERE c.match_uid IS NULL
            ),
            new_matches AS (
              SELECT
                2 AS source,
                'new_match' AS diff_type,
                1 AS diff_band,
                c.*
              FROM current_match c
              LEFT JOIN prev_match p
                ON p.match_uid = c.match_uid
              WHERE p.match_uid IS NULL
            )
            SELECT *
            FROM gone_matches
            UNION ALL
            SELECT *
            FROM new_matches;
            """
                )
            )
            >> (
                lambda _: sql_query(SQL("SELECT COUNT(*) AS n FROM shr_matches_diffs"))
                >> (
                    lambda rows: sql_export(
                        SQL(
                            """--sql
                    SELECT
                      *,
                      diff_band AS __diff_band
                    FROM shr_matches_diffs
                    ORDER BY
                      e_mid NULLS LAST,
                      e_uid,
                      source,
                      prob DESC NULLS LAST,
                      s_uid
                    """
                        ),
                        "shr_matches_diffs.xlsx",
                        "MatchesDiffs",
                        band_by_group_col="__diff_band",
                        band_wrap=2,
                    )
                    ^ put_line(
                        "[L] Wrote shr_matches_diffs.xlsx "
                        f"(match-only new/gone diffs, {rows[0]['n']} rows)."
                    )
                )
            )
            ^ pure(unit)
        )

    shr_matches_final_select = _shr_matches_final_select()
    return (
        _register_previous_shr_matches_if_exists()
        >> (
            lambda has_prev: (
                sql_exec(
                    SQL(
                        f"""--sql
                CREATE OR REPLACE TABLE shr_matches_current AS
                {shr_matches_final_select}
                """
                    )
                )
                ^ _maybe_export_shr_match_diffs(has_prev)
            )
        )
        ^ sql_export(
            shr_matches_final_select,
            "shr_matches.xlsx",
            "Matches",
            band_by_group_col="band_key",
            band_wrap=3,
        )
        ^ put_line(
            "[SHR] Wrote shr_matches.xlsx "
            "(final entities + SHR records with unique matches)."
        )
        ^ pure(unit)
    )


def _export_shr_debug_matches_excel() -> Run[Unit]:
    """
    Build a single worksheet that lists every article victim entity and every
    SHR record.

    Matching logic:
      - Assumes unique_id_l is always from entities, unique_id_r from SHR.
      - For SHR linkage, assume one-to-one; no per-article uniqueness needed.

    Output:
      - Exactly one row per entity, one per SHR record.
      - match_id = 'match_<entity_uid>' for matched groups, otherwise
        'entity_<...>' or 'shr_<...>'.
      - band_key = 0 (unmatched entity), 1 (unmatched SHR), 2 (matched
        group).
      - Ordering uses the entity midpoint for matched groups; otherwise row's
        own midpoint.
    """
    return (
        sql_export(
            SQL(
                """--sql
            -- Extract pairs directly (assuming l=entity, r=SHR)
            WITH pairs_raw AS (
              SELECT
                unique_id_l AS entity_uid,
                unique_id_r AS shr_uid,
                match_probability
              FROM shr_link_pairs
            ),
            combined AS (
              -- Matched pairs
              SELECT
                'match' AS rec_type,
                av.midpoint_day AS entity_midpoint_day,
                av.date_precision AS entity_date_precision,
                av.year AS entity_year, av.month AS entity_month,
                av.victim_age, av.victim_sex, av.victim_race, av.victim_ethnicity,
                av.relationship,
                av.victim_count,
                av.weapon, av.circumstance,
                av.offender_age, av.offender_sex, av.offender_race,
                av.offender_ethnicity,
                ver.article_ids_csv,
                ver.canonical_fullname AS canonical_victim_name,
                ver.offender_fullname AS canonical_offender_name,
                pr.match_probability,
                pr.entity_uid,
                pr.shr_uid,
                av.victim_age AS entity_victim_age,
                av.victim_count AS entity_victim_count,
                av.victim_sex AS entity_victim_sex,
                av.victim_race AS entity_victim_race,
                av.victim_ethnicity AS entity_victim_ethnicity,
                av.weapon AS entity_weapon,
                av.circumstance AS entity_circumstance,
                av.offender_age AS entity_offender_age,
                av.offender_sex AS entity_offender_sex,
                av.offender_race AS entity_offender_race,
                av.offender_ethnicity AS entity_offender_ethnicity,
                sc.victim_age AS shr_victim_age,
                sc.victim_count AS shr_victim_count,
                sc.victim_sex AS shr_victim_sex,
                sc.victim_race AS shr_victim_race,
                sc.victim_ethnicity AS shr_victim_ethnicity,
                sc.relationship AS shr_relationship,
                sc.weapon AS shr_weapon,
                sc.circumstance AS shr_circumstance,
                sc.offender_age AS shr_offender_age,
                sc.offender_sex AS shr_offender_sex,
                sc.offender_race AS shr_offender_race,
                sc.offender_ethnicity AS shr_offender_ethnicity,
                sc.year AS shr_year,
                sc.month AS shr_month,
                sc.date_precision AS shr_date_precision,
                sc.midpoint_day AS shr_midpoint_day
              FROM pairs_raw pr
              JOIN article_victims av ON pr.entity_uid = av.unique_id
              LEFT JOIN victim_entity_reps_postadj_orphancluster ver
                ON av.unique_id = ver.victim_entity_id
              JOIN shr_cached sc ON pr.shr_uid = sc.unique_id

              UNION ALL

              -- Unmatched entities
              SELECT
                'entity' AS rec_type,
                av.midpoint_day AS entity_midpoint_day,
                av.date_precision AS entity_date_precision,
                av.year AS entity_year, av.month AS entity_month,
                av.victim_age, av.victim_sex, av.victim_race, av.victim_ethnicity,
                av.relationship,
                av.victim_count,
                av.weapon, av.circumstance,
                av.offender_age, av.offender_sex, av.offender_race,
                av.offender_ethnicity,
                ver.article_ids_csv,
                ver.canonical_fullname AS canonical_victim_name,
                ver.offender_fullname AS canonical_offender_name,
                CAST(NULL AS DOUBLE) AS match_probability,
                av.unique_id AS entity_uid,
                CAST(NULL AS VARCHAR) AS shr_uid,
                av.victim_age AS entity_victim_age,
                av.victim_count AS entity_victim_count,
                av.victim_sex AS entity_victim_sex,
                av.victim_race AS entity_victim_race,
                av.victim_ethnicity AS entity_victim_ethnicity,
                av.weapon AS entity_weapon,
                av.circumstance AS entity_circumstance,
                av.offender_age AS entity_offender_age,
                av.offender_sex AS entity_offender_sex,
                av.offender_race AS entity_offender_race,
                av.offender_ethnicity AS entity_offender_ethnicity,
                CAST(NULL AS INTEGER) AS shr_victim_age,
                CAST(NULL AS INTEGER) AS shr_victim_count,
                CAST(NULL AS VARCHAR) AS shr_victim_sex,
                CAST(NULL AS VARCHAR) AS shr_victim_race,
                CAST(NULL AS VARCHAR) AS shr_victim_ethnicity,
                CAST(NULL AS VARCHAR) AS shr_relationship,
                CAST(NULL AS VARCHAR) AS shr_weapon,
                CAST(NULL AS VARCHAR) AS shr_circumstance,
                CAST(NULL AS INTEGER) AS shr_offender_age,
                CAST(NULL AS VARCHAR) AS shr_offender_sex,
                CAST(NULL AS VARCHAR) AS shr_offender_race,
                CAST(NULL AS VARCHAR) AS shr_offender_ethnicity,
                CAST(NULL AS INTEGER) AS shr_year,
                CAST(NULL AS INTEGER) AS shr_month,
                CAST(NULL AS VARCHAR) AS shr_date_precision,
                CAST(NULL AS DOUBLE) AS shr_midpoint_day
              FROM article_victims av
              LEFT JOIN victim_entity_reps_postadj_orphancluster ver
                ON av.unique_id = ver.victim_entity_id
              WHERE av.unique_id NOT IN (SELECT entity_uid FROM pairs_raw)

              UNION ALL

              -- Unmatched SHR
              SELECT
                'shr' AS rec_type,
                CAST(NULL AS DOUBLE) AS entity_midpoint_day,
                CAST(NULL AS VARCHAR) AS entity_date_precision,
                CAST(NULL AS INTEGER) AS entity_year,
                CAST(NULL AS INTEGER) AS entity_month,
                sc.victim_age, sc.victim_sex, sc.victim_race, sc.victim_ethnicity,
                CAST(NULL AS VARCHAR) AS relationship,
                sc.victim_count,
                sc.weapon, sc.circumstance,
                sc.offender_age, sc.offender_sex, sc.offender_race,
                sc.offender_ethnicity,
                CAST(NULL AS VARCHAR) AS article_ids_csv,
                CAST(NULL AS VARCHAR) AS canonical_victim_name,
                CAST(NULL AS VARCHAR) AS canonical_offender_name,
                CAST(NULL AS DOUBLE) AS match_probability,
                CAST(NULL AS VARCHAR) AS entity_uid,
                sc.unique_id AS shr_uid,
                CAST(NULL AS INTEGER) AS entity_victim_age,
                CAST(NULL AS INTEGER) AS entity_victim_count,
                CAST(NULL AS VARCHAR) AS entity_victim_sex,
                CAST(NULL AS VARCHAR) AS entity_victim_race,
                CAST(NULL AS VARCHAR) AS entity_victim_ethnicity,
                CAST(NULL AS VARCHAR) AS entity_weapon,
                CAST(NULL AS VARCHAR) AS entity_circumstance,
                CAST(NULL AS INTEGER) AS entity_offender_age,
                CAST(NULL AS VARCHAR) AS entity_offender_sex,
                CAST(NULL AS VARCHAR) AS entity_offender_race,
                CAST(NULL AS VARCHAR) AS entity_offender_ethnicity,
                sc.victim_age AS shr_victim_age,
                sc.victim_count AS shr_victim_count,
                sc.victim_sex AS shr_victim_sex,
                sc.victim_race AS shr_victim_race,
                sc.victim_ethnicity AS shr_victim_ethnicity,
                sc.relationship AS shr_relationship,
                sc.weapon AS shr_weapon,
                sc.circumstance AS shr_circumstance,
                sc.offender_age AS shr_offender_age,
                sc.offender_sex AS shr_offender_sex,
                sc.offender_race AS shr_offender_race,
                sc.offender_ethnicity AS shr_offender_ethnicity,
                sc.year AS shr_year,
                sc.month AS shr_month,
                sc.date_precision AS shr_date_precision,
                sc.midpoint_day AS shr_midpoint_day
              FROM shr_cached sc
              WHERE CAST(sc.unique_id as VARCHAR) NOT IN (SELECT shr_uid FROM pairs_raw)
            )
            SELECT
              rec_type,
              entity_uid AS e_uid,
              shr_uid AS s_uid,
              canonical_victim_name,
              canonical_offender_name,
              match_probability AS prob,
              entity_year AS e_year,
              shr_year AS s_year,
              entity_month AS e_mon,
              shr_month AS s_mon,
              entity_date_precision AS e_prec,
              shr_date_precision AS s_prec,
              entity_midpoint_day AS e_mid,
              shr_midpoint_day AS s_mid,
              entity_victim_age AS e_vage,
              shr_victim_age AS s_vage,
              entity_victim_count AS e_vcount,
              shr_victim_count AS s_vcount,
              entity_victim_sex AS e_vsex,
              shr_victim_sex AS s_vsex,
              entity_victim_race AS e_vrac,
              shr_victim_race AS s_vrac,
              entity_victim_ethnicity AS e_veth,
              shr_victim_ethnicity AS s_veth,
              relationship AS e_rel,
              shr_relationship AS s_rel,
              entity_weapon AS e_weap,
              shr_weapon AS s_weap,
              entity_circumstance AS e_circ,
              shr_circumstance AS s_circ,
              entity_offender_age AS e_oage,
              shr_offender_age AS s_oage,
              entity_offender_sex AS e_osex,
              shr_offender_sex AS s_osex,
              entity_offender_race AS e_orac,
              shr_offender_race AS s_orac,
              entity_offender_ethnicity AS e_oeth,
              shr_offender_ethnicity AS s_oeth,
              article_ids_csv AS entity_article_ids,
              CASE WHEN rec_type = 'match' THEN 2
                  WHEN rec_type = 'entity' THEN 0
                  ELSE 1
              END AS band_key
            FROM combined
            ORDER BY
              COALESCE(entity_midpoint_day, shr_midpoint_day),
              CASE rec_type WHEN 'match' THEN 0 WHEN 'entity' THEN 1 ELSE 2 END,
              entity_uid, match_probability DESC, shr_uid
            """
            ),
            "shr_debug_matches.xlsx",
            "Matches",
            band_by_group_col="band_key",
            band_wrap=3,
        )
        ^ put_line(
            "[SHR] Wrote shr_debug_matches.xlsx "
            "(final entities + SHR records after linkage)."
        )
        ^ pure(unit)
    )


def match_article_to_shr_victims() -> Run[NextStep]:
    """
    Export SHR and article victim data to DuckDB, then run Splink linkage.
    """
    return with_duckdb(
        _assert_postadj_orphancluster_canonical_exists() ^
        _assert_postadj_orphancluster_months_available() ^
        sql_exec(
            SQL(
                """
                CREATE OR REPLACE TABLE shr_linkage_months AS
                WITH entity_months AS (
                    SELECT DISTINCT
                        date_trunc(
                            'month',
                            DATE '1970-01-01' + to_days(entity_midpoint_day)
                        ) AS month_start
                    FROM victim_entity_reps_postadj_orphancluster
                    WHERE entity_midpoint_day IS NOT NULL
                    AND date_trunc(
                        'month',
                        DATE '1970-01-01' + to_days(entity_midpoint_day)
                    ) >= DATE '1977-01-01'
                    AND date_trunc(
                        'month',
                        DATE '1970-01-01' + to_days(entity_midpoint_day)
                    ) < DATE '1996-01-01'
                ),
                shr_months AS (
                    SELECT DISTINCT
                        make_date(
                            CAST(substring(YearMonth, 1, 4) AS BIGINT),
                            CAST(substring(YearMonth, 6, 2) AS BIGINT),
                            1
                        ) AS month_start
                    FROM sqldb.shr
                    WHERE State = 'District of Columbia'
                    AND Year >= 1977
                    AND Year <= 1995
                    AND YearMonth IS NOT NULL
                )
                SELECT em.month_start
                FROM entity_months em
                INNER JOIN shr_months sm
                    ON em.month_start = sm.month_start
                ORDER BY em.month_start
                """
            )
        ) ^
        put_line("Exporting SHR data to DuckDB...") ^
        sql_exec(
            SQL(
                """
                CREATE OR REPLACE TABLE shr_cached AS
                WITH shr_source AS (
                    SELECT
                        *,
                        DENSE_RANK() OVER (
                            PARTITION BY YearMonth
                            ORDER BY Incident
                        ) AS month_incident_rank
                    FROM sqldb.shr
                    WHERE State = 'District of Columbia' -- Only DC for now
                    AND Year >= 1977
                    AND Year <= 1995
                    AND YearMonth IS NOT NULL
                    AND EXISTS (
                        SELECT 1
                        FROM shr_linkage_months lm
                        WHERE lm.month_start = make_date(
                            CAST(substring(YearMonth, 1, 4) AS BIGINT),
                            CAST(substring(YearMonth, 6, 2) AS BIGINT),
                            1
                        )
                    )
                ),
                shr_ranked AS (
                    SELECT
                        *,
                        MAX(month_incident_rank) OVER (
                            PARTITION BY YearMonth
                        ) AS month_incident_count
                    FROM shr_source
                )
                SELECT
                    *,
                    midpoint_day AS midpoint_day_block,
                    year AS year_block
                FROM (
                    SELECT
                        "index" as unique_id,
                        CASE VicSex
                            WHEN 'Unknown' THEN NULL
                            ELSE lower(VicSex)
                        END as victim_sex,
                        CASE VicAge WHEN 999 THEN NULL ELSE VicAge END as victim_age,
                        CASE VicRace
                            WHEN 'Unknown' THEN NULL
                            ELSE VicRace
                        END as victim_race,
                        CASE VicEthnic
                            WHEN 'Unknown' THEN NULL
                            ELSE VicEthnic
                        END as victim_ethnicity,
                        VicCount+1 as victim_count,
                        CASE OffSex
                            WHEN 'Unknown' THEN NULL
                            ELSE lower(OffSex)
                        END as offender_sex,
                        CASE OffAge WHEN 999 THEN NULL ELSE OffAge END as offender_age,
                        CASE OffRace
                            WHEN 'Unknown' THEN NULL
                            ELSE OffRace
                        END as offender_race,
                        CASE OffEthnic
                            WHEN 'Unknown' THEN NULL
                            ELSE OffEthnic
                        END as offender_ethnicity,
                        CASE
                            WHEN OffCount IS NULL THEN NULL
                            ELSE OffCount + 1
                        END AS offender_count,
                        CASE
                            WHEN lower(trim(coalesce(Relationship, ''))) IN (
                                'acquaintance', 'son', 'daughter', 'husband',
                                'stranger', 'wife', 'ex-wife', 'ex-husband',
                                'brother', 'sister', 'other family',
                                'girlfriend', 'boyfriend', 'neighbor',
                                'stepfather', 'stepmother', 'stepson',
                                'friend', 'other known to victim', 'mother',
                                'father',
                                'in-law', 'employee', 'homosexual relationship'
                            ) THEN lower(trim(Relationship))
                            WHEN lower(trim(coalesce(Relationship, '')))
                                = 'other - known to victim'
                                THEN 'other known to victim'
                            WHEN lower(trim(coalesce(Relationship, '')))
                                = 'common-law wife'
                                THEN 'wife'
                            WHEN lower(trim(coalesce(Relationship, '')))
                                = 'common-law husband'
                                THEN 'husband'
                            ELSE NULL
                        END AS relationship,
                        -- Map SHR Weapon to GPT schema enums
                        CASE
                            WHEN LOWER(TRIM(Weapon)) LIKE '%knife%' THEN 'knife'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%shotgun%' THEN 'shotgun'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%rifle%' THEN 'rifle'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%handgun%' THEN 'firearm'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%firearm%' THEN 'firearm'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%gun%' THEN 'firearm'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%blunt%' THEN 'blunt object'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%personal%'
                                THEN 'personal weapon'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%fire%' THEN 'fire'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%strangulation%'
                                THEN 'strangulation'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%asphyxiation%'
                                THEN 'asphyxiation'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%drugs%' THEN 'drugs'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%explosives%'
                                THEN 'explosives'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%drowning%' THEN 'drowning'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%poison%' THEN 'poison'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%pushed%'
                                THEN 'pushed from height'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%other%' THEN 'other'
                            WHEN LOWER(TRIM(Weapon)) LIKE '%unknown%' THEN 'unknown'
                            ELSE 'unknown'  -- Default to unknown for unmapped values
                        END AS weapon,
                        -- Map SHR Circumstance to GPT schema enums
                        CASE
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%narcotic%'
                                THEN 'narcotics related'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%lover%'
                                THEN 'lover''s triangle'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%undetermined%'
                                THEN 'undetermined'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%argument%'
                                THEN 'argument'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%robbery%'
                                THEN 'robbery'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%brawl%' THEN 'brawl'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%motor vehicle%'
                                THEN 'other felony related'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%felony%'
                                THEN 'other felony related'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%arson%' THEN 'arson'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%negligent%'
                                THEN 'negligence'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%manslaughter%'
                                THEN 'negligence'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%babysitter%'
                                THEN 'child killed by babysitter'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%gang%'
                                THEN 'gang killing'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%burglary%'
                                THEN 'burglary'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%rape%' THEN 'rape'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%playing with gun%'
                                THEN 'negligence'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%hunting accident%'
                                THEN 'negligence'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%institutional%'
                                THEN 'institutional killing'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%prostitution%'
                                THEN 'other felony related'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%larceny%'
                                THEN 'other felony related'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%gambling%'
                                THEN 'other felony related'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%sex offense%'
                                THEN 'rape'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%police%'
                                THEN 'felon killed by police'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%private citizen%'
                                THEN 'felon killed by private citizen'
                            WHEN LOWER(TRIM(Circumstance)) LIKE '%other%' THEN 'other'
                            ELSE 'undetermined'
                            -- Default to undetermined for unmapped values
                        END AS circumstance,
                        'month' AS date_precision,
                        CASE WHEN YearMonth IS NOT NULL THEN
                            ROUND(
                                date_diff(
                                    'day',
                                    DATE '1970-01-01',
                                    make_date(
                                        CAST(substring(YearMonth, 1, 4) AS BIGINT),
                                        CAST(substring(YearMonth, 6, 2) AS BIGINT),
                                        1
                                    )
                                ) - 1 +
                                month_incident_rank * day(
                                    make_date(
                                        CAST(substring(YearMonth, 1, 4) AS BIGINT),
                                        CAST(substring(YearMonth, 6, 2) AS BIGINT),
                                        1
                                    ) + INTERVAL 1 MONTH - INTERVAL 1 DAY
                                ) / month_incident_count
                            )
                        ELSE NULL END AS midpoint_day,
                        CAST(substring(YearMonth, 1, 4) AS INTEGER) AS year,
                        CAST(substring(YearMonth, 6, 2) AS INTEGER) AS month,
                        -- NULL AS lat,  -- SHR may not have precise coords
                        -- NULL AS lon,
                        2 AS city_id  -- Corresponds to DC (PublicationID of Washi Post)
                    FROM shr_ranked
                ) AS shr_rows
                """
            )
        ) ^
        put_line("Exporting article victim entities to DuckDB...") ^
        sql_exec(
            SQL(
                """
                CREATE OR REPLACE TABLE article_victims AS
                SELECT
                    *,
                    midpoint_day AS midpoint_day_block,
                    year AS year_block
                FROM (
                    SELECT
                        victim_entity_id AS unique_id,
                        city_id,
                        entity_midpoint_day AS midpoint_day,
                        entity_date_precision AS date_precision,
                        -- lat_centroid AS lat,
                        -- lon_centroid AS lon,
                        extract(
                            year from (
                                DATE '1970-01-01' + to_days(entity_midpoint_day)
                            )
                        ) AS year,
                        extract(
                            month from (
                                DATE '1970-01-01' + to_days(entity_midpoint_day)
                            )
                        ) AS month,
                        canonical_age AS victim_age,
                        canonical_victim_count AS victim_count,
                        canonical_sex AS victim_sex,
                        canonical_race AS victim_race,
                        canonical_ethnicity AS victim_ethnicity,
                        canonical_relationship AS relationship,
                        canonical_offender_age AS offender_age,
                        canonical_offender_sex AS offender_sex,
                        canonical_offender_race AS offender_race,
                        canonical_offender_ethnicity AS offender_ethnicity,
                        canonical_offender_count AS offender_count,
                        CASE
                            WHEN LOWER(TRIM(mode_weapon)) = 'handgun' THEN 'firearm'
                            ELSE mode_weapon
                        END AS weapon,
                        mode_circumstance AS circumstance
                    FROM victim_entity_reps_postadj_orphancluster
                    WHERE entity_midpoint_day IS NOT NULL
                    AND date_trunc(
                        'month',
                        DATE '1970-01-01' + to_days(entity_midpoint_day)
                    ) >= DATE '1977-01-01'
                    AND date_trunc(
                        'month',
                        DATE '1970-01-01' + to_days(entity_midpoint_day)
                    ) < DATE '1996-01-01'
                    AND EXISTS (
                        SELECT 1
                        FROM shr_linkage_months lm
                        WHERE lm.month_start = date_trunc(
                            'month',
                            DATE '1970-01-01' + to_days(entity_midpoint_day)
                        )
                    )
                ) AS article_rows
                """
            )
        ) ^
        put_line("Running Splink linkage...") ^
        splink_dedupe_job(
            input_table=PredictionInputTableNames(("article_victims", "shr_cached")),
            settings=shr_linkage_settings,
            predict_threshold=0.5,
            deterministic_rules=SHR_DETERMINISTIC_BLOCKS,
            deterministic_recall=0.01,
            pairs_out=PairsTableName("shr_link_pairs"),
            train_first=True,
            training_blocking_rules=SHR_TRAINING_BLOCKS,
            unique_matching=True,
            unique_pairs_table=UniquePairsTableName("shr_max_weight_matches"),
            post_train_ratio_copy_comparisons=SHR_POST_TRAIN_RATIO_COPY_COMPARISONS,
            em_max_runs=1,
            visualize=False,
            splink_key=SplinkType.SHR,
        )
        >> (
            lambda result: put_line(
                "Linkage complete. "
                f"Pairs table: {result.pairs_table}, "
                f"Clusters table: {result.clusters_table}"
            )
            ^ sql_exec(
                SQL(
                    f"""
                    CREATE OR REPLACE TABLE shr_entity_matches AS
                    SELECT
                        unique_id_l AS entity_uid,
                        unique_id_r AS shr_id,
                        match_probability
                    FROM {result.pairs_table}
                    """
                )
            )
            ^ _export_shr_final_matches_excel()
            ^ _export_shr_debug_matches_excel()
            ^ put_line("Exported SHR matches to shr_matches.xlsx.")
            ^ pure(NextStep.CONTINUE)
        )
    )
