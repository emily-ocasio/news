"""
Dedupe incident records using Splink.
"""

from itertools import groupby
import splink.internals.comparison_library as cl

from blocking import (
    NAMED_VICTIM_BLOCKS,
    NAMED_VICTIM_DETERMINISTIC_BLOCKS,
    NAMED_VICTIM_BLOCKS_FOR_TRAINING,
)
from comparison import (
    NAME_COMP,
    DATE_COMP,
    AGE_COMP,
    VICTIM_COUNT_COMP,
    DIST_COMP_NEW as DIST_COMP,
    OFFENDER_COMP,
    OFFENDER_AGE_COMP,
    OFFENDER_SEX_COMP,
    WEAPON_COMP,
    CIRC_COMP,
    SUMMARY_COMP
)
from menuprompts import NextStep
from splink_types import SplinkType
from cluster_compare import ClusterCompareRequest, compare_cluster_tables
from pymonad import (
    Run,
    SQL,
    Unit,
    put_line,
    pure,
    PredictionInputTableName,
    PairsTableName,
    ClustersTableName,
    BlockedPairsTableName,
    DoNotLinkTableName,
    splink_dedupe_job,
    sql_exec,
    sql_export,
    sql_query,
    sql_import,
    unit,
    with_duckdb,
)

def _create_victims_named_table() -> Run[Unit]:
    """
    Create a table of victims that have name present.
    This will be used for the initial Splink deduplication step.
    """
    return (
        sql_exec(
            SQL(
                """--sql
        CREATE OR REPLACE TABLE victims_named AS
        SELECT
          vce.*,
          vce.year AS year_block,
          CAST(vce.article_id AS varchar) AS exclusion_id
        FROM victims_cached_enh vce
        WHERE vce.victim_forename_norm IS NOT NULL
           OR vce.victim_surname_norm IS NOT NULL;
        """
            )
        )
        ^ put_line("[D] Created victims_named table.")
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victims_named"))
        >> (lambda rows: put_line(f"[D] victims_named rows: {rows[0]['n']}"))
        ^ pure(unit)
    )


def _dedupe_named_victims(_: Unit) -> Run[Unit]:
    """
    Run initial pass of Splink deduplication on the victims_named table.
    """
    return splink_dedupe_job(
        input_table=PredictionInputTableName("victims_named"),
        settings=_settings_for_victim_dedupe(),
        predict_threshold=0.1,
        cluster_threshold=0.85,
        pairs_out=PairsTableName("victim_pairs"),
        clusters_out=ClustersTableName("victim_clusters"),
        train_first=True,
        training_blocking_rules=NAMED_VICTIM_BLOCKS_FOR_TRAINING,
        deterministic_rules=NAMED_VICTIM_DETERMINISTIC_BLOCKS,
        deterministic_recall=0.8,
        em_max_runs=1,
        visualize=False,
        splink_key=SplinkType.DEDUP,
        do_not_link_table=DoNotLinkTableName("victim_cluster_exclusion"),
        blocked_pairs_out=BlockedPairsTableName("victim_cluster_blocked_edges"),
    ) >> (
        lambda outnames: put_line(
            f"[D] Wrote {outnames[1]} and {outnames[2]} in DuckDB."
        )
    ) ^ pure(unit)


def _create_cluster_tables() -> Run[Unit]:
    """
    Create tables for victim clusters and their members.
    """
    return (
        sql_exec(
            SQL(
                """--sql
            CREATE OR REPLACE TABLE victim_entity_members AS
            SELECT
            c.cluster_id AS victim_entity_id,
            v.*
            FROM victim_clusters AS c
            JOIN victims_cached_enh AS v
            ON v.victim_row_id = c.victim_row_id;
        """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
            CREATE OR REPLACE VIEW victim_entities AS
            SELECT
            victim_entity_id,
            mode(victim_name_norm)
                FILTER (WHERE victim_fullname_concat IS NOT NULL) AS canonical_fullname,
            mode(victim_sex)
                FILTER (WHERE victim_sex IS NOT NULL) AS canonical_sex,
            mode(victim_race)
                FILTER (WHERE victim_race IS NOT NULL) AS canonical_race,
            mode(victim_ethnicity)
                FILTER (WHERE victim_ethnicity IS NOT NULL) AS canonical_ethnicity,
            min(event_start_day) AS min_event_day,
            max(event_end_day)   AS max_event_day,
            mode(incident_date) 
                FILTER (WHERE incident_date IS NOT NULL) AS incident_date,
            mode(city_id)   AS city_id
            FROM victim_entity_members
            GROUP BY victim_entity_id;
        """
            )
        )
        ^ put_line("[D] Created victim_entities view.")
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victim_entities"))
        >> (lambda rows: put_line(f"[D] victim_entities rows: {rows[0]['n']}"))
        ^ pure(unit)
    )


def _show_initial_clusters() -> Run[Unit]:
    """
    Show the top victim clusters by member count, with details."""
    return (
        sql_query(
            SQL(
                """
        SELECT cluster_id, member_count
        FROM victim_clusters_counts
        ORDER BY member_count DESC, cluster_id
        LIMIT 100;
    """
            )
        )
        >> (
            lambda top_clusters: (
                # If no top clusters, just print a message; otherwise fetch members
                put_line("[D] Top 100 victim clusters by member count:")
                ^ (
                    sql_query(
                        SQL(
                            "SELECT c.cluster_id, c.member_count, "
                            "v.victim_row_id, "
                            "v.article_id, v.incident_date, v.victim_name_raw, "
                            "v.victim_forename_norm, v.victim_surname_norm, "
                            "v.midpoint_day, v.date_precision, "
                            "v.victim_age, v.victim_sex, "
                            "v.lat, v.lon, "
                            "coalesce(v.geo_address_norm, '') AS address, "
                            "coalesce(v.offender_name_norm, '') AS offender, "
                            "coalesce(v.weapon, '') AS weapon "
                            "FROM victim_clusters_counts c "
                            "JOIN victim_clusters vc ON c.cluster_id = vc.cluster_id "
                            "JOIN victims_cached_enh v "
                            "ON vc.victim_row_id = v.victim_row_id "
                            "WHERE c.cluster_id IN ("
                            + (
                                ",".join(f"'{r['cluster_id']}'" for r in top_clusters)
                                if len(top_clusters) > 0
                                else "NULL"
                            )
                            + ") "
                            "ORDER BY c.member_count DESC, c.cluster_id, v.article_id;"
                        )
                    )
                    >> (
                        lambda member_rows: put_line(
                            "\n\n".join(
                                # Build a block per cluster
                                (
                                    f"Cluster {cid} (members={mc}):\n"
                                    + "\n".join(
                                        f"  id={m['victim_row_id']} "
                                        f"{m['incident_date']} "
                                        f"midp={m['midpoint_day']} "
                                        f"({m['date_precision']}) "
                                        f"'{m['victim_forename_norm']} "
                                        f"{m['victim_surname_norm']}' "
                                        f"{m['victim_age']} yo "
                                        f"{m['victim_sex']} "
                                        f"'{m['address']}' "
                                        f"off='{m['offender']}' "
                                        f"weap='{m['weapon']}' "
                                        for m in members
                                    )
                                )
                                for cid, mc, members, _ in _cluster_blocks(member_rows)
                            )
                        )
                    )
                )
            )
        )
        ^ pure(unit)
    )


def _build_representative_victims() -> Run[Unit]:
    """
    Build a table of representative victims for each cluster with clarified rules:
      - Date logic (day > month > year) with incident_date only for 'day' precision
      - weapon: grouped-mode with tie-break (handgun > rifle > shotgun > firearm),
                excluding NULL and 'unknown'
      - circumstance: mode excluding NULL and 'undetermined'
      - canonical_age: mode(victim_age) excluding NULL
      - offender: pick mode(offender_fullname_concat) and corresponding forename/surname
    """
    return (
        sql_exec(SQL(
            """--
            -- This unnests summary vectors and averages them per component
            -- to create an entity-level canonical summary vector per victim_entity.
            CREATE OR REPLACE TABLE entity_summary_vectors AS
            WITH per_comp AS (
                SELECT
                    m.victim_entity_id,
                    i.generate_series AS comp_idx,
                    m.summary_vec[i.generate_series] AS val
                FROM victim_entity_members m
                CROSS JOIN generate_series(1, 384) AS i
            ),
            avg_per_component AS (
                SELECT
                    victim_entity_id,
                    comp_idx,
                    AVG(val) AS avg_val
                FROM per_comp
                GROUP BY victim_entity_id, comp_idx
            )
            SELECT
                victim_entity_id,
                CAST(
                  array_agg(avg_val ORDER BY comp_idx)
                  AS DOUBLE[384]
                ) AS entity_summary_vec
            FROM avg_per_component
            GROUP BY victim_entity_id;
            """
        ))
        ^ sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE victim_entity_reps AS
                WITH agg AS (
                  SELECT
                    ve.victim_entity_id,
                    ve.city_id,
                    esv.entity_summary_vec AS summary_vec,
                    ve.min_event_day,
                    ve.max_event_day,
                    -- canonical attributes
                    ve.canonical_fullname,
                    ve.canonical_sex,
                    ve.canonical_race,
                    ve.canonical_ethnicity,
                    -- canonical offender attributes
                    -- counts by precision
                    count_if(m.date_precision = 'day')   AS n_day,
                    count_if(m.date_precision = 'month') AS n_month,
                    count_if(m.date_precision = 'year')  AS n_year,

                    -- mode values by precision
                    mode(m.incident_date)
                      FILTER (WHERE m.date_precision = 'day' AND m.incident_date IS NOT NULL)
                      AS mode_day_date,
                    mode(m.midpoint_day)
                      FILTER (WHERE m.date_precision = 'month')
                      AS mode_month_mid,
                    mode(m.midpoint_day)
                      FILTER (WHERE m.date_precision = 'year')
                      AS mode_year_mid,

                    -- stats
                    avg(m.lat)  FILTER (WHERE m.lat IS NOT NULL) AS lat_centroid,
                    avg(m.lon)  FILTER (WHERE m.lon IS NOT NULL) AS lon_centroid,

                    -- canonical age: mode excluding NULLs
                    mode(m.victim_age) FILTER (WHERE m.victim_age IS NOT NULL) AS canonical_age,
                    max(m.victim_count) FILTER (WHERE m.victim_count IS NOT NULL) AS canonical_victim_count,
                    max(m.offender_count) FILTER (WHERE m.offender_count IS NOT NULL) AS canonical_offender_count,
                    -- keep these if you still want distribution context
                    avg(m.victim_age) FILTER (WHERE m.victim_age IS NOT NULL) AS avg_age,
                    min(m.victim_age) FILTER (WHERE m.victim_age IS NOT NULL) AS min_age,
                    max(m.victim_age) FILTER (WHERE m.victim_age IS NOT NULL) AS max_age,

                    -- circumstance: mode excluding NULL + 'undetermined'
                    mode(m.circumstance) FILTER (
                      WHERE m.circumstance IS NOT NULL AND lower(m.circumstance) <> 'undetermined'
                    ) AS mode_circumstance,

                    count(*) AS cluster_size
                  FROM victim_entities ve
                  JOIN victim_entity_members m USING (victim_entity_id)
                  LEFT JOIN entity_summary_vectors esv
                    ON ve.victim_entity_id = esv.victim_entity_id
                  -- Include redundant fields in GROUP BY for clarity, as they are functionally dependent on victim_entity_id
                  GROUP BY ve.victim_entity_id, ve.city_id, esv.entity_summary_vec, ve.min_event_day, ve.max_event_day, ve.canonical_fullname, ve.canonical_sex, ve.canonical_race, ve.canonical_ethnicity
                ),
                -- -------- Weapon grouped-mode logic --------
                -- 1) Clean + filter weapons (exclude NULL/unknown)
                weapon_clean AS (
                  SELECT
                    victim_entity_id,
                    lower(trim(weapon)) AS weapon
                  FROM victim_entity_members
                  WHERE weapon IS NOT NULL AND lower(weapon) <> 'unknown'
                ),
                -- 2) Map semi-equivalent terms into a common group; others are their own group
                weapon_map AS (
                  SELECT * FROM (
                    VALUES
                      ('handgun','firearm_family', 1),
                      ('rifle'  ,'firearm_family', 2),
                      ('shotgun','firearm_family', 3),
                      ('firearm','firearm_family', 4)
                  ) AS t(weapon, grp, pri)
                ),
                weapon_grouped AS (
                  SELECT
                    wc.victim_entity_id,
                    wc.weapon,
                    COALESCE(wm.grp, wc.weapon) AS grp,
                    COALESCE(wm.pri, 999)        AS pri    -- non-family terms get low priority within their singleton
                  FROM weapon_clean wc
                  LEFT JOIN weapon_map wm
                    ON wc.weapon = wm.weapon
                ),
                -- 3) Count per (entity, group) and per (entity, group, weapon)
                weapon_group_counts AS (
                  SELECT victim_entity_id, grp, COUNT(*) AS grp_cnt
                  FROM weapon_grouped
                  GROUP BY victim_entity_id, grp
                ),
                weapon_term_counts AS (
                  SELECT victim_entity_id, grp, weapon, pri, COUNT(*) AS term_cnt
                  FROM weapon_grouped
                  GROUP BY victim_entity_id, grp, weapon, pri
                ),
                -- 4) Pick top group for each entity
                weapon_group_winner AS (
                  SELECT victim_entity_id, grp, grp_cnt,
                         ROW_NUMBER() OVER (
                           PARTITION BY victim_entity_id
                           ORDER BY grp_cnt DESC, grp
                         ) AS rn
                  FROM weapon_group_counts
                ),
                winning_group AS (
                  SELECT victim_entity_id, grp, grp_cnt
                  FROM weapon_group_winner
                  WHERE rn = 1
                ),
                -- 5) Within winning group, pick most common weapon; tie-break by priority (handgun>rifle>shotgun>firearm)
                weapon_choice AS (
                  SELECT
                    tc.victim_entity_id,
                    tc.weapon AS canonical_weapon,
                    ROW_NUMBER() OVER (
                      PARTITION BY tc.victim_entity_id
                      ORDER BY tc.term_cnt DESC, tc.pri ASC, tc.weapon
                    ) AS rn
                  FROM weapon_term_counts tc
                  JOIN winning_group wg
                    ON tc.victim_entity_id = wg.victim_entity_id
                   AND tc.grp = wg.grp
                ),
                weapon_canonical AS (
                  SELECT victim_entity_id, canonical_weapon
                  FROM weapon_choice
                  WHERE rn = 1
                ),
                -- -------- Offender modal selection (unchanged) --------
                offender_counts AS (
                  SELECT
                    victim_entity_id,
                    offender_fullname_concat AS offender_fullname,
                    COUNT(*) AS cnt
                  FROM victim_entity_members
                  WHERE offender_fullname_concat IS NOT NULL
                  GROUP BY 1,2
                ),
                offender_ranked AS (
                  SELECT
                    victim_entity_id, offender_fullname, cnt,
                    row_number() OVER (
                      PARTITION BY victim_entity_id
                      ORDER BY cnt DESC, offender_fullname
                    ) AS rn
                  FROM offender_counts
                ),
                offender_top AS (
                  SELECT victim_entity_id, offender_fullname
                  FROM offender_ranked
                  WHERE rn = 1
                ),
                offender_names AS (
                  SELECT
                    ot.victim_entity_id,
                    ot.offender_fullname,
                    mode(m.offender_forename_norm) FILTER (
                      WHERE m.offender_fullname_concat = ot.offender_fullname
                        AND m.offender_forename_norm IS NOT NULL
                    ) AS offender_forename,
                    mode(m.offender_surname_norm) FILTER (
                      WHERE m.offender_fullname_concat = ot.offender_fullname
                        AND m.offender_surname_norm IS NOT NULL
                    ) AS offender_surname,
                    mode(m.offender_age) FILTER (
                      WHERE m.offender_fullname_concat = ot.offender_fullname
                        AND m.offender_age IS NOT NULL
                    ) AS offender_age,
                    mode(m.offender_sex) FILTER (
                      WHERE m.offender_fullname_concat = ot.offender_fullname
                        AND m.offender_sex IS NOT NULL
                    ) AS offender_sex,
                    mode(m.offender_race) FILTER (
                      WHERE m.offender_fullname_concat = ot.offender_fullname
                        AND m.offender_race IS NOT NULL
                    ) AS offender_race,
                    mode(m.offender_ethnicity) FILTER (
                      WHERE m.offender_fullname_concat = ot.offender_fullname
                        AND m.offender_ethnicity IS NOT NULL
                    ) AS offender_ethnicity
                  FROM offender_top ot
                  JOIN victim_entity_members m USING (victim_entity_id)
                  GROUP BY ot.victim_entity_id, ot.offender_fullname
                ),
                entity_articles AS (
                  SELECT
                    victim_entity_id,
                    string_agg(DISTINCT cast(article_id AS varchar), ',') AS article_ids_csv
                  FROM victim_entity_members
                  GROUP BY victim_entity_id
                )

                SELECT
                  a.victim_entity_id,
                  a.city_id,
                  a.min_event_day,
                  a.max_event_day,
                  a.summary_vec,

                  -- choose precision: day > month > year
                  CASE
                    WHEN a.n_day   > 0 THEN 'day'
                    WHEN a.n_month > 0 THEN 'month'
                    ELSE 'year'
                  END AS entity_date_precision,

                  -- incident_date only when precision is 'day'
                  CASE
                    WHEN a.n_day > 0 THEN a.mode_day_date
                    ELSE NULL
                  END AS incident_date,

                  -- midpoint per chosen precision
                  CAST(
                    CASE
                      WHEN a.n_day > 0 THEN date_diff('day', DATE '1970-01-01', a.mode_day_date)
                      WHEN a.n_month > 0 THEN a.mode_month_mid
                      ELSE a.mode_year_mid
                    END AS INTEGER
                  ) AS entity_midpoint_day,
                  -- canonical attributes
                  a.canonical_fullname,
                  a.canonical_sex,
                  a.canonical_race,
                  a.canonical_ethnicity,

                  -- canonical offender attributes
                  onm.offender_age AS canonical_offender_age,
                  onm.offender_sex AS canonical_offender_sex,
                  onm.offender_race AS canonical_offender_race,
                  onm.offender_ethnicity AS canonical_offender_ethnicity,
                  a.canonical_offender_count,

                  -- location/age
                  a.lat_centroid,
                  a.lon_centroid,
                  a.canonical_age,
                  a.canonical_victim_count,
                  a.avg_age,
                  a.min_age,
                  a.max_age,

                  -- weapon (grouped mode) and circumstance (filtered mode)
                  wc.canonical_weapon AS mode_weapon,
                  a.mode_circumstance,

                  -- offender (mode fullname + corresponding parts)
                  onm.offender_fullname,
                  onm.offender_forename,
                  onm.offender_surname,

                  a.cluster_size,
                  ea.article_ids_csv
                FROM agg a
                LEFT JOIN weapon_canonical wc
                  ON wc.victim_entity_id = a.victim_entity_id
                LEFT JOIN offender_names onm
                  ON onm.victim_entity_id = a.victim_entity_id
                LEFT JOIN entity_articles ea
                  ON ea.victim_entity_id = a.victim_entity_id;
                """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victim_entity_reps"))
        >> (lambda rows: put_line(f"[D] victim_entity_reps rows: {rows[0]['n']}"))
        ^ pure(unit)
    )


def _export_final_clusters_excel() -> Run[Unit]:
    """
    Export all clusters (one row per article/member) to an Excel file.
    """
    final_clusters_select = SQL(
        """--sql
    WITH canon AS (
      SELECT
        vc.cluster_id AS cluster_id,
        mode(v.victim_surname_norm) AS canonical_surname
      FROM victim_clusters vc
      JOIN victims_cached_enh v
        ON vc.victim_row_id = v.victim_row_id
      WHERE v.victim_surname_norm IS NOT NULL
      GROUP BY vc.cluster_id
    )
    SELECT
      c.cluster_id,
      c.member_count,
      cn.canonical_surname,
      v.article_id,
      v.victim_row_id,
      v.city_id,
      v.incident_date,
      v.midpoint_day,
      v.date_precision,
      v.victim_name_raw,
      v.victim_fullname_concat,
      v.victim_forename_norm,
      v.victim_surname_norm,
      v.victim_age,
      v.victim_count,
      v.offender_count,
      v.victim_sex,
      v.lat,
      v.lon,
      v.address_type,
      v.geo_address_short,
      v.geo_address_short_2,
      v.geo_score,
      COALESCE(v.geo_address_norm, '') AS address,
      COALESCE(v.offender_name_norm, '') AS offender,
      COALESCE(v.weapon, '') AS weapon
    FROM victim_clusters_counts c
    JOIN victim_clusters vc
      ON c.cluster_id = vc.cluster_id
    JOIN victims_cached_enh v
      ON vc.victim_row_id = v.victim_row_id
    LEFT JOIN canon cn
      ON c.cluster_id = cn.cluster_id
    ORDER BY
      cn.canonical_surname NULLS LAST,
      c.cluster_id,
      v.victim_surname_norm NULLS LAST,
      v.victim_forename_norm NULLS LAST,
      v.victim_name_raw,
      v.article_id
    """
    )

    def _register_previous_final_clusters_if_exists() -> Run[bool]:
        return sql_query(
            SQL(
                """--sql
            SELECT CASE
              WHEN COUNT(*) > 0 THEN TRUE
              ELSE FALSE
            END AS exists_flag
            FROM glob('final_clusters.xlsx');
            """
            )
        ) >> (
            lambda rows: (
                (
                    sql_import("final_clusters.xlsx", "final_clusters_prev")
                    ^ put_line(
                        "[D] Loaded existing final_clusters.xlsx into final_clusters_prev."
                    )
                    ^ pure(True)
                )
                if rows and rows[0]["exists_flag"]
                else (put_line("[D] No existing final_clusters.xlsx found.") ^ pure(False))
            )
        )

    def _maybe_export_diffs(has_prev: bool) -> Run[Unit]:
        if not has_prev:
            return pure(unit)
        return (
            compare_cluster_tables(
                ClusterCompareRequest(
                    left_table="final_clusters_prev",
                    right_table="final_clusters_current",
                    member_id_col="victim_row_id",
                    output_table="cluster_diffs",
                )
            )
            >> (
                lambda _: sql_export(
                    SQL(
                        """--sql
                    SELECT
                      *,
                      concat(cast(source AS varchar), '::', cast(cluster_id AS varchar)) AS __band_group
                    FROM cluster_diffs
                    ORDER BY
                      source,
                      canonical_surname NULLS LAST,
                      cluster_id,
                      victim_surname_norm NULLS LAST,
                      victim_forename_norm NULLS LAST,
                      victim_name_raw,
                      article_id
                    """
                    ),
                    "cluster_diffs.xlsx",
                    "ClusterDiffs",
                    band_by_group_col="__band_group",
                    band_wrap=2,
                )
            )
            ^ put_line("[D] Wrote cluster_diffs.xlsx.")
            ^ pure(unit)
        )

    return (
        _register_previous_final_clusters_if_exists()
        >> (
            lambda has_prev: (
                sql_exec(
                    SQL(
                        f"""--sql
                CREATE OR REPLACE TABLE final_clusters_current AS
                {final_clusters_select}
                """
                    )
                )
                ^ _maybe_export_diffs(has_prev)
            )
        )
        ^ sql_export(
            final_clusters_select,
            "final_clusters.xlsx",
            "FinalClusters",
            band_by_group_col="cluster_id",
            band_wrap=2,
        )
        ^ put_line(
            "[D] Wrote final_clusters.xlsx (All clusters, one row per article)."
        )
        ^ pure(unit)
    )


def dedupe_incidents_with_splink() -> Run[NextStep]:
    """
    Deduplicate incident records using Splink.
    """
    return (
        _create_victims_named_table()
        >> _dedupe_named_victims
        ^ _create_cluster_tables()
        ^ _build_representative_victims()
        ^ _export_final_clusters_excel()
        ^ pure(NextStep.CONTINUE)
    )


def _cluster_blocks(rows):
    """
    Given a sequence of member_rows (dicts with 'cluster_id' and 'member_count'),
    return a list of tuples (cluster_id, member_count, [rows...]) grouped by cluster_id.
    """
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: (r["cluster_id"], r["article_id"]))
    blocks = []
    for key, grp in groupby(sorted_rows, key=lambda r: r["cluster_id"]):
        group_list = list(grp)
        member_count = group_list[0].get("member_count", 0)
        last_name = group_list[0].get("victim_surname_norm", "")
        blocks.append((key, member_count, group_list, last_name))
    sorted_blocks = sorted(blocks, key=lambda b: b[3])
    return sorted_blocks


def _settings_for_victim_dedupe() -> dict:

    comparisons = [
        NAME_COMP,
        DATE_COMP,
        AGE_COMP,
        VICTIM_COUNT_COMP,
        DIST_COMP,
        cl.ExactMatch("victim_sex"),
        OFFENDER_COMP,
        OFFENDER_AGE_COMP,
        OFFENDER_SEX_COMP,
        WEAPON_COMP,
        CIRC_COMP,
        SUMMARY_COMP
        # same_article_comp,  # Add the new comparison here
    ]

    return {
        "link_type": "dedupe_only",
        "unique_id_column_name": "victim_row_id",
        "blocking_rules_to_generate_predictions": NAMED_VICTIM_BLOCKS,
        "comparisons": comparisons,
    }


def dedupe_incidents() -> Run[NextStep]:
    """
    Entry point for controller to deduplicate incidents
    """
    return with_duckdb(dedupe_incidents_with_splink())
