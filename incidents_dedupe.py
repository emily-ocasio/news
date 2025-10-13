"""
Dedupe incident records using Splink.
"""

from itertools import groupby
import splink.comparison_library as cl

from blocking import (
    NAMED_VICTIM_BLOCKS,
    NAMED_VICTIM_DETERMINISTIC_BLOCKS,
    NAMED_VICTIM_BLOCKS_FOR_TRAINING,
    ORPHAN_VICTIM_BLOCKS,
    ORPHAN_DETERMINISTIC_BLOCKS,
    ORPHAN_TRAINING_BLOCKS,
)
from comparison import (
    NAME_COMP,
    DATE_COMP,
    AGE_COMP,
    DIST_COMP,
    OFFENDER_COMP,
    WEAPON_COMP,
    CIRC_COMP,
)
from menuprompts import NextStep
from pymonad import (
    Environment,
    Run,
    SQL,
    Unit,
    ask,
    put_line,
    pure,
    splink_dedupe_job,
    sql_exec,
    sql_export,
    sql_query,
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


def _dedupe_named_victims(env: Environment) -> Run[Unit]:
    """
    Run initial pass of Splink deduplication on the victims_named table.
    """
    return splink_dedupe_job(
        duckdb_path=env["duckdb_path"],
        input_table="victims_named",
        settings=_settings_for_victim_dedupe(),
        predict_threshold=0.0,
        cluster_threshold=0.65,
        pairs_out="victim_pairs",
        clusters_out="victim_clusters",
        train_first=True,
        training_blocking_rules=NAMED_VICTIM_BLOCKS_FOR_TRAINING,
        deterministic_rules=NAMED_VICTIM_DETERMINISTIC_BLOCKS,
        deterministic_recall=0.8,
    ) >> (
        lambda outnames: put_line(
            f"[D] Wrote {outnames[0]} and {outnames[1]} in DuckDB."
        )
    ) ^ pure(
        unit
    )


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
                """
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
        FROM victim_cluster_counts
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
                            "FROM victim_cluster_counts c "
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


# --- Export top clusters to Excel ---
def _export_top_clusters_excel() -> Run[Unit]:
    """
    Export the top 100 clusters (one row per article/member) to an Excel file.
    """
    return (
        sql_export(
            SQL("""--sql
            WITH top AS (
              SELECT cluster_id
              FROM victim_cluster_counts
              ORDER BY member_count DESC
              LIMIT 100
            ),
            canon AS (
              SELECT
                vc.cluster_id AS cluster_id,
                mode(v.victim_surname_norm) AS canonical_surname
              FROM victim_clusters vc
              JOIN victims_cached_enh v
                ON vc.victim_row_id = v.victim_row_id
              WHERE vc.cluster_id IN (SELECT cluster_id FROM top)
                AND v.victim_surname_norm IS NOT NULL
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
              v.victim_forename_norm,
              v.victim_surname_norm,
              v.victim_age,
              v.victim_sex,
              v.lat,
              v.lon,
              COALESCE(v.geo_address_norm, '') AS address,
              COALESCE(v.offender_name_norm, '') AS offender,
              COALESCE(v.weapon, '') AS weapon
            FROM victim_cluster_counts c
            JOIN victim_clusters vc
              ON c.cluster_id = vc.cluster_id
            JOIN victims_cached_enh v
              ON vc.victim_row_id = v.victim_row_id
            LEFT JOIN canon cn
              ON c.cluster_id = cn.cluster_id
            WHERE c.cluster_id IN (SELECT cluster_id FROM top)
            ORDER BY
              cn.canonical_surname NULLS LAST,
              v.victim_surname_norm NULLS LAST,
              v.victim_forename_norm NULLS LAST,
              v.victim_name_raw
            """
            ),
            "top_clusters.xlsx",
            "Top100",
            band_by_group_col="cluster_id",
            band_wrap=2,
        )
        ^ put_line(
            "[D] Wrote top_clusters.xlsx (Top 100 clusters, one row per article)."
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
        sql_exec(
            SQL(
                """--sql
                CREATE OR REPLACE TABLE victim_entity_reps AS
                WITH agg AS (
                  SELECT
                    ve.victim_entity_id,
                    ve.city_id,
                    ve.min_event_day,
                    ve.max_event_day,
                    -- canonical attributes
                    ve.canonical_fullname,
                    ve.canonical_sex,
                    ve.canonical_race,
                    ve.canonical_ethnicity,

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
                  GROUP BY
                    ve.victim_entity_id, ve.city_id, ve.min_event_day, ve.max_event_day,
                    ve.canonical_fullname, ve.canonical_sex, ve.canonical_race,
                    ve.canonical_ethnicity
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
                    ) AS offender_surname
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

                  -- location/age
                  a.lat_centroid,
                  a.lon_centroid,
                  a.canonical_age,
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


def _create_orphans_view() -> Run[Unit]:
    """
    Create a view of victims that are not in any cluster (no names)
    """
    return (
        sql_exec(
            SQL(
                """--sql
            CREATE OR REPLACE VIEW victims_orphan AS
            SELECT *
            FROM victims_cached_enh
            WHERE victim_row_id NOT IN (
              SELECT victim_row_id FROM victim_entity_members
            );
            --AND victim_fullname_concat IS NULL;  seems redundant
        """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victims_orphan"))
        >> (lambda rows: put_line(f"[D] victims_orphan rows: {rows[0]['n']}"))
        ^ pure(unit)
    )


def _create_linkage_input_tables() -> Run[Unit]:
    """
    Create tables for linkage input: one for clustered victims, one for orphans.
    """
    return (
        sql_exec(
            SQL(
                """--sql
                -- entity_link_input: expose entity_date_precision AS date_precision
                CREATE OR REPLACE TABLE entity_link_input AS
                SELECT
                    cast(victim_entity_id AS varchar) AS unique_id,
                    city_id,
                    entity_midpoint_day AS midpoint_day,
                    incident_date,
                    entity_date_precision AS date_precision,
                    -- derived calendar fields for your blocking
                    EXTRACT(
                      YEAR FROM COALESCE(
                        incident_date,
                        date_add(DATE '1970-01-01', INTERVAL (CAST(entity_midpoint_day AS INTEGER)) DAY)
                      )
                    ) AS year,
                    EXTRACT(
                      MONTH FROM COALESCE(
                        incident_date,
                        date_add(DATE '1970-01-01', INTERVAL (CAST(entity_midpoint_day AS INTEGER)) DAY)
                      )
                    ) AS month,
                    -- NEW: all article ids from cluster as CSV
                    article_ids_csv,
                    lat_centroid AS lat,
                    lon_centroid AS lon,
                    canonical_age AS victim_age,
                    canonical_sex AS victim_sex,
                    canonical_race AS victim_race,
                    canonical_ethnicity AS victim_ethnicity,
                    mode_weapon AS weapon,
                    mode_circumstance AS circumstance,
                    offender_forename AS offender_forename_norm,
                    offender_surname  AS offender_surname_norm,
                    cast(victim_entity_id AS varchar) AS exclusion_id
                FROM victim_entity_reps;
    """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
                -- orphan_link_input: pass through date_precision from victims_orphan
                CREATE OR REPLACE TABLE orphan_link_input AS
                SELECT
                    cast(victim_row_id AS varchar) AS unique_id,
                    city_id,
                    midpoint_day,
                    incident_date,
                    date_precision,
                    article_id,
                    year,
                    month,
                    lat,
                    lon,
                    victim_age,
                    victim_sex,
                    victim_race,
                    victim_ethnicity,
                    weapon,
                    circumstance,
                    offender_forename_norm,
                    offender_surname_norm,
                    CAST(article_id AS varchar) AS article_ids_csv,
                    '' AS exclusion_id  
                FROM victims_orphan;
    """
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
            -- 1) Expand entity articles to scalar rows
            CREATE OR REPLACE TEMP VIEW _entity_articles AS
            SELECT
              eli.unique_id AS entity_uid,
              TRY_CAST(article_id_str AS BIGINT) AS article_id
            FROM entity_link_input eli,
                UNNEST(string_split(COALESCE(eli.article_ids_csv, ''), ',')) AS t(article_id_str)
            WHERE article_id_str <> '';
            -- 2) For each orphan, find all matching entities by article_id
            CREATE OR REPLACE TEMP VIEW _orphan_entity_article_matches AS
            SELECT
              o.unique_id AS orphan_uid,
              e.entity_uid
            FROM orphan_link_input o
            JOIN _entity_articles e
              ON o.article_id = e.article_id;

            -- 3) Count how many clusters match per orphan
            CREATE OR REPLACE TEMP VIEW _orphan_match_counts AS
            SELECT
              orphan_uid,
              COUNT(DISTINCT entity_uid) AS n_clusters
            FROM _orphan_entity_article_matches
            GROUP BY orphan_uid;

            -- 4) Choose an exclusion target per orphan:
            --    - if exactly one cluster: choose that one
            --    - if multiple: pick a deterministic one (min by id)
            CREATE OR REPLACE TEMP VIEW _orphan_exclusion_choice AS
            SELECT
              m.orphan_uid,
              MIN(m.entity_uid) AS chosen_entity_uid,   -- deterministic pick if multiple
              c.n_clusters
            FROM _orphan_entity_article_matches m
            JOIN _orphan_match_counts c USING (orphan_uid)
            GROUP BY m.orphan_uid, c.n_clusters;

            -- 5) Rebuild orphan_link_input with exclusion_id set
            CREATE OR REPLACE TABLE orphan_link_input AS
            SELECT
              o.* EXCLUDE (exclusion_id),
              COALESCE(c.chosen_entity_uid, o.exclusion_id) AS exclusion_id  -- stays '' if no match
            FROM orphan_link_input o
            LEFT JOIN _orphan_exclusion_choice c
              ON c.orphan_uid = o.unique_id;

            -- 6) Report: how many orphans had >1 matching clusters,
            --    and how many potential exclusions "fell through the cracks"
            --    (i.e., additional clusters beyond the chosen one)
            CREATE OR REPLACE TABLE orphan_article_exclusion_report AS
            SELECT
              COALESCE(SUM(CASE WHEN n_clusters = 1 THEN 1 ELSE 0 END), 0) 
                AS orphans_one_cluster_match,
              COALESCE(SUM(CASE WHEN n_clusters > 1 THEN 1 ELSE 0 END), 0) 
                AS orphans_multi_cluster_match,
              -- Each orphan with k>1 has (k-1) clusters
              -- not excluded by our single chosen id
              COALESCE(SUM(CASE WHEN n_clusters > 1 THEN n_clusters - 1 ELSE 0 END), 0)
                AS fell_through_pairs
            FROM _orphan_match_counts;

"""
            )
        )
        ^ sql_query(SQL("SELECT * FROM orphan_article_exclusion_report"))
        >> (
            lambda rows: put_line(
                "[D] Orphan article exclusion report: "
                f"one_cluster={rows[0]['orphans_one_cluster_match']}, "
                f"multi_cluster={rows[0]['orphans_multi_cluster_match']}, "
                f"fell_through_pairs={rows[0]['fell_through_pairs']}"
            )
        )
        ^ sql_query(
            SQL(
                """--sql
        SELECT
            (SELECT COUNT(*) FROM entity_link_input)  AS n_entities,
            (SELECT COUNT(*) FROM orphan_link_input) AS n_orphans;
    """
            )
        )
        >> (
            lambda rows: put_line(
                f"[D] entity_link_input={rows[0]['n_entities']}, "
                f"orphan_link_input={rows[0]['n_orphans']}\n"
                f"[D] Now run link_entities to link these two tables.\n"
            )
        )
        ^ pure(unit)
    )


def _link_orphans_to_entities(env: Environment) -> Run[Unit]:
    """
    Link orphan victims to existing victim entities using Splink.
    """
    return splink_dedupe_job(
        duckdb_path=env["duckdb_path"],
        input_table=["entity_link_input", "orphan_link_input"],
        settings={
            "link_type": "link_only",
            "unique_id_column_name": "unique_id",
            "blocking_rules_to_generate_predictions": ORPHAN_VICTIM_BLOCKS,
            "comparisons": [
                DATE_COMP,
                AGE_COMP,
                DIST_COMP,
                cl.ExactMatch("victim_sex"),
                WEAPON_COMP,
                CIRC_COMP,
            ],
        },
        predict_threshold=0.6,
        cluster_threshold=0.0,
        pairs_out="orphan_entity_pairs",
        clusters_out="",
        deterministic_rules=ORPHAN_DETERMINISTIC_BLOCKS,
        deterministic_recall=0.1,
        train_first=True,
        training_blocking_rules=ORPHAN_TRAINING_BLOCKS,
        do_cluster=False,
        visualize=False,
    ) >> (lambda outnames: put_line(f"[D] Wrote {outnames[0]} in DuckDB.")) ^ pure(unit)


def _debug_preview_orphans() -> Run[Unit]:
    """
    Print a quick snapshot of 20 orphans with the key fields used in blocking,
    so we can diagnose why deterministic rules might be yielding no matches.
    """
    return (
        # quick summary first
        sql_query(
            SQL(
                """--sql
                SELECT
                  COUNT(*) AS n_orphans,
                  SUM(CASE WHEN exclusion_id <> '' THEN 1 ELSE 0 END) AS n_with_exclusion,
                  SUM(CASE WHEN year IS NULL OR month IS NULL THEN 1 ELSE 0 END) AS n_missing_year_month,
                  SUM(CASE WHEN lat IS NULL OR lon IS NULL THEN 1 ELSE 0 END) AS n_missing_lat_lon
                FROM orphan_link_input;
                """
            )
        )
        >> (
            lambda rows: put_line(
                "[D] Orphan snapshot: "
                f"total={rows[0]['n_orphans']}, "
                f"with_exclusion={rows[0]['n_with_exclusion']}, "
                f"missing_year_or_month={rows[0]['n_missing_year_month']}, "
                f"missing_lat_or_lon={rows[0]['n_missing_lat_lon']}"
            )
        )
        ^
        # print 20 example rows
        sql_query(
            SQL(
                """--sql
                SELECT
                  unique_id, city_id,
                  year, month, date_precision,
                  incident_date, midpoint_day,
                  lat, lon,
                  victim_age, victim_sex, weapon, circumstance,
                  article_id,
                  exclusion_id
                FROM orphan_link_input
                ORDER BY COALESCE(year, 0) DESC,
                         COALESCE(month, 0) DESC,
                         city_id,
                         unique_id
                LIMIT 20;
                """
            )
        )
        >> (
            lambda rows: put_line(
                "[D] Top 20 orphans (key blocking fields):\n"
                + "\n".join(
                    "  "
                    f"id={r['unique_id']} city={r['city_id']} "
                    f"Y={r['year']} M={r['month']} prec={r['date_precision']} "
                    f"date={r['incident_date']} mid={r['midpoint_day']} "
                    f"lat={r['lat']} lon={r['lon']} "
                    f"age={r['victim_age']} sex={r['victim_sex']} "
                    f"weap='{r['weapon']}' circ='{r['circumstance']}' "
                    f"art={r['article_id']} excl='{r['exclusion_id']}'"
                    for r in rows
                )
            )
        )
        ^ pure(unit)
    )


def dedupe_incidents_with_splink(env) -> Run[NextStep]:
    """
    Deduplicate incident records using Splink.
    """
    return (
        _create_victims_named_table()
        ^ _dedupe_named_victims(env)
        ^ _create_cluster_tables()
        ^ _show_initial_clusters()
        ^ _export_top_clusters_excel()
        ^ _build_representative_victims()
        ^ _create_orphans_view()
        ^ _create_linkage_input_tables()
        ^ _debug_preview_orphans()
        ^ _link_orphans_to_entities(env)
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
        DIST_COMP,
        cl.ExactMatch("victim_sex"),
        OFFENDER_COMP,
        WEAPON_COMP,
        CIRC_COMP,
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
    return with_duckdb(ask() >> dedupe_incidents_with_splink)
