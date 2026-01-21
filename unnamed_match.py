"""
Match unnamed (orphan) victims to previously deduped victim clusters.
"""

import hashlib
import networkx as nx
from splink.internals import comparison_library as cl

from blocking import (
    ORPHAN_VICTIM_BLOCKS,
    ORPHAN_DETERMINISTIC_BLOCKS,
    ORPHAN_TRAINING_BLOCKS,
)

from comparison import (
    DATE_COMP_ORPHAN,
    AGE_COMP_ORPHAN,
    VICTIM_COUNT_COMP,
    DIST_COMP,
    OFFENDER_COMP,
    TF_WEAPON_COMP,
    CIRC_COMP,
    SUMMARY_COMP,
)

from menuprompts import NextStep
from splink_types import SplinkType
from pymonad import (
    Environment,
    Run,
    ask,
    pure,
    with_duckdb,
    splink_dedupe_job,
    sql_exec,
    sql_export,
    SQL,
    sql_query,
    put_line,
    Unit,
    unit
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
                    summary_vec,
                    -- derived calendar fields for your blocking
                    EXTRACT(
                      YEAR FROM COALESCE(
                        incident_date,
                        date_add(DATE '1970-01-01', INTERVAL (CAST(entity_midpoint_day AS INTEGER)) DAY)
                      )
                    ) AS year,
                    EXTRACT(
                      YEAR FROM COALESCE(
                        incident_date,
                        date_add(DATE '1970-01-01', INTERVAL (CAST(entity_midpoint_day AS INTEGER)) DAY)
                      )
                    ) AS year_block,
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
                    canonical_victim_count AS victim_count,
                    canonical_sex AS victim_sex,
                    canonical_race AS victim_race,
                    canonical_ethnicity AS victim_ethnicity,
                    canonical_fullname AS victim_fullname_norm,  -- Added: victim fullname from entities
                    canonical_offender_age AS offender_age,
                    canonical_offender_sex AS offender_sex,
                    canonical_offender_race AS offender_race,
                    canonical_offender_ethnicity AS offender_ethnicity,
                    mode_weapon AS weapon,
                    mode_circumstance AS circumstance,
                    offender_forename AS offender_forename_norm,
                    offender_surname  AS offender_surname_norm,
                    CASE
                      WHEN offender_forename IS NOT NULL AND offender_surname IS NOT NULL
                        THEN offender_forename || ' ' || offender_surname
                      WHEN offender_forename IS NOT NULL
                        THEN offender_forename
                    END AS offender_fullname_concat,
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
                    year AS year_block,
                    month,
                    lat,
                    lon,
                    victim_age,
                    victim_count,
                    victim_sex,
                    victim_race,
                    victim_ethnicity,
                    NULL AS victim_fullname_norm,  -- Added: orphans have no victim fullname
                    weapon,
                    circumstance,
                    offender_forename_norm,
                    offender_surname_norm,
                    offender_fullname_concat,
                    offender_age AS offender_age,
                    offender_sex AS offender_sex,
                    offender_race AS offender_race,
                    offender_ethnicity AS offender_ethnicity,
                    summary_vec,
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


def _link_orphans_to_entities(env: Environment) -> Run[Unit]:
    """
    Link orphan victims to existing victim entities using Splink.
    """
    return splink_dedupe_job(
        input_table=["entity_link_input", "orphan_link_input"],
        settings={
            "link_type": "link_only",
            "unique_id_column_name": "unique_id",
            "blocking_rules_to_generate_predictions": ORPHAN_VICTIM_BLOCKS,
            "comparisons": [
                DATE_COMP_ORPHAN,
                AGE_COMP_ORPHAN,
                DIST_COMP,
                cl.ExactMatch("victim_sex").configure(term_frequency_adjustments=True),
                OFFENDER_COMP,
                TF_WEAPON_COMP,
                CIRC_COMP,
                SUMMARY_COMP,
                VICTIM_COUNT_COMP,
            ],
        },
        predict_threshold=0.1,
        cluster_threshold=0.0,
        pairs_out="orphan_entity_pairs",
        clusters_out="",
        deterministic_rules=ORPHAN_DETERMINISTIC_BLOCKS,
        deterministic_recall=0.1,
        train_first=True,
        training_blocking_rules=ORPHAN_TRAINING_BLOCKS,
        do_cluster=False,
        visualize=False,
        em_max_runs=5,
        splink_key=SplinkType.ORPHAN,
    ) >> (
        lambda outnames: put_line(f"[D] Wrote {outnames[1]} in DuckDB.")
    ) ^ pure(unit)


def _integrate_orphan_matches() -> Run[Unit]:
    """
    Integrate the final orphan matches into victim_entity_reps_new.
    Creates victim_entity_reps_new (augmented from victim_entity_reps) based on the entity-orphan matching results.
    """
    def _stable_edge_weight(match_probability, entity_uid, orphan_uid) -> float:
        base = 0.0 if match_probability is None else float(match_probability)
        seed = f"{entity_uid}|{orphan_uid}".encode("utf-8")
        digest = hashlib.md5(seed).hexdigest()
        jitter = (int(digest[:8], 16) % 1000000) / 1e15
        return base + jitter

    def _max_weight_pairs_by_article(rows) -> list[tuple[str, str, int | None, float | None]]:
        by_article: dict[int | None, list[dict]] = {}
        for row in rows:
            by_article.setdefault(row["article_id"], []).append(row)

        matched: list[tuple[str, str, int | None, float | None]] = []
        for article_id, group in by_article.items():
            G: nx.Graph = nx.Graph()  # pylint: disable=C0103
            for row in group:
                entity_uid = str(row["entity_uid"])
                orphan_uid = str(row["orphan_uid"])
                prob = row["match_probability"]
                G.add_edge(
                    f"e:{entity_uid}",
                    f"o:{orphan_uid}",
                    weight=_stable_edge_weight(prob, entity_uid, orphan_uid),
                    raw_prob=prob,
                )
            if not G.edges:
                continue
            matching = nx.max_weight_matching(G, maxcardinality=True, weight="weight")
            for u, v in matching:
                if u.startswith("o:"):
                    u, v = v, u
                entity_uid = u[2:]
                orphan_uid = v[2:]
                prob = G[u][v]["raw_prob"]
                matched.append((entity_uid, orphan_uid, article_id, prob))
        return matched

    def _write_final_orphan_matches(rows) -> Run[None]:
        matches = _max_weight_pairs_by_article(rows)
        if not matches:
            return sql_exec(
                SQL(
                    """--sql
                    CREATE OR REPLACE TABLE final_orphan_matches (
                      entity_uid VARCHAR,
                      orphan_uid VARCHAR,
                      article_id BIGINT,
                      match_probability DOUBLE
                    );
                    """
                )
            )
        values = []
        for entity_uid, orphan_uid, article_id, prob in matches:
            art_sql = "NULL" if article_id is None else str(int(article_id))
            prob_sql = "NULL" if prob is None else str(float(prob))
            values.append(f"({entity_uid!r}, {orphan_uid!r}, {art_sql}, {prob_sql})")
        values_sql = ", ".join(values)
        return sql_exec(
            SQL(
                f"""--sql
                CREATE OR REPLACE TABLE final_orphan_matches AS
                SELECT * FROM (VALUES {values_sql})
                AS t(entity_uid, orphan_uid, article_id, match_probability);
                """
            )
        )

    return (
        # Drop existing new table if present (for reruns)
        sql_exec(SQL("DROP TABLE IF EXISTS victim_entity_reps_new;"))
        # Create final matches table using the validated logic
        ^ sql_query(
            SQL(
                """--sql
                SELECT
                  bp.unique_id_l AS entity_uid,
                  bp.unique_id_r AS orphan_uid,
                  oi.article_id,
                  bp.match_probability
                FROM orphan_entity_pairs bp
                JOIN orphan_link_input oi
                  ON oi.unique_id = bp.unique_id_r;
                """
            )
        )
        >> _write_final_orphan_matches
        # Create victim_entity_reps_new by augmenting victim_entity_reps with orphan data
        ^ sql_exec(
            SQL(
                """--sql
                CREATE TABLE victim_entity_reps_new AS
                SELECT * FROM victim_entity_reps;
                """
            )
        )
        # Update matched entities in victim_entity_reps_new
        ^ sql_exec(
            SQL(
                """--sql
                WITH matched_orphans AS (
                  SELECT
                    fom.entity_uid,
                    vce.lat, vce.lon, vce.midpoint_day, vce.article_id, vce.victim_count
                  FROM final_orphan_matches fom
                  JOIN victims_cached_enh vce ON vce.victim_row_id = CAST(fom.orphan_uid AS VARCHAR)
                ),
                matched_updates AS (
                  SELECT
                    entity_uid,
                    COUNT(*) AS num_orphans,
                    AVG(lat) AS avg_orphan_lat,
                    AVG(lon) AS avg_orphan_lon,
                    MIN(midpoint_day) AS min_orphan_mid,
                    MAX(midpoint_day) AS max_orphan_mid,
                    STRING_AGG(DISTINCT CAST(article_id AS VARCHAR), ',') AS orphan_article_ids,
                    MAX(victim_count) FILTER (WHERE victim_count IS NOT NULL) AS max_orphan_victim_count
                  FROM matched_orphans
                  GROUP BY entity_uid
                )
                UPDATE victim_entity_reps_new
                SET
                  cluster_size = cluster_size + COALESCE(mu.num_orphans, 0),
                  lat_centroid = CASE WHEN mu.num_orphans IS NOT NULL THEN
                    (lat_centroid * cluster_size + mu.avg_orphan_lat * mu.num_orphans) / (cluster_size + mu.num_orphans)
                  ELSE lat_centroid END,
                  lon_centroid = CASE WHEN mu.num_orphans IS NOT NULL THEN
                    (lon_centroid * cluster_size + mu.avg_orphan_lon * mu.num_orphans) / (cluster_size + mu.num_orphans)
                  ELSE lon_centroid END,
                  min_event_day = CASE WHEN mu.num_orphans IS NOT NULL THEN LEAST(min_event_day, mu.min_orphan_mid) ELSE min_event_day END,
                  max_event_day = CASE WHEN mu.num_orphans IS NOT NULL THEN GREATEST(max_event_day, mu.max_orphan_mid) ELSE max_event_day END,
                  article_ids_csv = article_ids_csv || ',' || COALESCE(mu.orphan_article_ids, ''),
                  canonical_victim_count = CASE
                    WHEN mu.max_orphan_victim_count IS NOT NULL AND canonical_victim_count IS NOT NULL
                      THEN GREATEST(canonical_victim_count, mu.max_orphan_victim_count)
                    WHEN mu.max_orphan_victim_count IS NOT NULL
                      THEN mu.max_orphan_victim_count
                    ELSE canonical_victim_count
                  END
                FROM matched_updates mu
                WHERE victim_entity_reps_new.victim_entity_id = mu.entity_uid;
                """
            )
        )
        # Insert unmatched orphans as new entities in victim_entity_reps_new
        ^ sql_exec(
            SQL(
                """--sql
                WITH unmatched_orphans AS (
                  SELECT oli.*, vce.*
                  FROM orphan_link_input oli
                  LEFT JOIN final_orphan_matches fom ON oli.unique_id = fom.orphan_uid
                  JOIN victims_orphan vo ON vo.victim_row_id = CAST(oli.unique_id AS VARCHAR)
                  JOIN victims_cached_enh vce ON vce.victim_row_id = vo.victim_row_id
                  WHERE fom.orphan_uid IS NULL
                )
                INSERT INTO victim_entity_reps_new (
                  victim_entity_id,
                  canonical_fullname,
                  canonical_sex,
                  canonical_race,
                  canonical_ethnicity,
                  city_id,
                  canonical_age,
                  canonical_victim_count,
                  mode_weapon,
                  mode_circumstance,
                  offender_fullname,
                  cluster_size,
                  article_ids_csv,
                  lat_centroid,
                  lon_centroid,
                  min_event_day,
                  max_event_day,
                  entity_date_precision,
                  incident_date,
                  entity_midpoint_day
                )
                SELECT
                  victim_row_id AS victim_entity_id,
                  NULL AS canonical_fullname,
                  victim_sex AS canonical_sex,
                  victim_race AS canonical_race,
                  victim_ethnicity AS canonical_ethnicity,
                  city_id AS city_id,
                  victim_age AS canonical_age,
                  victim_count AS canonical_victim_count,
                  weapon AS mode_weapon,
                  circumstance AS mode_circumstance,
                  offender_name AS offender_fullname,
                  1 AS cluster_size,
                  CAST(article_id AS VARCHAR) AS article_ids_csv,
                  lat AS lat_centroid,
                  lon AS lon_centroid,
                  midpoint_day AS min_event_day,
                  midpoint_day AS max_event_day,
                  date_precision AS entity_date_precision,
                  incident_date,
                  midpoint_day AS entity_midpoint_day
                FROM unmatched_orphans;
                """
            )
        )
        # Check if there are matched orphans; if so, recalculate midpoint for affected entities
        ^ sql_query(SQL("SELECT COUNT(DISTINCT entity_uid) AS n_affected FROM final_orphan_matches"))
        >> (
            lambda rows: (
                put_line(f"[D] Checking for matched orphans: {rows[0]['n_affected']} affected entities.")
                ^ (
                    sql_exec(SQL("CREATE TEMP TABLE affected_entities AS SELECT DISTINCT entity_uid FROM final_orphan_matches;"))
                    ^ sql_exec(
                        SQL(
                            """--sql
                            CREATE TEMP TABLE all_members_temp AS
                            SELECT
                                m.victim_entity_id,
                                m.date_precision,
                                m.incident_date,
                                m.midpoint_day
                            FROM victim_entity_members m
                            JOIN affected_entities ae ON m.victim_entity_id = ae.entity_uid
                            UNION ALL
                            SELECT
                                fom.entity_uid AS victim_entity_id,
                                vce.date_precision,
                                vce.incident_date,
                                vce.midpoint_day
                            FROM final_orphan_matches fom
                            JOIN victims_cached_enh vce ON vce.victim_row_id = CAST(fom.orphan_uid AS VARCHAR);
                            """
                        )
                    )
                    ^ sql_exec(
                        SQL(
                            """--sql
                            CREATE TEMP TABLE recomputed_agg AS
                            WITH agg AS (
                              SELECT
                                victim_entity_id,
                                count_if(date_precision = 'day') AS n_day,
                                count_if(date_precision = 'month') AS n_month,
                                count_if(date_precision = 'year') AS n_year,
                                mode(incident_date) FILTER (WHERE date_precision = 'day' AND incident_date IS NOT NULL) AS mode_day_date,
                                mode(midpoint_day) FILTER (WHERE date_precision = 'month') AS mode_month_mid,
                                mode(midpoint_day) FILTER (WHERE date_precision = 'year') AS mode_year_mid
                              FROM all_members_temp
                              GROUP BY victim_entity_id
                            )
                            SELECT
                              victim_entity_id,
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
                              ) AS entity_midpoint_day
                            FROM agg;
                            """
                        )
                    )
                    ^ sql_exec(
                        SQL(
                            """--sql
                            UPDATE victim_entity_reps_new
                            SET
                              entity_date_precision = ra.entity_date_precision,
                              incident_date = ra.incident_date,
                              entity_midpoint_day = ra.entity_midpoint_day
                            FROM recomputed_agg ra
                            WHERE victim_entity_reps_new.victim_entity_id = ra.victim_entity_id;
                            """
                        )
                    )
                    ^ put_line("[D] Recalculated and updated midpoints for affected entities.")
                )
                if rows[0]['n_affected'] > 0
                else put_line("[D] No matched orphans; skipping midpoint recalculation.")
            )
        )
        ^ pure(unit)
    )


def _export_orphan_matches_debug_excel() -> Run[Unit]:
    """
    Build a single worksheet that lists every victim entity and every orphan.

    Matching logic:
      - Use final_orphan_matches derived from max-weight bipartite matching
        per orphan article.

    Output:
      - Exactly one row per entity, one per orphan.
      - match_id = 'match_<entity_uid>' for matched groups, otherwise 'entity_<...>' or 'orphan_<...>'.
      - band_key = 0 (unmatched entity), 1 (unmatched orphan), 2 (matched group).
      - Ordering uses the entity midpoint for matched groups; otherwise row’s own midpoint.
    """
    return (
        sql_export(
            SQL(
                """--sql
            WITH orphan_choice AS (
              SELECT
                o.*,
                fm.entity_uid,
                fm.match_probability
              FROM orphan_link_input o
              LEFT JOIN final_orphan_matches fm
                ON o.unique_id = fm.orphan_uid
            ),
            -- Compute match_id for entities: match_<entity_uid> if they have ≥1 orphans, else entity_<id>
            entity_with_match AS (
              SELECT
                e.*,
                CASE WHEN EXISTS (SELECT 1 FROM final_orphan_matches fm WHERE fm.entity_uid = e.unique_id)
                    THEN CONCAT('match_', e.unique_id)
                    ELSE CONCAT('entity_', e.unique_id)
                END AS match_id
              FROM entity_link_input e
            ),
            combined AS (
              -- Entity rows
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
                e.city_id, e.year, e.month,
                e.lat, e.lon,
                e.victim_age, e.victim_sex, e.victim_race, e.victim_ethnicity,
                e.victim_fullname_norm,  -- Added
                e.weapon, e.circumstance,
                e.offender_forename_norm, e.offender_surname_norm,
                e.victim_count,
                CAST(NULL AS DOUBLE) AS match_probability
              FROM entity_with_match e

              UNION ALL

              -- Orphan rows (both matched and unmatched)
              -- (inherit entity midpoint for ordering if matched)
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
                o.city_id, o.year, o.month,
                o.lat, o.lon,
                o.victim_age, o.victim_sex, o.victim_race, o.victim_ethnicity,
                o.victim_fullname_norm,  -- Added
                o.weapon, o.circumstance,
                o.offender_forename_norm, o.offender_surname_norm,
                o.victim_count,
                o.match_probability
              FROM orphan_choice o
              LEFT JOIN entity_link_input e
                ON e.unique_id = o.entity_uid
            )
            SELECT
              rec_type,
              match_id,
              CASE WHEN match_id LIKE 'match_%' THEN 2
                  WHEN rec_type = 'entity' THEN 0
                  ELSE 1
              END AS band_key,
              group_midpoint_day,
              uid,
              article_id,
                city_id, year, month,
              date_precision, incident_date, midpoint_day,
              lat, lon,
              victim_age, victim_sex, victim_race, victim_ethnicity,
              victim_fullname_norm,  -- Added
              weapon, circumstance,
              offender_forename_norm, offender_surname_norm,
              victim_count,
              match_probability,
              article_ids_csv
            FROM combined
            ORDER BY
              group_midpoint_day NULLS LAST,
              match_id,
              CASE rec_type WHEN 'entity' THEN 0 ELSE 1 END,
              uid
            """
            ),
            "orphan_matches_final.xlsx",
            "Matches",
            band_by_group_col="band_key",
            band_wrap=3,
        )
        ^ put_line("[D] Wrote orphan_matches_final.xlsx (final entities + orphans after integration).")
        ^ pure(unit)
    )


def export_final_victim_entities_excel() -> Run[Unit]:
    """
    Export the final victim_entity_reps_new table to Excel with color coding:
    - 0: unmatched entities (original entities not matched to orphans)
    - 1: entities newly matched to orphans
    - 2: singleton orphans that became new entities
    """
    return (
        sql_export(
            SQL(
                """--sql
                SELECT
                  vern.*,
                  CASE
                    WHEN vern.victim_entity_id IN (SELECT victim_entity_id FROM victim_entity_reps) THEN
                      CASE WHEN vern.victim_entity_id IN (SELECT DISTINCT entity_uid FROM final_orphan_matches) THEN 'newly_matched'
                           ELSE 'unmatched_entity'
                      END
                    ELSE 'singleton_orphan'
                  END AS category,
                  CASE
                    WHEN vern.victim_entity_id IN (SELECT victim_entity_id FROM victim_entity_reps) THEN
                      CASE WHEN vern.victim_entity_id IN (SELECT DISTINCT entity_uid FROM final_orphan_matches) THEN 1
                           ELSE 0
                      END
                    ELSE 2
                  END AS band_key
                FROM victim_entity_reps_new vern
                ORDER BY entity_midpoint_day
                """
            ),
            "final_victim_entities.xlsx",
            "Entities",
            band_by_group_col="band_key",
            band_wrap=3,
        )
        ^ put_line("[D] Wrote final_victim_entities.xlsx (final victim entities with color coding).")
        ^ pure(unit)
    )


def match_orphans_with_splink(env: Environment) -> Run[NextStep]:
    """
    Match unnamed victims with existing victim clusters using SPLINK.
    """
    # Placeholder for the actual matching logic
    # This would involve querying the database for orphan victims,
    # applying the SPLINK algorithm, and updating the database with matches.

    # For now, we just return a NextStep indicating completion.
    return (
        _create_orphans_view()
        ^ _create_linkage_input_tables()
        ^ _debug_preview_orphans()
        ^ _link_orphans_to_entities(env)
        ^ _integrate_orphan_matches()
        ^ _export_orphan_matches_debug_excel()
        ^ export_final_victim_entities_excel()
        ^ pure(NextStep.CONTINUE)
    )


def match_unnamed_victims() -> Run[NextStep]:
    """
    Entry point for controller to match unnamed victims.
    """
    return with_duckdb(ask() >> match_orphans_with_splink)
