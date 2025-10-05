"""
Dedupe incident records using Splink.
"""

from itertools import groupby
import splink.comparison_library as cl

from blocking import (
    NAMED_VICTIM_BLOCKS,
    NAMED_VICTIM_DETERMINISTIC_BLOCKS,
    NAMED_VICTIM_BLOCKS_FOR_TRAINING,
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
    Run,
    ask,
    put_line,
    pure,
    sql_query,
    sql_exec,
    SQL,
    splink_dedupe_job,
    with_duckdb,
)


def dedupe_incidents_with_splink(env) -> Run[NextStep]:
    """
    Deduplicate incident records using Splink.
    """
    return (
        sql_exec(
            SQL(
                """--sql
            CREATE OR REPLACE VIEW victims_named AS
            SELECT *
            FROM victims_cached_enh
            WHERE victim_forename_norm IS NOT NULL
            OR victim_surname_norm IS NOT NULL;
        """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victims_named"))
        >> (lambda rows: put_line(f"[D] victims_named rows: {rows[0]['n']}"))
        ^ splink_dedupe_job(
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
        )
        >> (
            lambda outnames: put_line(
                f"[D] Wrote {outnames[0]} and {outnames[1]} in DuckDB."
            )
        )
        ^ sql_exec(
            SQL(
                """--sql
            CREATE OR REPLACE VIEW victim_entity_members AS
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
            mode(city_id)   AS city_id
            FROM victim_entity_members
            GROUP BY victim_entity_id;
        """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victim_entities"))
        >> (lambda rows: put_line(f"[D] victim_entities rows: {rows[0]['n']}"))
        # --- Summary: top 10 clusters by duplicate count + member details ---
        ^ sql_exec(
            SQL(
                """--beginsql
            CREATE OR REPLACE VIEW victim_cluster_counts AS
            SELECT cluster_id, COUNT(*) AS member_count
            FROM victim_clusters
            GROUP BY cluster_id;
        """
            )
        )
        ^ sql_query(
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
                                ",".join(f"'{r["cluster_id"]}'" for r in top_clusters)
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
        # --- Build gold-set representatives (one row per entity) ---
        ^ sql_exec(
            SQL(
                """--sql
            CREATE OR REPLACE VIEW victim_entity_reps AS
            SELECT
              ve.victim_entity_id,
              ve.city_id,
              ve.min_event_day,
              ve.max_event_day,
              floor((ve.min_event_day + ve.max_event_day)/2) AS entity_midpoint_day,
              ve.canonical_fullname,
              ve.canonical_sex,
              ve.canonical_race,
              ve.canonical_ethnicity,
              avg(m.lat)  FILTER (WHERE m.lat IS NOT NULL) AS lat_centroid,
              avg(m.lon)  FILTER (WHERE m.lon IS NOT NULL) AS lon_centroid,
              avg(m.victim_age) FILTER (WHERE m.victim_age IS NOT NULL) AS avg_age,
              min(m.victim_age) FILTER (WHERE m.victim_age IS NOT NULL) AS min_age,
              max(m.victim_age) FILTER (WHERE m.victim_age IS NOT NULL) AS max_age,
              mode(m.weapon)       AS mode_weapon,
              mode(m.circumstance) AS mode_circumstance,
              count(*) AS cluster_size
            FROM victim_entities ve
            JOIN victim_entity_members m USING (victim_entity_id)
            GROUP BY
              ve.victim_entity_id, ve.city_id, ve.min_event_day, ve.max_event_day,
              ve.canonical_fullname, ve.canonical_sex, ve.canonical_race,
              ve.canonical_ethnicity;
        """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victim_entity_reps"))
        >> (lambda rows: put_line(f"[D] victim_entity_reps rows: {rows[0]['n']}"))
        # --- Orphans: non-named victims not already in clusters ---
        ^ sql_exec(
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
        # --- Unified input for link_only pass (entities vs orphans) ---
        ^ sql_exec(
            SQL(
                """--sql
            CREATE OR REPLACE TEMPORARY TABLE linkage_input AS
            SELECT
              'entity' AS source_dataset,
              cast(victim_entity_id AS varchar) AS unique_id,
              city_id,
              entity_midpoint_day AS midpoint_day,
              lat_centroid AS lat,
              lon_centroid AS lon,
              avg_age AS victim_age,
              canonical_sex AS victim_sex,
              canonical_race AS victim_race,
              canonical_ethnicity AS victim_ethnicity,
              mode_weapon AS weapon,
              mode_circumstance AS circumstance,
              NULL::varchar AS victim_row_id
            FROM victim_entity_reps
            UNION ALL
            SELECT
              'orphan' AS source_dataset,
              cast(victim_row_id AS varchar) AS unique_id,
              city_id,
              midpoint_day,
              lat,
              lon,
              victim_age,
              victim_sex,
              victim_race,
              victim_ethnicity,
              weapon,
              circumstance,
              cast(victim_row_id AS varchar) AS victim_row_id
            FROM victims_orphan;
        """
            )
        )
        ^ sql_query(
            SQL(
                """--sql
            SELECT
              sum(CASE WHEN source_dataset='entity' THEN 1 ELSE 0 END) AS n_entities,
              sum(CASE WHEN source_dataset='orphan' THEN 1 ELSE 0 END) AS n_orphans
            FROM linkage_input;
        """
            )
        )
        >> (
            lambda rows: put_line(
                f"[D] linkage_input entities={rows[0]['n_entities']}, "
                f"orphans={rows[0]['n_orphans']}"
            )
        )
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
