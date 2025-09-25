"""
Dedupe incident records using Splink.
"""

import splink.comparison_library as cl  # v4 uses CamelCase constructors

from menuprompts import NextStep
from pymonad import Run, ask, put_line, pure, \
    sql_query, sql_exec, SQL, splink_dedupe_job, with_duckdb

def dedupe_incidents_with_splink(env) -> Run[NextStep]:
    """
    Deduplicate incident records using Splink.
    """
    return \
        sql_query(SQL("SELECT COUNT(*) AS n FROM victims_cached_enh")) >> \
        (lambda rows: \
        put_line(f"[D] victims_cached_enh rows: {rows[0]['n']}")) ^ \
        splink_dedupe_job(
            duckdb_path=env["duckdb_path"],
            input_table="victims_cached_enh",
            settings=_settings_for_victim_dedupe(),
            predict_threshold=0.60,
            cluster_threshold=0.60,
            pairs_out="victim_pairs",
            clusters_out="victim_clusters",
            train_first=True
        ) >> (lambda outnames:
            put_line(f"[D] Wrote {outnames[0]} and {outnames[1]} in DuckDB.")
        ) ^ \
        sql_exec(SQL(r"""
            CREATE OR REPLACE TABLE victim_entity_members AS
            SELECT
            c.cluster_id AS victim_entity_id,
            v.*
            FROM victim_clusters AS c
            JOIN victims_cached_enh AS v
            ON v.victim_row_id = c.victim_row_id
        """)) ^ \
        sql_exec(SQL(r"""
            CREATE OR REPLACE TABLE victim_entities AS
            SELECT
            victim_entity_id,
            any_value(victim_name_norm) FILTER (WHERE victim_name_norm IS NOT NULL) AS canonical_name,
            any_value(victim_surname_norm) FILTER (WHERE victim_surname_norm IS NOT NULL) AS canonical_surname,
            any_value(victim_sex)   FILTER (WHERE victim_sex   IS NOT NULL) AS canonical_sex,
            any_value(victim_race)  FILTER (WHERE victim_race  IS NOT NULL) AS canonical_race,
            any_value(victim_ethnicity) FILTER (WHERE victim_ethnicity IS NOT NULL) AS canonical_ethnicity,
            min(event_start_day) AS min_event_day,
            max(event_end_day)   AS max_event_day,
            any_value(city_id)   AS city_id
            FROM victim_entity_members
            GROUP BY victim_entity_id
        """)) ^ \
        sql_query(SQL("SELECT COUNT(*) AS n FROM victim_entities")) >> \
            (lambda rows:
            put_line(f"[D] victim_entities rows: {rows[0]['n']}")
        ) ^ pure(NextStep.CONTINUE)

def _settings_for_victim_dedupe() -> dict:
    # Blocking rules (OR across rules)
    b0 = "l.article_id = r.article_id"  # within-article dupes (cheap)
    b1 = "l.city_id = r.city_id AND l.incident_date = r.incident_date"
    b2 = ("l.city_id = r.city_id AND l.event_start_day IS NOT NULL "
          "AND r.event_start_day IS NOT NULL "
          "AND l.event_start_day <= r.event_end_day "
          "AND r.event_start_day <= l.event_end_day")
    b3 = ("l.city_id = r.city_id AND l.event_start_day IS NOT NULL "
          "AND r.event_start_day IS NOT NULL "
          "AND floor(l.event_start_day/30) = floor(r.event_start_day/30) "
          "AND l.street_token IS NOT NULL AND r.street_token IS NOT NULL "
          "AND l.street_token = r.street_token")
    b4 = ("l.city_id = r.city_id "
          "AND l.victim_surname_norm IS NOT NULL "
          "AND r.victim_surname_norm IS NOT NULL "
          "AND l.victim_surname_norm = r.victim_surname_norm "
          "AND floor(l.event_start_day/90) = floor(r.event_start_day/90)")

    name_comp = cl.ForenameSurnameComparison(
        "victim_forename_norm",
        "victim_surname_norm",
        jaro_winkler_thresholds=[0.96, 0.92, 0.88],  # tune if needed
        forename_surname_concat_col_name="victim_fullname_concat"
    ).configure(term_frequency_adjustments=True)

    comparisons = [
        # Name (with TF on full name)
        name_comp,

        # Demographics
        cl.ExactMatch("victim_sex"),
        cl.ExactMatch("victim_race"),
        cl.ExactMatch("victim_ethnicity"),

        # Age (coarse)
        cl.ExactMatch("victim_age_bucket5"),

        # Context
        cl.ExactMatch("weapon"),
        cl.ExactMatch("circumstance"),

        # Optional soft signal
        cl.JaroWinklerAtThresholds("offender_name_norm", [0.98, 0.94]),
    ]

    return {
        "link_type": "dedupe_only",
        "unique_id_column_name": "victim_row_id",               # CHANGED
        "blocking_rules_to_generate_predictions": [b0, b1, b2, b3, b4],
        "comparisons": comparisons,
        # Optional: add TF adjustments later for very common surnames/names
    }

def dedupe_incidents() -> Run[NextStep]:
    """
    Entry point for controller to deduplicate incidents
    """
    return with_duckdb(
        ask() >> dedupe_incidents_with_splink
    )
