"""
Match article victim entities to SHR victims using Splink linkage.
"""

from pymonad import (
    Run,
    SQL,
    ask,
    put_line,
    pure,
    splink_dedupe_job,
    sql_exec,
)
from menuprompts import NextStep
from comparison import (
    DATE_COMP,
    AGE_COMP,
    DIST_COMP,
    WEAPON_COMP,
    CIRC_COMP,
    OFFENDER_AGE_COMP,
    OFFENDER_SEX_COMP,
    OFFENDER_RACE_COMP,
    OFFENDER_ETHNICITY_COMP,
)


# Define linkage settings for SHR matching
shr_linkage_settings = {
    "link_type": "link_only",
    "comparisons": [
        DATE_COMP,
        AGE_COMP,
        OFFENDER_AGE_COMP,
        OFFENDER_SEX_COMP,
        OFFENDER_RACE_COMP,
        OFFENDER_ETHNICITY_COMP,
        DIST_COMP,
        WEAPON_COMP,
        CIRC_COMP,
    ],
    "blocking_rules_to_generate_predictions": [
        "l.incident_date = r.incident_date",
        "l.city_id = r.city_id AND abs(l.midpoint_day - r.midpoint_day) <= 30",
        "l.victim_age = r.victim_age AND l.victim_sex = r.victim_sex",
        "l.offender_age = r.offender_age AND l.offender_sex = r.offender_sex",
    ],
    "unique_id_column_name": "unique_id",  # for article victims
    "unique_id_column_name_r": "shr_id",  # for SHR
}


def match_article_to_shr_victims() -> Run[NextStep]:
    """
    Export SHR and article victim data to DuckDB, then run Splink linkage.
    """
    return (
        put_line("Exporting SHR data to DuckDB...") ^
        sql_exec(
            SQL(
                """
                CREATE OR REPLACE TABLE shr_cached AS
                SELECT
                    ID as shr_id,
                    Year,
                    Month,
                    Incident,
                    CASE VicSex WHEN 'M' THEN 'male' WHEN 'F' THEN 'female' ELSE NULL END as victim_sex,
                    VicAge as victim_age,
                    CASE VicRace WHEN 'W' THEN 'White' WHEN 'B' THEN 'Black' WHEN 'I' THEN 'Native American' WHEN 'A' THEN 'Asian' ELSE NULL END as victim_race,
                    CASE VicEthnic WHEN 'H' THEN 'Hispanic' WHEN 'N' THEN 'Non-Hispanic' ELSE NULL END as victim_ethnicity,
                    CASE OffSex WHEN 'M' THEN 'male' WHEN 'F' THEN 'female' ELSE NULL END as offender_sex,
                    OffAge as offender_age,
                    CASE OffRace WHEN 'W' THEN 'White' WHEN 'B' THEN 'Black' WHEN 'I' THEN 'Native American' WHEN 'A' THEN 'Asian' ELSE NULL END as offender_race,
                    CASE OffEthnic WHEN 'H' THEN 'Hispanic' WHEN 'N' THEN 'Non-Hispanic' ELSE NULL END as offender_ethnicity
                FROM sqldb.shr
                """
            )
        ) ^
        put_line("Exporting article victim entities to DuckDB...") ^
        sql_exec(SQL("CREATE OR REPLACE TABLE article_victims AS SELECT * FROM entity_link_input")) ^
        put_line("Running Splink linkage...") ^
        ask() >>
        (lambda env: splink_dedupe_job(
            duckdb_path=env["duckdb_path"],
            input_table=["article_victims", "shr_cached"],
            settings=shr_linkage_settings,
            predict_threshold=0.5,
            cluster_threshold=0.65,
            pairs_out="shr_link_pairs",
            clusters_out="shr_link_clusters",
        ) >>
        (lambda pairs_clusters: put_line(
            f"Linkage complete. Pairs table: {pairs_clusters[0]}, Clusters table: {pairs_clusters[1]}"
        ) ^ pure(NextStep.CONTINUE)))
    )