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
    sql_export,
    with_duckdb,
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
    "unique_id_column_name": "unique_id",  # for both tables
}


def match_article_to_shr_victims() -> Run[NextStep]:
    """
    Export SHR and article victim data to DuckDB, then run Splink linkage.
    """
    return with_duckdb(
        put_line("Exporting SHR data to DuckDB...") ^
        sql_exec(
            SQL(
                """
                CREATE OR REPLACE TABLE shr_cached AS
                SELECT
                    ID as unique_id,
                    Year,
                    YearMonth,
                    Incident,
                    CASE VicSex WHEN 'M' THEN 'male' WHEN 'F' THEN 'female' ELSE NULL END as victim_sex,
                    VicAge as victim_age,
                    CASE VicRace WHEN 'W' THEN 'White' WHEN 'B' THEN 'Black' WHEN 'I' THEN 'Native American' WHEN 'A' THEN 'Asian' ELSE NULL END as victim_race,
                    CASE VicEthnic WHEN 'H' THEN 'Hispanic' WHEN 'N' THEN 'Non-Hispanic' ELSE NULL END as victim_ethnicity,
                    CASE OffSex WHEN 'M' THEN 'male' WHEN 'F' THEN 'female' ELSE NULL END as offender_sex,
                    OffAge as offender_age,
                    CASE OffRace WHEN 'W' THEN 'White' WHEN 'B' THEN 'Black' WHEN 'I' THEN 'Native American' WHEN 'A' THEN 'Asian' ELSE NULL END as offender_race,
                    CASE OffEthnic WHEN 'H' THEN 'Hispanic' WHEN 'N' THEN 'Non-Hispanic' ELSE NULL END as offender_ethnicity,
                    -- Map SHR Weapon to GPT schema enums
                    CASE 
                        WHEN LOWER(TRIM(Weapon)) LIKE '%knife%' THEN 'knife'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%shotgun%' THEN 'shotgun'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%rifle%' THEN 'rifle'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%handgun%' THEN 'handgun'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%firearm%' THEN 'firearm'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%gun%' THEN 'firearm'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%blunt%' THEN 'blunt object'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%personal%' THEN 'personal weapon'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%fire%' THEN 'fire'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%strangulation%' THEN 'strangulation'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%asphyxiation%' THEN 'asphyxiation'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%drugs%' THEN 'drugs'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%explosives%' THEN 'explosives'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%drowning%' THEN 'drowning'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%poison%' THEN 'poison'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%pushed%' THEN 'pushed from height'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%other%' THEN 'other'
                        WHEN LOWER(TRIM(Weapon)) LIKE '%unknown%' THEN 'unknown'
                        ELSE 'unknown'  -- Default to unknown for unmapped values
                    END AS weapon,
                    -- Computed fields for Splink
                    'month' AS date_precision,
                    NULL AS incident_date,
                    CASE WHEN YearMonth IS NOT NULL THEN
                        floor((
                            date_diff('day', DATE '1970-01-01', make_date(CAST(substring(YearMonth, 1, 4) AS BIGINT), CAST(substring(YearMonth, 6, 2) AS BIGINT), 1)) +
                            date_diff('day', DATE '1970-01-01', (make_date(CAST(substring(YearMonth, 1, 4) AS BIGINT), CAST(substring(YearMonth, 6, 2) AS BIGINT), 1) + INTERVAL 1 MONTH - INTERVAL 1 DAY))
                        ) / 2)
                    ELSE NULL END AS midpoint_day,
                    NULL AS lat,  -- SHR may not have precise coords; use NULL or default DC
                    NULL AS lon,
                    'SHR' AS city_id  -- Placeholder; could map from Incident if available
                FROM sqldb.shr
                """
            )
        ) ^
        put_line("Exporting article victim entities to DuckDB...") ^
        sql_exec(
            SQL(
                """
                CREATE OR REPLACE TABLE article_victims AS
                SELECT
                    victim_entity_id AS unique_id,
                    city_id,
                    entity_midpoint_day AS midpoint_day,
                    incident_date,
                    entity_date_precision AS date_precision,
                    lat_centroid AS lat,
                    lon_centroid AS lon,
                    canonical_age AS victim_age,
                    canonical_sex AS victim_sex,
                    canonical_race AS victim_race,
                    canonical_ethnicity AS victim_ethnicity,
                    canonical_offender_age AS offender_age,
                    canonical_offender_sex AS offender_sex,
                    canonical_offender_race AS offender_race,
                    canonical_offender_ethnicity AS offender_ethnicity,
                    mode_weapon AS weapon,
                    mode_circumstance AS circumstance,
                    article_ids_csv,
                    '' AS exclusion_id
                FROM victim_entity_reps_new
                """
            )
        ) ^
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
        ) ^
        sql_exec(
            SQL(
                """
                CREATE OR REPLACE TABLE shr_entity_matches AS
                SELECT
                    unique_id_l AS entity_uid,
                    unique_id_r AS shr_id,
                    match_probability
                FROM shr_link_pairs
                WHERE match_probability >= 0.5
                """
            )
        ) ^
        sql_export(
            SQL(
                """
                SELECT
                    sem.entity_uid,
                    sem.shr_id,
                    sem.match_probability,
                    ver.canonical_fullname,
                    ver.canonical_age,
                    ver.canonical_sex,
                    ver.canonical_race,
                    ver.article_ids_csv,
                    sc.Year,
                    sc.YearMonth,
                    sc.victim_age AS shr_victim_age,
                    sc.victim_sex AS shr_victim_sex,
                    sc.victim_race AS shr_victim_race
                FROM shr_entity_matches sem
                JOIN victim_entity_reps_new ver ON sem.entity_uid = ver.victim_entity_id
                JOIN shr_cached sc ON sem.shr_id = sc.unique_id
                ORDER BY sem.match_probability DESC
                """
            ),
            "shr_matches.xlsx",
            "SHR_Matches",
        ) ^
        put_line("Exported SHR matches to shr_matches.xlsx.") ^
        pure(NextStep.CONTINUE)))
    )