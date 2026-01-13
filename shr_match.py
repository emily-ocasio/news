"""
Match article victim entities to SHR victims using Splink linkage.
"""
import splink.internals.comparison_library as cl

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
    Unit, unit,
)
from menuprompts import NextStep
from comparison import (
    DATE_COMP_SHR,
    AGE_COMP,
    #DIST_COMP,
    WEAPON_COMP,
    CIRC_COMP,
    OFFENDER_AGE_COMP,
    OFFENDER_SEX_COMP,
    OFFENDER_RACE_COMP,
    OFFENDER_ETHNICITY_COMP,
)
from blocking import SHR_OVERALL_BLOCKS, SHR_DETERMINISTIC_BLOCKS


# Define linkage settings for SHR matching
shr_linkage_settings = {
    "link_type": "link_only",
    "comparisons": [
        DATE_COMP_SHR,
        AGE_COMP,
        OFFENDER_AGE_COMP,
        OFFENDER_SEX_COMP,
        OFFENDER_RACE_COMP,
        OFFENDER_ETHNICITY_COMP,
        #DIST_COMP,  - no location in SHR for DC
        WEAPON_COMP,
        CIRC_COMP,
        cl.ExactMatch("victim_sex"),
        cl.ExactMatch("victim_race"),
        cl.ExactMatch("victim_ethnicity"),
    ],
    "blocking_rules_to_generate_predictions": SHR_OVERALL_BLOCKS,
    "unique_id_column_name": "unique_id",  # for both tables
}


def _export_shr_debug_matches_excel() -> Run[Unit]:
    """
    Build a single worksheet that lists every article victim entity and every SHR record.

    Matching logic:
      - Assumes unique_id_l is always from entities, unique_id_r from SHR.
      - For SHR linkage, assume one-to-one; no per-article uniqueness needed.

    Output:
      - Exactly one row per entity, one per SHR record.
      - match_id = 'match_<entity_uid>' for matched groups, otherwise 'entity_<...>' or 'shr_<...>'.
      - band_key = 0 (unmatched entity), 1 (unmatched SHR), 2 (matched group).
      - Ordering uses the entity midpoint for matched groups; otherwise row’s own midpoint.
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
                av.weapon, av.circumstance,
                av.offender_age, av.offender_sex, av.offender_race, av.offender_ethnicity,
                pr.match_probability,
                pr.entity_uid,
                pr.shr_uid,
                av.victim_age AS entity_victim_age,
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
                sc.victim_sex AS shr_victim_sex,
                sc.victim_race AS shr_victim_race,
                sc.victim_ethnicity AS shr_victim_ethnicity,
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
              JOIN shr_cached sc ON pr.shr_uid = sc.unique_id

              UNION ALL

              -- Unmatched entities
              SELECT
                'entity' AS rec_type,
                av.midpoint_day AS entity_midpoint_day,
                av.date_precision AS entity_date_precision,
                av.year AS entity_year, av.month AS entity_month,
                av.victim_age, av.victim_sex, av.victim_race, av.victim_ethnicity,
                av.weapon, av.circumstance,
                av.offender_age, av.offender_sex, av.offender_race, av.offender_ethnicity,
                CAST(NULL AS DOUBLE) AS match_probability,
                av.unique_id AS entity_uid,
                CAST(NULL AS VARCHAR) AS shr_uid,
                av.victim_age AS entity_victim_age,
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
                CAST(NULL AS VARCHAR) AS shr_victim_sex,
                CAST(NULL AS VARCHAR) AS shr_victim_race,
                CAST(NULL AS VARCHAR) AS shr_victim_ethnicity,
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
              WHERE av.unique_id NOT IN (SELECT entity_uid FROM pairs_raw)

              UNION ALL

              -- Unmatched SHR
              SELECT
                'shr' AS rec_type,
                CAST(NULL AS DOUBLE) AS entity_midpoint_day,
                CAST(NULL AS VARCHAR) AS entity_date_precision,
                CAST(NULL AS INTEGER) AS entity_year, CAST(NULL AS INTEGER) AS entity_month,
                sc.victim_age, sc.victim_sex, sc.victim_race, sc.victim_ethnicity,
                sc.weapon, sc.circumstance,
                sc.offender_age, sc.offender_sex, sc.offender_race, sc.offender_ethnicity,
                CAST(NULL AS DOUBLE) AS match_probability,
                CAST(NULL AS VARCHAR) AS entity_uid,
                sc.unique_id AS shr_uid,
                CAST(NULL AS INTEGER) AS entity_victim_age,
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
                sc.victim_sex AS shr_victim_sex,
                sc.victim_race AS shr_victim_race,
                sc.victim_ethnicity AS shr_victim_ethnicity,
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
              CASE WHEN rec_type = 'match' THEN 2
                  WHEN rec_type = 'entity' THEN 0
                  ELSE 1
              END AS band_key,
              entity_uid,
              shr_uid,
              match_probability,
              entity_year,
              shr_year,
              entity_month,
              shr_month,
              entity_date_precision,
              shr_date_precision,
              entity_midpoint_day,
              shr_midpoint_day,
              entity_victim_age,
              shr_victim_age,
              entity_victim_sex,
              shr_victim_sex,
              entity_victim_race,
              shr_victim_race,
              entity_victim_ethnicity,
              shr_victim_ethnicity,
              entity_weapon,
              shr_weapon,
              entity_circumstance,
              shr_circumstance,
              entity_offender_age,
              shr_offender_age,
              entity_offender_sex,
              shr_offender_sex,
              entity_offender_race,
              shr_offender_race,
              entity_offender_ethnicity,
              shr_offender_ethnicity
            FROM combined
            ORDER BY
              CASE rec_type WHEN 'match' THEN 0 WHEN 'entity' THEN 1 ELSE 2 END,
              entity_uid, match_probability DESC, shr_uid
            """
            ),
            "shr_debug_matches.xlsx",
            "Matches",
            band_by_group_col="band_key",
            band_wrap=3,
        )
        ^ put_line("[SHR] Wrote shr_debug_matches.xlsx (final entities + SHR records after linkage).")
        ^ pure(unit)
    )


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
                    "index" as unique_id,
                    CASE VicSex WHEN 'Unknown' THEN NULL ELSE lower(VicSex) END as victim_sex,
                    VicAge as victim_age,
                    CASE VicRace WHEN 'Unknown' THEN NULL ELSE VicRace END as victim_race,
                    CASE VicEthnic WHEN 'Unknown' THEN NULL ELSE VicEthnic END as victim_ethnicity,
                    CASE OffSex WHEN 'Unknown' THEN NULL ELSE lower(OffSex) END as offender_sex,
                    CASE OffAge WHEN 999 THEN NULL ELSE OffAge END as offender_age,
                    CASE OffRace WHEN 'Unknown' THEN NULL ELSE OffRace END as offender_race,
                    CASE OffEthnic WHEN 'Unknown' THEN NULL ELSE OffEthnic END as offender_ethnicity,
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
                    -- Map SHR Circumstance to GPT schema enums
                    CASE 
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%narcotic%' THEN 'narcotics related'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%lover%' THEN 'lover''s triangle'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%undetermined%' THEN 'undetermined'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%argument%' THEN 'argument'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%robbery%' THEN 'robbery'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%brawl%' THEN 'brawl'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%motor vehicle%' THEN 'other felony related'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%felony%' THEN 'other felony related'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%arson%' THEN 'arson'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%negligent%' THEN 'negligence'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%manslaughter%' THEN 'negligence'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%babysitter%' THEN 'child killed by babysitter'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%gang%' THEN 'gang killing'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%burglary%' THEN 'burglary'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%rape%' THEN 'rape'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%playing with gun%' THEN 'negligence'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%hunting accident%' THEN 'negligence'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%institutional%' THEN 'institutional killing'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%prostitution%' THEN 'other felony related'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%larceny%' THEN 'other felony related'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%gambling%' THEN 'other felony related'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%sex offense%' THEN 'rape'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%police%' THEN 'felon killed by police'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%private citizen%' THEN 'other'
                        WHEN LOWER(TRIM(Circumstance)) LIKE '%other%' THEN 'other'
                        ELSE 'undetermined'  -- Default to undetermined for unmapped values
                    END AS circumstance,
                    'month' AS date_precision,
                    CASE WHEN YearMonth IS NOT NULL THEN
                        floor((
                            date_diff('day', DATE '1970-01-01', make_date(CAST(substring(YearMonth, 1, 4) AS BIGINT), CAST(substring(YearMonth, 6, 2) AS BIGINT), 1)) +
                            date_diff('day', DATE '1970-01-01', (make_date(CAST(substring(YearMonth, 1, 4) AS BIGINT), CAST(substring(YearMonth, 6, 2) AS BIGINT), 1) + INTERVAL 1 MONTH - INTERVAL 1 DAY))
                        ) / 2)
                    ELSE NULL END AS midpoint_day,
                    CAST(substring(YearMonth, 1, 4) AS INTEGER) AS year,
                    CAST(substring(YearMonth, 6, 2) AS INTEGER) AS month,
                    -- NULL AS lat,  -- SHR may not have precise coords; use NULL or default DC
                    -- NULL AS lon,
                    2 AS city_id  -- Corresponds to DC (PublicationID of Washi Post)
                FROM sqldb.shr
                WHERE State = 'District of Columbia' -- Only DC for now
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
                    entity_date_precision AS date_precision,
                    -- lat_centroid AS lat,
                    -- lon_centroid AS lon,
                    extract(year from (DATE '1970-01-01' + to_days(entity_midpoint_day))) AS year,
                    extract(month from (DATE '1970-01-01' + to_days(entity_midpoint_day))) AS month,
                    canonical_age AS victim_age,
                    canonical_sex AS victim_sex,
                    canonical_race AS victim_race,
                    canonical_ethnicity AS victim_ethnicity,
                    canonical_offender_age AS offender_age,
                    canonical_offender_sex AS offender_sex,
                    canonical_offender_race AS offender_race,
                    canonical_offender_ethnicity AS offender_ethnicity,
                    mode_weapon AS weapon,
                    mode_circumstance AS circumstance
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
            predict_threshold=0.01,
            cluster_threshold=0.65,
            deterministic_rules=SHR_DETERMINISTIC_BLOCKS,
            deterministic_recall=0.01,
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
                    sc.VicAge AS shr_victim_age,
                    sc.VicSex AS shr_victim_sex,
                    sc.VicRace AS shr_victim_race
                FROM shr_entity_matches sem
                JOIN victim_entity_reps_new ver ON sem.entity_uid = ver.victim_entity_id
                JOIN sqldb.shr sc ON sem.shr_id = sc.ID
                ORDER BY sem.match_probability DESC
                """
            ),
            "shr_matches.xlsx",
            "SHR_Matches",
        ) ^
        _export_shr_debug_matches_excel() ^
        put_line("Exported SHR matches to shr_matches.xlsx.") ^
        pure(NextStep.CONTINUE)))
    )