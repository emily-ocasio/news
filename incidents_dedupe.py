"""
Dedupe incident records using Splink.
"""

import pprint

from enum import Enum
from itertools import groupby
import splink.comparison_library as cl
import splink.comparison_level_library as cll

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


class BlockComp(str, Enum):
    """
    Components that can be used in building blocking rules for deduplication.
    """

    SAME_CITY = "l.city_id = r.city_id"
    MIDPOINT_EXISTS = "l.midpoint_day IS NOT NULL AND r.midpoint_day IS NOT NULL"
    MIDPOINT_7MONTH = (
        "floor(l.midpoint_day/213) = floor(r.midpoint_day/213) "
        "AND floor((l.midpoint_day+106)/213) = floor((r.midpoint_day+106)/213)"
    )
    EXACT_YEAR_MONTH = "l.year = r.year AND l.month = r.month"
    EXACT_YEAR_MONTH_DAY = "l.incident_date = r.incident_date"
    SAME_NAMES = (
        "l.victim_surname_norm = r.victim_surname_norm "
        "AND l.victim_forename_norm = r.victim_forename_norm"
    )
    SAME_SURNAME_SOUNDEX = "l.victim_surname_soundex = r.victim_surname_soundex"
    SAME_FORENAME_SOUNDEX = "l.victim_forename_soundex = r.victim_forename_soundex"
    SAME_AGE_SEX = "l.victim_age = r.victim_age AND l.victim_sex = r.victim_sex"
    MIDPOINT_30DAYS = "abs(l.midpoint_day - r.midpoint_day) <= 30"
    MIDPOINT_90DAYS = "abs(l.midpoint_day - r.midpoint_day) <= 90"
    DIFFERENT_ARTICLE = "l.article_id <> r.article_id"
    LONG_LAT_EXISTS = (
        "l.lat IS NOT NULL AND r.lat IS NOT NULL "
        "AND l.lon IS NOT NULL AND r.lon IS NOT NULL"
    )
    CLOSE_LONG_LAT = "abs(l.lat - r.lat) <= 0.0045 AND abs(l.lon - r.lon) <= 0.0055"


class ComparisonComp(str, Enum):
    """
    Components that can be used in building comparison clauses for deduplication.
    """

    EXACT_YEAR_MONTH_DAY = "incident_date_l = incident_date_r"
    MIDPOINT_EXISTS = "midpoint_day_l IS NOT NULL AND midpoint_day_r IS NOT NULL"
    MIDPOINT_2DAYS = "abs(midpoint_day_l - midpoint_day_r) <= 2"
    MIDPOINT_7DAYS = "abs(midpoint_day_l - midpoint_day_r) <= 7"
    MIDPOINT_10DAYS = "abs(midpoint_day_l - midpoint_day_r) <= 10"
    MIDPOINT_20DAYS = "abs(midpoint_day_l - midpoint_day_r) <= 20"
    MIDPOINT_30DAYS = "abs(midpoint_day_l - midpoint_day_r) <= 30"
    MIDPOINT_90DAYS = "abs(midpoint_day_l - midpoint_day_r) <= 90"
    MIDPOINT_7MONTH = "abs(midpoint_day_l - midpoint_day_r) <= 210"
    YEAR_PRECISION = "date_precision_l = 'year' OR date_precision_r = 'year'"
    MONTH_PRECISION = (
        "(date_precision_l = 'month' OR date_precision_r = 'month') "
        "AND date_precision_l <> 'year' AND date_precision_r <> 'year'"
    )
    DAY_PRECISION = "date_precision_l = 'day' AND date_precision_r = 'day'"
    EXACT_AGE = "victim_age_l = victim_age_r"
    AGE_NULL = "victim_age_l IS NULL OR victim_age_r IS NULL"
    AGE_2YEAR = "abs(victim_age_l - victim_age_r) <= 2"
    AGE_5YEARS = "abs(victim_age_l - victim_age_r) <= 5"


def _clause_from_comps(*components: Enum) -> str:
    return " AND ".join([component.value for component in components])


def _block_from_comps(
    *components: BlockComp, add_article_exclusion: bool = True
) -> str:
    comp_list = [BlockComp.SAME_CITY, *components]
    if add_article_exclusion:
        comp_list.append(BlockComp.DIFFERENT_ARTICLE)
    return _clause_from_comps(*comp_list)


def dedupe_incidents_with_splink(env) -> Run[NextStep]:
    """
    Deduplicate incident records using Splink.
    """
    return (
        sql_exec(
            SQL(
                """
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
            training_blocking_rules=[
                _block_from_comps(
                    BlockComp.EXACT_YEAR_MONTH, add_article_exclusion=False
                ),
                _block_from_comps(
                    BlockComp.SAME_NAMES,
                    add_article_exclusion=False,
                ),
                _block_from_comps(
                    BlockComp.LONG_LAT_EXISTS,
                    BlockComp.CLOSE_LONG_LAT,
                    add_article_exclusion=False,
                ),
                _block_from_comps(BlockComp.SAME_AGE_SEX, add_article_exclusion=False),
            ],
            deterministic_rules=[
                _block_from_comps(
                    BlockComp.SAME_NAMES,
                    BlockComp.MIDPOINT_30DAYS,
                ),
                _block_from_comps(
                    BlockComp.EXACT_YEAR_MONTH_DAY,
                    BlockComp.SAME_AGE_SEX,
                    BlockComp.LONG_LAT_EXISTS,
                    BlockComp.CLOSE_LONG_LAT,
                ),
            ],
            deterministic_recall=0.95,
        )
        >> (
            lambda outnames: put_line(
                f"[D] Wrote {outnames[0]} and {outnames[1]} in DuckDB."
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            CREATE OR REPLACE TABLE victim_entity_members AS
            SELECT
            c.cluster_id AS victim_entity_id,
            v.*
            FROM victim_clusters AS c
            JOIN victims_cached_enh AS v
            ON v.victim_row_id = c.victim_row_id
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
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
        """
            )
        )
        ^ sql_query(SQL("SELECT COUNT(*) AS n FROM victim_entities"))
        >> (lambda rows: put_line(f"[D] victim_entities rows: {rows[0]['n']}"))
        # --- Summary: top 10 clusters by duplicate count + member details ---
        ^ sql_exec(
            SQL(
                r"""
            CREATE OR REPLACE TABLE victim_cluster_counts AS
            SELECT cluster_id, COUNT(*) AS member_count
            FROM victim_clusters
            GROUP BY cluster_id;
        """
            )
        )
        ^ sql_query(
            SQL(
                r"""
            SELECT cluster_id, member_count
            FROM victim_cluster_counts
            ORDER BY member_count DESC, cluster_id
            LIMIT 50;
        """
            )
        )
        >> (
            lambda top_clusters: (
                # If no top clusters, just print a message; otherwise fetch members
                put_line("[D] Top 50 victim clusters by member count:")
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
                            "JOIN victims_cached_enh v ON vc.victim_row_id = v.victim_row_id "
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
    # Blocking rules

    b1 = _block_from_comps(BlockComp.SAME_NAMES)

    b2 = _block_from_comps(BlockComp.EXACT_YEAR_MONTH)

    b3 = _block_from_comps(
        BlockComp.MIDPOINT_EXISTS, BlockComp.MIDPOINT_7MONTH, BlockComp.CLOSE_LONG_LAT
    )

    b4 = _block_from_comps(BlockComp.SAME_AGE_SEX)

    b5 = _block_from_comps(BlockComp.SAME_FORENAME_SOUNDEX)

    b6 = _block_from_comps(BlockComp.SAME_SURNAME_SOUNDEX)

    # Comparisons
    name_comp = cl.CustomComparison(
        output_column_name="victim_name",
        comparison_levels=[
            # 0) NULL level (same as stock)
            {
                "is_null_level": True,
                "label_for_charts": "victim names NULL",
                "sql_condition": '("victim_forename_norm_l" IS NULL '
                'OR "victim_forename_norm_r" IS NULL) '
                'AND ("victim_surname_norm_l" IS NULL '
                'OR "victim_surname_norm_r" IS NULL)',
            },
            # 1) Exact match on concatenated fullname (with TF)
            {
                "label_for_charts": "Exact match victim name",
                "sql_condition": '"victim_fullname_concat_l" = "victim_fullname_concat_r"',
                # "tf_adjustment_column": "victim_fullname_concat",
                # "tf_adjustment_weight": 1.0,
            },
            # 2) JW >= 0.96 on both parts (same-direction)
            {
                "label_for_charts": "JW >= 0.96 victim names"
                "AND (Jaro-Winkler distance of victim_surname_norm >= 0.96)",
                "sql_condition": "(jaro_winkler_similarity("
                '"victim_forename_norm_l",'
                '"victim_forename_norm_r") >= 0.96) '
                "AND (jaro_winkler_similarity("
                '"victim_surname_norm_l",'
                '"victim_surname_norm_r") >= 0.96)',
            },
            # 3) MERGED level: (reversed exact)
            # OR (JW >= 0.92 both parts, same-direction)
            {
                "label_for_charts": "Reversed exact OR JW>=0.92 on both parts",
                "sql_condition": '(("victim_forename_norm_l" = "victim_surname_norm_r" '
                'AND "victim_forename_norm_r" = "victim_surname_norm_l") '
                "OR (jaro_winkler_similarity("
                '"victim_forename_norm_l",'
                '"victim_forename_norm_r") >= 0.92 '
                "AND jaro_winkler_similarity("
                '"victim_surname_norm_l",'
                '"victim_surname_norm_r") >= 0.92))',
            },
            # 4) JW >= 0.88 on both parts (same-direction)
            {
                "label_for_charts": "JW victim name >= 0.80"
                "AND (Jaro-Winkler distance of victim_surname_norm >= 0.80)",
                "sql_condition": "(jaro_winkler_similarity("
                '"victim_forename_norm_l","victim_forename_norm_r") >= 0.80) '
                "AND (jaro_winkler_similarity("
                '"victim_surname_norm_l","victim_surname_norm_r") >= 0.80)',
            },
            # # 5) Exact surname (with TF)
            # {
            #     "label_for_charts": "Exact match victim_surname_norm",
            #     "sql_condition": '"victim_surname_norm_l" = "victim_surname_norm_r"',
            #     "tf_adjustment_column": "victim_surname_norm",
            #     "tf_adjustment_weight": 1.0,
            # },
            # # 6) Exact forename (with TF)
            # {
            #     "label_for_charts": "Exact match on victim_forename_norm",
            #     "sql_condition": '"victim_forename_norm_l" = "victim_forename_norm_r"',
            #     "tf_adjustment_column": "victim_forename_norm",
            #     "tf_adjustment_weight": 1.0,
            # },
            # 7) Else
            {"label_for_charts": "All other comparisons", "sql_condition": "ELSE"},
        ],
    )

    dist_comp = cl.DistanceInKMAtThresholds(
        lat_col="lat",
        long_col="lon",
        km_thresholds=[0.1, 0.5, 1.5],
    )

    offender_comp = cl.CustomComparison(
        output_column_name="offender",
        comparison_levels=[
            {
                "is_null_level": True,
                "label_for_charts": "offender NULL",
                "sql_condition": '("offender_fullname_concat_l" IS NULL '
                'OR offender_fullname_concat_r IS NULL) '
            },
            {
                "label_for_charts": "Exact match offender name",
                "sql_condition": '"offender_fullname_concat_l" = "offender_fullname_concat_r"',
            },
            {
                "label_for_charts": "JW >= 0.90 offender name",
                "sql_condition": 'jaro_winkler_similarity('
                '"offender_fullname_concat_l","offender_fullname_concat_r") >= 0.90',
            },
            cll.ElseLevel(),
        ]
    )

    # New comparison: Same article penalty
    # Same-article heavy penalty using CustomComparison (doc'd)
    # We force m << u on the "same_article" level to yield a strong negative weight.
    same_article_comp = cl.CustomComparison(
        output_column_name="same_article_penalty",
        comparison_levels=[
            {
                "sql_condition": "article_id_l = article_id_r",
                "label_for_charts": "same_article",
                "m_probability": 1e-12,
                "u_probability": 1e-3,
            },
            {
                "sql_condition": "article_id_l IS NULL OR article_id_r IS NULL",
                "label_for_charts": "article_id_null",
                "is_null_level": True,
            },
            cll.ElseLevel(),
        ],
    ).configure(term_frequency_adjustments=False)

    date_comp = cl.CustomComparison(
        output_column_name="date_proximity",
        comparison_levels=[
            {
                "label_for_charts": "Exact match",
                "sql_condition": ComparisonComp.EXACT_YEAR_MONTH_DAY.value,
            },
            {
                "label_for_charts": "midpoint_day within 2 days",
                "sql_condition": _clause_from_comps(
                    ComparisonComp.MIDPOINT_EXISTS,
                    ComparisonComp.MIDPOINT_2DAYS,
                    ComparisonComp.DAY_PRECISION,
                ),
            },
            {
                "label_for_charts": "midpoint_day within 10 days",
                "sql_condition": _clause_from_comps(
                    ComparisonComp.MIDPOINT_EXISTS,
                    ComparisonComp.MIDPOINT_10DAYS,
                    ComparisonComp.DAY_PRECISION
                ),
            },
            {
                "label_for_charts": "midpoint_day within 30 days",
                "sql_condition": _clause_from_comps(
                    ComparisonComp.MIDPOINT_EXISTS,
                    ComparisonComp.MIDPOINT_90DAYS,
                    ComparisonComp.MONTH_PRECISION,
                ),
            },
            {
                "label_for_charts": "midpoint_day within 7 months",
                "sql_condition": _clause_from_comps(
                    ComparisonComp.MIDPOINT_EXISTS,
                    ComparisonComp.MIDPOINT_7MONTH,
                    ComparisonComp.YEAR_PRECISION,
                ),
            },
            {
                "label_for_charts": "Null dates",
                "sql_condition": "midpoint_day_l IS NULL OR midpoint_day_r IS NULL",
                "is_null_level": True,
            },
            cll.ElseLevel(),
        ],
    )

    age_comp = cl.CustomComparison(
        output_column_name="age_proximity",
        comparison_levels=[
            {
                "label_for_charts": "Null age",
                "sql_condition": ComparisonComp.AGE_NULL.value,
                "is_null_level": True,
            },
            {
                "label_for_charts": "Exact match",
                "sql_condition": ComparisonComp.EXACT_AGE.value,
            },
            {
                "label_for_charts": "Age within 2 years",
                "sql_condition": ComparisonComp.AGE_2YEAR.value,
            },
            cll.ElseLevel(),
        ],
    )

    weapon_comp = cl.CustomComparison(
        output_column_name="weapon",
        comparison_levels=[
            {
                "label_for_charts": "Null weapon",
                "sql_condition": "weapon_l IS NULL OR weapon_r IS NULL "
                "OR weapon_l = 'unknown' OR weapon_r = 'unknown'",
                "is_null_level": True,
            },
            {
                "label_for_charts": "Exact match",
                "sql_condition": "weapon_l = weapon_r "
                "OR (weapon_l IN ('firearm', 'handgun', 'rifle', 'shotgun') "
                "AND weapon_r IN ('firearm', 'handgun', 'rifle', 'shotgun'))",
            },
            cll.ElseLevel(),
        ],
    )

    circ_comp = cl.CustomComparison(
        output_column_name="circumstance",
        comparison_levels=[
            {
                "label_for_charts": "Null circumstance",
                "sql_condition": "circumstance_l IS NULL OR circumstance_r IS NULL "
                "OR circumstance_l = 'undetermined' OR circumstance_r = 'undetermined'",
                "is_null_level": True,
            },
            {
                "label_for_charts": "Exact match",
                "sql_condition": "circumstance_l = circumstance_r",
            },
            cll.ElseLevel(),
        ],
    )

    comparisons = [
        name_comp,
        date_comp,
        age_comp,
        dist_comp,
        cl.ExactMatch("victim_sex"),
        offender_comp,
        weapon_comp,
        circ_comp,
        # same_article_comp,  # Add the new comparison here
    ]

    return {
        "link_type": "dedupe_only",
        "unique_id_column_name": "victim_row_id",
        "blocking_rules_to_generate_predictions": [b1, b2, b3, b4, b5, b6],
        "comparisons": comparisons,
    }


def dedupe_incidents() -> Run[NextStep]:
    """
    Entry point for controller to deduplicate incidents
    """
    return with_duckdb(ask() >> dedupe_incidents_with_splink)
