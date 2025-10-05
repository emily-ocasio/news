"""
Dedupe incident records using Splink.
"""

from itertools import groupby
import splink.comparison_library as cl
import splink.comparison_level_library as cll

from blocking import (
    NAMED_VICTIM_BLOCKS,
    NAMED_VICTIM_DETERMINISTIC_BLOCKS,
    NAMED_VICTIM_BLOCKS_FOR_TRAINING,
    _clause_from_comps,
)
from comparison import NAME_COMP, ComparisonComp
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
            SQL("""--sql
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
            CREATE OR REPLACE TABLE victim_entities AS
            SELECT
            victim_entity_id,
            mode(victim_name_norm)
                FILTER (WHERE victim_fullname_concat IS NOT NULL) AS canonical_name,
            mode(victim_surname_norm)
                FILTER (WHERE victim_surname_norm IS NOT NULL) AS canonical_surname,
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
            CREATE OR REPLACE TABLE victim_cluster_counts AS
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
                "label_for_charts": "JW >= 0.96 victim names",
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
            # 4) JW >= 0.80 on both parts (same-direction)
            {
                "label_for_charts": "JW victim name >= 0.80",
                "sql_condition": "(jaro_winkler_similarity("
                '"victim_forename_norm_l","victim_forename_norm_r") >= 0.80) '
                "AND (jaro_winkler_similarity("
                '"victim_surname_norm_l","victim_surname_norm_r") >= 0.80)',
            },
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
                "OR offender_fullname_concat_r IS NULL) ",
            },
            # {
            #     "label_for_charts": "Exact match offender name",
            #     "sql_condition":
            #     '"offender_fullname_concat_l" = "offender_fullname_concat_r"',
            # },
            {
                "label_for_charts": "JW >= 0.85 offender names",
                "sql_condition": "(jaro_winkler_similarity("
                '"offender_forename_norm_l",'
                '"offender_forename_norm_r") >= 0.85) '
                "AND (jaro_winkler_similarity("
                '"offender_surname_norm_l",'
                '"offender_surname_norm_r") >= 0.85)',
            },
            cll.ElseLevel(),
        ],
    )

    # New comparison: Same article penalty
    # Same-article heavy penalty using CustomComparison (doc'd)
    # We force m << u on the "same_article" level to yield a strong negative weight.
    # same_article_comp = cl.CustomComparison(
    #     output_column_name="same_article_penalty",
    #     comparison_levels=[
    #         {
    #             "sql_condition": "article_id_l = article_id_r",
    #             "label_for_charts": "same_article",
    #             "m_probability": 1e-12,
    #             "u_probability": 1e-3,
    #         },
    #         {
    #             "sql_condition": "article_id_l IS NULL OR article_id_r IS NULL",
    #             "label_for_charts": "article_id_null",
    #             "is_null_level": True,
    #         },
    #         cll.ElseLevel(),
    #     ],
    # ).configure(term_frequency_adjustments=False)

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
                    ComparisonComp.DAY_PRECISION,
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
                    ComparisonComp.MONTH_PRECISION,
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
        NAME_COMP,
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
        "blocking_rules_to_generate_predictions": NAMED_VICTIM_BLOCKS,
        "comparisons": comparisons,
    }


def dedupe_incidents() -> Run[NextStep]:
    """
    Entry point for controller to deduplicate incidents
    """
    return with_duckdb(ask() >> dedupe_incidents_with_splink)
