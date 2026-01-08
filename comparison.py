"""
Classes and helper functions for defining comparison levels in Splink
"""

from enum import StrEnum
from dataclasses import dataclass, asdict

import splink.internals.comparison_library as cl
import splink.internals.comparison_level_library as cll

from blocking import _clause_from_comps

def _specific_value_comp_builder(field: str, value: str) -> str:
    return f'("{field}_l" {value} AND "{field}_r" {value})'

def _null_comp_builder(field: str, result: str = "IS NULL") -> str:
    return f'("{field}_l" {result} OR "{field}_r" {result})'

def _exact_comp_builder(field: str) -> str:
    return f'"{field}_l" = "{field}_r"'

def _distance_comp_builder(field: str, distance: int) -> str:
    return f'abs("{field}_l" - "{field}_r") <= {distance}'

def _similarity_comp_builder2(field1: str, field2: str, threshold: float) -> str:
    return f'(jaro_winkler_similarity("{field1}_l", "{field1}_r") >= {threshold}) ' \
           f'AND (jaro_winkler_similarity("{field2}_l", "{field2}_r") >= {threshold})'

class ComparisonComp(StrEnum):
    """
    Components that can be used in building comparison clauses for deduplication.
    """
    EXACT_YEAR_MONTH_DAY = _exact_comp_builder("incident_date")
    MIDPOINT_EXISTS = _specific_value_comp_builder("midpoint_day", "IS NOT NULL")
    MIDPOINT_2DAYS = _distance_comp_builder("midpoint_day", 2)
    MIDPOINT_7DAYS = _distance_comp_builder("midpoint_day", 7)
    MIDPOINT_10DAYS = _distance_comp_builder("midpoint_day", 10)
    MIDPOINT_20DAYS = _distance_comp_builder("midpoint_day", 20)
    MIDPOINT_30DAYS = _distance_comp_builder("midpoint_day", 30)
    MIDPOINT_90DAYS = _distance_comp_builder("midpoint_day", 90)
    MIDPOINT_7MONTH = _distance_comp_builder("midpoint_day", 210)
    YEAR_PRECISION = """
        "date_precision_l" = 'year' OR "date_precision_r" = 'year'
    """
    MONTH_PRECISION = """
        ("date_precision_l" = 'month' OR "date_precision_r" = 'month')
        AND ("date_precision_l" <> 'year' AND "date_precision_r" <> 'year')
    """
        # "AND date_precision_l <> 'year' AND date_precision_r <> 'year'"
    DAY_PRECISION = _specific_value_comp_builder("date_precision", "= 'day'")
    DATE_NULL = _null_comp_builder("midpoint_day")
    EXACT_AGE = _exact_comp_builder("victim_age")
    AGE_NULL = _null_comp_builder("victim_age")
    AGE_2YEAR = _distance_comp_builder("victim_age", 2)
    AGE_5YEARS = _distance_comp_builder("victim_age", 5)
    VICTIM_FORENAME_NULL = _null_comp_builder("victim_forename_norm")
    VICTIM_SURNAME_NULL = _null_comp_builder("victim_surname_norm")
    EXACT_VICTIM_FULLNAME = _exact_comp_builder("victim_fullname_concat")
    EXACT_VICTIM_SURNAME = _exact_comp_builder("victim_surname_norm")
    VICTIM_EXACT_REVERSED = "victim_forename_norm_l = victim_surname_norm_r " \
                             "AND victim_surname_norm_l = victim_forename_norm_r"
    OFFENDER_NULL = _null_comp_builder("offender_fullname_concat")
    OFFENDER_CLOSE = _similarity_comp_builder2(
        "offender_forename_norm", "offender_surname_norm", 0.85)
    WEAPON_NULL = _null_comp_builder("weapon") + ' OR ' + \
        _null_comp_builder("weapon", "= 'unknown'")
    WEAPON_EXACT = _exact_comp_builder("weapon") + " OR " + \
        _specific_value_comp_builder(
            "weapon", "IN ('firearm', 'handgun', 'rifle', 'shotgun')")
    CIRC_NULL = _null_comp_builder("circumstance") + ' OR ' + \
        _null_comp_builder("circumstance", "= 'undetermined'")
    CIRC_EXACT = _exact_comp_builder("circumstance")

@dataclass(frozen=True)
class ComparisonLevel:
    """
    Dataclass for defining a comparison level in Splink
    """
    label_for_charts: str
    sql_condition: str

    def to_dict(self):
        """Convert the ComparisonLevel instance to a dictionary."""
        return asdict(self)

@dataclass(frozen=True)
class NullComparisonLevel(ComparisonLevel):
    """
    Comparison level for null values
    """
    is_null_level: bool = True

NAME_COMP = cl.CustomComparison(
    output_column_name="victim_name",
    comparison_levels=[
        NullComparisonLevel(
            "victim names NULL",
            _clause_from_comps(
                ComparisonComp.VICTIM_SURNAME_NULL,
                ComparisonComp.VICTIM_FORENAME_NULL
            )
        ).to_dict(),
        ComparisonLevel(
            "exact match victim name",
            ComparisonComp.EXACT_VICTIM_FULLNAME.value
        ).to_dict(),
        ComparisonLevel(
            "JW >= 0.96 victim names",
            _similarity_comp_builder2(
                "victim_forename_norm", "victim_surname_norm", 0.96)
        ).to_dict(),
        ComparisonLevel(
            "Reversed exact or JW >= 0.92 victim names or exact single surname",
            ComparisonComp.VICTIM_EXACT_REVERSED.value + " OR " +
            _similarity_comp_builder2(
                "victim_forename_norm", "victim_surname_norm", 0.92) + " OR " +
            "( " + ComparisonComp.EXACT_VICTIM_SURNAME.value + " AND " +
                ComparisonComp.VICTIM_FORENAME_NULL.value + " )"
        ).to_dict(),
        ComparisonLevel(
            "JW > 0.80 victim names",
            _similarity_comp_builder2(
                "victim_forename_norm", "victim_surname_norm", 0.80)
        ).to_dict(),
        ComparisonLevel(
            "All other comparisons",
            "ELSE"
        ).to_dict()
    ]
)

DATE_COMP = cl.CustomComparison(
    output_column_name="incident_date",
    comparison_levels=[
        NullComparisonLevel(
            "incident dates NULL",
            ComparisonComp.DATE_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match incident date",
            ComparisonComp.EXACT_YEAR_MONTH_DAY.value
        ).to_dict(),
        ComparisonLevel(
            "midpoint within 2 days",
            _clause_from_comps(
                ComparisonComp.MIDPOINT_EXISTS,
                ComparisonComp.MIDPOINT_2DAYS,
                ComparisonComp.DAY_PRECISION
            )
        ).to_dict(),
        ComparisonLevel(
            "midpoint within 10 days",
            _clause_from_comps(
                ComparisonComp.MIDPOINT_EXISTS,
                ComparisonComp.MIDPOINT_10DAYS,
                ComparisonComp.DAY_PRECISION
            )
        ).to_dict(),
        ComparisonLevel(
            "midpoint within 90 days",
            _clause_from_comps(
                ComparisonComp.MIDPOINT_EXISTS,
                ComparisonComp.MIDPOINT_90DAYS,
                ComparisonComp.MONTH_PRECISION
            )
        ).to_dict(),
        ComparisonLevel(
            "midpoint within 7 months",
            _clause_from_comps(
                ComparisonComp.MIDPOINT_EXISTS,
                ComparisonComp.MIDPOINT_7MONTH,
                ComparisonComp.YEAR_PRECISION
            )
        ).to_dict(),
        cll.ElseLevel()
    ]
)

AGE_COMP = cl.CustomComparison(
    output_column_name="victim_age",
    comparison_levels=[
        NullComparisonLevel(
            "victim ages NULL",
            ComparisonComp.AGE_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match victim age",
            ComparisonComp.EXACT_AGE.value
        ).to_dict(),
        ComparisonLevel(
            "victim ages within 2 years",
            ComparisonComp.AGE_2YEAR.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)

DIST_COMP = cl.DistanceInKMAtThresholds(
    lat_col="lat",
    long_col="lon",
    km_thresholds=[0.1, 0.5, 1.5],
)

OFFENDER_COMP = cl.CustomComparison(
    output_column_name="offender",
    comparison_levels=[
        NullComparisonLevel(
            "offender NULL",
            ComparisonComp.OFFENDER_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "offender names close",
            ComparisonComp.OFFENDER_CLOSE.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)

WEAPON_COMP = cl.CustomComparison(
    output_column_name="weapon",
    comparison_levels=[
        NullComparisonLevel(
            "weapon NULL",
            ComparisonComp.WEAPON_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match weapon",
            ComparisonComp.WEAPON_EXACT.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)

CIRC_COMP = cl.CustomComparison(
    output_column_name="circumstance",
    comparison_levels=[
        NullComparisonLevel(
            "circumstance NULL",
            ComparisonComp.CIRC_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match circumstance",
            ComparisonComp.CIRC_EXACT.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)

SUMMARY_COMP = cl.CosineSimilarityAtThresholds(
    col_name="summary_vec",
    score_threshold_or_thresholds=[0.80, 0.65, 0.50, 0.35]
)
