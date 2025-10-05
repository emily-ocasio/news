"""
Classes and helper functions for defining comparison levels in Splink
"""

from enum import StrEnum
from dataclasses import dataclass, asdict

import splink.comparison_library as cl
import splink.comparison_level_library as cll

from blocking import _clause_from_comps

def _specific_value_comp_builder(field: str, value: str) -> str:
    return f'"{field}_l" {value} AND "{field}_r" {value}'

def _null_comp_builder(field: str, result: str = "IS NULL") -> str:
    return f'"{field}_l" {result} OR "{field}_r" {result}'

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
        "date_precision_l" <> 'day' OR "date_precision_r" <> 'day'
    """
        # "AND date_precision_l <> 'year' AND date_precision_r <> 'year'"
    DAY_PRECISION = _specific_value_comp_builder("date_precision", "= 'day'")
    EXACT_AGE = _exact_comp_builder("victim_age")
    AGE_NULL = _null_comp_builder("victim_age")
    AGE_2YEAR = _distance_comp_builder("victim_age", 2)
    AGE_5YEARS = _distance_comp_builder("victim_age", 5)
    VICTIM_FORENAME_NULL = _null_comp_builder("victim_forename_norm")
    VICTIM_SURNAME_NULL = _null_comp_builder("victim_surname_norm")
    EXACT_VICTIM_FULLNAME = _exact_comp_builder("victim_fullname_concat")
    VICTIM_EXACT_REVERSED = "victim_forename_norm_l = victim_surname_norm_r " \
                             "AND victim_surname_norm_l = victim_forename_norm_r"

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
            "Reversed exact or JW >= 0.92 victim names",
            ComparisonComp.VICTIM_EXACT_REVERSED.value + " OR " +
            _similarity_comp_builder2(
                "victim_forename_norm", "victim_surname_norm", 0.92)
        ).to_dict(),
        ComparisonLevel(
            "JW >= 0.80 victim names",
            _similarity_comp_builder2(
                "victim_forename_norm", "victim_surname_norm", 0.80)
        ).to_dict(),
        ComparisonLevel(
            "All other comparisons",
            "ELSE"
        ).to_dict()
    ]
)
