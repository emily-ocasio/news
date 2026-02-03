"""
Classes and helper functions for defining comparison levels in Splink
"""

from enum import StrEnum
from dataclasses import dataclass, asdict

import splink.internals.comparison_library as cl
import splink.internals.comparison_level_library as cll
from splink import ColumnExpression
import splink.internals.comparison_level_composition as cllc

from blocking import _clause_from_comps

def _specific_value_comp_builder(field: str, value: str) -> str:
    return f'("{field}_l" {value} AND "{field}_r" {value})'

def _null_comp_builder(field: str, result: str = "IS NULL") -> str:
    return f'("{field}_l" {result} OR "{field}_r" {result})'

def _exact_comp_builder(field: str) -> str:
    return f'"{field}_l" = "{field}_r"'

def _distance_comp_builder(field: str, distance: int) -> str:
    return f'abs("{field}_l" - "{field}_r") <= {distance}'

def _jw_similarity_comp_builder(field1: str, field2: str, threshold: float) -> str:
    return f'(jaro_winkler_similarity("{field1}_l", "{field1}_r") >= {threshold}) ' \
           f'AND (jaro_winkler_similarity("{field2}_l", "{field2}_r") >= {threshold})'

class ComparisonComp(StrEnum):
    """
    Components that can be used in building comparison clauses for deduplication.
    """
    EXACT_YEAR_MONTH_DAY = _exact_comp_builder("incident_date")
    EXACT_YEAR_MONTH = \
        f"{_exact_comp_builder('year')} AND {_exact_comp_builder('month')}"
    MIDPOINT_EXISTS = _specific_value_comp_builder("midpoint_day", "IS NOT NULL")
    MIDPOINT_2DAYS = _distance_comp_builder("midpoint_day", 2)
    MIDPOINT_7DAYS = _distance_comp_builder("midpoint_day", 7)
    MIDPOINT_10DAYS = _distance_comp_builder("midpoint_day", 10)
    MIDPOINT_15DAYS_RIGHT = (
        _distance_comp_builder("midpoint_day", 15)
        + " AND midpoint_day_l <= midpoint_day_r"
    )
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
    AGE_1YEAR = _distance_comp_builder("victim_age", 1)
    AGE_2YEAR = _distance_comp_builder("victim_age", 2)
    AGE_5YEARS = _distance_comp_builder("victim_age", 5)
    AGE_10YEARS = _distance_comp_builder("victim_age", 10)
    VICTIM_FORENAME_NULL = _null_comp_builder("victim_forename_norm")
    VICTIM_SURNAME_NULL = _null_comp_builder("victim_surname_norm")
    EXACT_VICTIM_FULLNAME = _exact_comp_builder("victim_fullname_concat")
    EXACT_VICTIM_SURNAME = _exact_comp_builder("victim_surname_norm")
    EXACT_VICTIM_FORENAME = _exact_comp_builder("victim_forename_norm")
    VICTIM_EXACT_REVERSED = "victim_forename_norm_l = victim_surname_norm_r " \
                             "AND victim_surname_norm_l = victim_forename_norm_r"
    VICTIM_FORENAME_ONLY = ' OR '.join((
        ' AND '.join((
            "(victim_forename_norm_l IS NULL",
            "victim_surname_norm_l = victim_forename_norm_r)",
        )),
        ' AND '.join((
            "(victim_forename_norm_r IS NULL",
            "victim_forename_norm_l = victim_surname_norm_r)",
        )),
    ))
    VICTIM_SURNAME_ONLY = ' AND '.join((
        _null_comp_builder("victim_forename_norm"),
        _exact_comp_builder("victim_surname_norm"),
    ))
    OFFENDER_NULL = _null_comp_builder("offender_fullname_concat")
    OFFENDER_CLOSE = _jw_similarity_comp_builder(
        "offender_forename_norm", "offender_surname_norm", 0.85)
    WEAPON_NULL = _null_comp_builder("weapon") + ' OR ' + \
        _null_comp_builder("weapon", "= 'unknown'") + ' OR ' + \
        _null_comp_builder("weapon", "= 'other'")
    WEAPON_EXACT = _exact_comp_builder("weapon") + " OR " + \
        _specific_value_comp_builder(
            "weapon", "IN ('firearm', 'handgun', 'rifle', 'shotgun')")
    WEAPON_FIREARM = _specific_value_comp_builder(
        "weapon", "IN ('firearm', 'handgun')")
    CIRC_NULL = _null_comp_builder("circumstance") + ' OR ' + \
        _null_comp_builder("circumstance", "= 'undetermined'")
    CIRC_EXACT = _exact_comp_builder("circumstance")
    EXACT_OFFENDER_AGE = _exact_comp_builder("offender_age")
    OFFENDER_AGE_NULL = _null_comp_builder("offender_age")
    OFFENDER_AGE_2YEAR = _distance_comp_builder("offender_age", 2)
    OFFENDER_AGE_5YEARS = _distance_comp_builder("offender_age", 5)
    EXACT_OFFENDER_SEX = _exact_comp_builder("offender_sex")
    OFFENDER_SEX_NULL = _null_comp_builder("offender_sex")
    EXACT_OFFENDER_RACE = _exact_comp_builder("offender_race")
    OFFENDER_RACE_NULL = _null_comp_builder("offender_race")
    EXACT_OFFENDER_ETHNICITY = _exact_comp_builder("offender_ethnicity")
    OFFENDER_ETHNICITY_NULL = _null_comp_builder("offender_ethnicity")

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

@dataclass(frozen=True)
class TFComparisonLevel(ComparisonLevel):
    """
    Comparison level for term frequency-adjusted comparisons
    """
    tf_adjustment_column: str
    tf_adjustment_weight: float = 1.0
    tf_minimum_u_value: float = 0.001


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
            "Reversed exact or JW >= 0.80 victim names or exact single surname",
            ' OR '.join((
                ComparisonComp.VICTIM_EXACT_REVERSED.value,
                _jw_similarity_comp_builder(
                    "victim_forename_norm",
                    "victim_surname_norm",
                    0.80
                ),
                ComparisonComp.VICTIM_SURNAME_ONLY.value,
                ComparisonComp.VICTIM_FORENAME_ONLY.value,
            ))
        ).to_dict(),
        ComparisonLevel(
            "Forename or surname exact match",
            ' OR '.join((
                ComparisonComp.EXACT_VICTIM_FORENAME.value,
                ComparisonComp.EXACT_VICTIM_SURNAME.value,
            ))
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
        cllc.Or(
            cllc.And(
                cll.ExactMatchLevel("year"),
                cll.ExactMatchLevel("month")
            ),
            cllc.And(
                cll.AbsoluteDifferenceLevel("midpoint_day", 2),
                cll.LiteralMatchLevel("date_precision", "day", "string")
            ),
        ).configure(label_for_charts="exact yr/mon or within 2 days"),
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


DATE_COMP_ORPHAN = cl.CustomComparison(
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
        # ComparisonLevel(
        #     "midpoint within 7 months",
        #     _clause_from_comps(
        #         ComparisonComp.MIDPOINT_EXISTS,
        #         ComparisonComp.MIDPOINT_7MONTH,
        #         ComparisonComp.YEAR_PRECISION
        #     )
        # ).to_dict(),
        cll.ElseLevel()
    ]
)

DATE_COMP_SHR = cl.CustomComparison(
    output_column_name="incident_date",
    comparison_levels=[
        NullComparisonLevel(
            "incident dates NULL",
            ComparisonComp.DATE_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact year and month",
            ComparisonComp.EXACT_YEAR_MONTH.value
        ).to_dict(),
        ComparisonLevel(
            "midpoint within 10 days",
            _clause_from_comps(
                ComparisonComp.MIDPOINT_EXISTS,
                ComparisonComp.MIDPOINT_15DAYS_RIGHT,
                ComparisonComp.MONTH_PRECISION
            )
        ).to_dict(),
        # ComparisonLevel(
        #     "midpoint within 30 days",
        #     _clause_from_comps(
        #         ComparisonComp.MIDPOINT_EXISTS,
        #         ComparisonComp.MIDPOINT_30DAYS,
        #         ComparisonComp.MONTH_PRECISION
        #     )
        # ).to_dict(),
        ComparisonLevel(
            "midpoint within 90 days",
            _clause_from_comps(
                ComparisonComp.MIDPOINT_EXISTS,
                ComparisonComp.MIDPOINT_90DAYS,
                ComparisonComp.MONTH_PRECISION
            )
        ).to_dict(),
        # ComparisonLevel(
        #     "midpoint within 7 months",
        #     _clause_from_comps(
        #         ComparisonComp.MIDPOINT_EXISTS,
        #         ComparisonComp.MIDPOINT_7MONTH,
        #         ComparisonComp.YEAR_PRECISION
        #     )
        # ).to_dict(),
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

SUMMARY_NULL_COMP = (
    cllc.Or(
        cll.NullLevel("summary_vec"),
        cll.LiteralMatchLevel(
            ColumnExpression(
                "list_max(list_transform(summary_vec, x -> abs(x)))"
            ),
            "0",
            "float",
            side_of_comparison="left",
        ),
        cll.LiteralMatchLevel(
            ColumnExpression(
                "list_max(list_transform(summary_vec, x -> abs(x)))"
            ),
            "0",
            "float",
            side_of_comparison="right",
        ),
    )
)

VICTIM_COUNT_COMP = cl.CustomComparison(
    output_column_name="victim_count",
    comparison_levels=[
        cllc.Or(
            cll.NullLevel("victim_count"),
            SUMMARY_NULL_COMP
        ).configure(
            label_for_charts="victim counts NULL or zero",
            is_null_level=True,
        ),
        NullComparisonLevel(
            "victim counts NULL",
            _null_comp_builder("victim_count")
        ).to_dict(),
        TFComparisonLevel(
            "exact match victim count",
            _exact_comp_builder("victim_count"),
            "victim_count",
            0.8,
            0.0001
        ).to_dict(),
        # ComparisonLevel(
        #     "victim counts within 1 (counts > 1)",
        #     (
        #         '"victim_count_l" > 1 AND "victim_count_r" > 1 '
        #         'AND abs("victim_count_l" - "victim_count_r") <= 1'
        #     )
        # ).to_dict(),
        cll.ElseLevel()
    ]
)

AGE_COMP_SHR = cl.CustomComparison(
    output_column_name="victim_age",
    comparison_levels=[
        NullComparisonLevel(
            "victim ages NULL",
            ComparisonComp.AGE_NULL.value
        ).to_dict(),
        TFComparisonLevel(
            "exact match victim age",
            ComparisonComp.EXACT_AGE.value,
            "victim_age"
        ).to_dict(),
        TFComparisonLevel(
            "victim ages within 1 year",
            ComparisonComp.AGE_1YEAR.value,
            "victim_age"
        ).to_dict(),
        TFComparisonLevel(
            "victim ages within 2 years",
            ComparisonComp.AGE_2YEAR.value,
            "victim_age"
        ).to_dict(),
        TFComparisonLevel(
            "victim ages within 10 years",
            ComparisonComp.AGE_10YEARS.value,
            "victim_age"
        ).to_dict(),
        cll.ElseLevel()
    ]
)

AGE_COMP_ORPHAN = cl.CustomComparison(
    output_column_name="victim_age",
    comparison_levels=[
        NullComparisonLevel(
            "victim ages NULL",
            ComparisonComp.AGE_NULL.value
        ).to_dict(),
        # ComparisonLevel(
        #     "exact match victim age",
        #     ComparisonComp.EXACT_AGE.value
        # ).to_dict(),
        TFComparisonLevel(
            "exact match victim age",
            ComparisonComp.EXACT_AGE.value,
            "victim_age"
        ).to_dict(),
        TFComparisonLevel(
            "victim ages within 2 years",
            ComparisonComp.AGE_2YEAR.value,
            "victim_age"
        ).to_dict(),
        cll.ElseLevel()
    ]
)

DIST_COMP = cl.DistanceInKMAtThresholds(
    lat_col="lat",
    long_col="lon",
    km_thresholds=[0.1, 0.5, 1.5],
)

DIST_STREET_TYPE = cllc.Or(
    cll.LiteralMatchLevel("address_type", "ADDRESS", "string", "left"),
    cll.LiteralMatchLevel("address_type", "ADDRESS", "string", "right"),
    cll.LiteralMatchLevel("address_type", "INTERSECTION", "string", "left"),
    cll.LiteralMatchLevel("address_type", "INTERSECTION", "string", "right"),
    cll.LiteralMatchLevel("address_type", "BLOCK", "string", "left"),
    cll.LiteralMatchLevel("address_type", "BLOCK", "string", "right"),
    cll.LiteralMatchLevel("address_type", "STREET ONLY", "string", "left"),
    cll.LiteralMatchLevel("address_type", "STREET ONLY", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_ADDRESS", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_ADDRESS", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_INTERSECTION", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_INTERSECTION", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_BLOCK", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_BLOCK", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_STREET_ONLY", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_STREET_ONLY", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_ADDRESS", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_ADDRESS", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_INTERSECTION", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_INTERSECTION", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_BLOCK", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_BLOCK", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_STREET_ONLY", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT_STREET_ONLY", "string", "right"),
)

DIST_PLACE_TYPE = cllc.Or(
    cll.LiteralMatchLevel("address_type", "NAMED_PLACE", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NAMED_PLACE", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_SUCCESS", "string", "right"),
    cll.LiteralMatchLevel("address_type", "UNRECOGNIZED_PLACE", "string", "left"),
    cll.LiteralMatchLevel("address_type", "UNRECOGNIZED_PLACE", "string", "right"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT", "string", "left"),
    cll.LiteralMatchLevel("address_type", "NO_RESULT", "string", "right"),
    cll.LiteralMatchLevel("address_type", "APPROXIMATE_PLACE", "string", "left"),
    cll.LiteralMatchLevel("address_type", "APPROXIMATE_PLACE", "string", "right"),
)

DIST_COMP_NEW = cl.CustomComparison(
    output_column_name = "location",
    comparison_levels = [
        cll.NullLevel(ColumnExpression("geo_address_norm").nullif("UNKNOWN")),
        cllc.Or(
            cllc.And(
                cll.DistanceInKMLevel("lat", "lon", 0.0001),
                cllc.Or(
                    cll.LiteralMatchLevel("address_type", "ADDRESS", "string"),
                    cll.LiteralMatchLevel("address_type", "INTERSECTION", "string"),
                    cll.LiteralMatchLevel("address_type", "BLOCK", "string"),
                    cll.LiteralMatchLevel("geo_score", "100", "float"),
                    cll.ExactMatchLevel("geo_address_norm")
                )
            ),
            cllc.And(
                cll.LiteralMatchLevel("address_type", "NAMED_PLACE", "string"),
                cll.ExactMatchLevel("geo_address_norm")
            ),
        ).configure(label_for_charts="exact location match"),
        cllc.Or(
            cll.DistanceInKMLevel("lat", "lon", 0.4),
            cllc.And(
                DIST_STREET_TYPE,
                cll.JaroWinklerLevel(
                    "geo_address_short", 0.90)
            ),
            cllc.And(
                cllc.Or(
                    cll.LiteralMatchLevel("address_type", "INTERSECTION", "string", "right"),
                    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_INTERSECTION", "string", "right"),
                    cll.LiteralMatchLevel("address_type", "NO_RESULT_INTERSECTION", "string", "right"),
                ),
                cll.CustomLevel(
                    'jaro_winkler_similarity("geo_address_short_l", "geo_address_short_2_r") >= 0.90',
                    label_for_charts="short vs short2 jw >= 0.90 (right intersection)"
                ),
            ),
            cllc.And(
                cllc.Or(
                    cll.LiteralMatchLevel("address_type", "INTERSECTION", "string", "left"),
                    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_INTERSECTION", "string", "left"),
                    cll.LiteralMatchLevel("address_type", "NO_RESULT_INTERSECTION", "string", "left"),
                ),
                cll.CustomLevel(
                    'jaro_winkler_similarity("geo_address_short_2_l", "geo_address_short_r") >= 0.90',
                    label_for_charts="short2 vs short jw >= 0.90 (left intersection)"
                ),
            ),
            cllc.And(
                cllc.Or(
                    cll.LiteralMatchLevel("address_type", "INTERSECTION", "string", "left"),
                    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_INTERSECTION", "string", "left"),
                    cll.LiteralMatchLevel("address_type", "NO_RESULT_INTERSECTION", "string", "left"),
                ),
                cllc.Or(
                    cll.LiteralMatchLevel("address_type", "INTERSECTION", "string", "right"),
                    cll.LiteralMatchLevel("address_type", "NO_SUCCESS_INTERSECTION", "string", "right"),
                    cll.LiteralMatchLevel("address_type", "NO_RESULT_INTERSECTION", "string", "right"),
                ),
                cll.CustomLevel(
                    'jaro_winkler_similarity("geo_address_short_2_l", "geo_address_short_2_r") >= 0.90',
                    label_for_charts="short2 vs short2 jw >= 0.90 (both intersection)"
                ),
            ),
            cllc.And(
                DIST_PLACE_TYPE,
                cll.JaroLevel(
                    "geo_address_norm", 0.5)
            ),
        ).configure(label_for_charts="within 0.4 km or similar address"),
        cll.ElseLevel()
    ]
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

TF_WEAPON_COMP = cl.CustomComparison(
    output_column_name="weapon",
    comparison_levels=[
        NullComparisonLevel(
            "weapon NULL",
            ComparisonComp.WEAPON_NULL.value
        ).to_dict(),
        cll.ExactMatchLevel("weapon").configure(
            tf_adjustment_column="weapon",
            tf_minimum_u_value=0.001),
        TFComparisonLevel(
            "firearm class weapon match",
            ComparisonComp.WEAPON_FIREARM.value,
            tf_adjustment_column="weapon",
        ).to_dict(),
        cll.ElseLevel()
    ]
)

TF_WEAPON_COMP_SHR = cl.CustomComparison(
    output_column_name="weapon",
    comparison_levels=[
        NullComparisonLevel(
            "weapon NULL",
            ComparisonComp.WEAPON_NULL.value
        ).to_dict(),
        cll.ExactMatchLevel("weapon").configure(
            tf_adjustment_column="weapon",
            tf_minimum_u_value=0.001),
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

OFFENDER_AGE_COMP = cl.CustomComparison(
    output_column_name="offender_age",
    comparison_levels=[
        NullComparisonLevel(
            "offender ages NULL",
            ComparisonComp.OFFENDER_AGE_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match offender age",
            ComparisonComp.EXACT_OFFENDER_AGE.value
        ).to_dict(),
        ComparisonLevel(
            "offender ages within 2 years",
            ComparisonComp.OFFENDER_AGE_2YEAR.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)

OFFENDER_SEX_COMP = cl.CustomComparison(
    output_column_name="offender_sex",
    comparison_levels=[
        NullComparisonLevel(
            "offender sex NULL",
            ComparisonComp.OFFENDER_SEX_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match offender sex",
            ComparisonComp.EXACT_OFFENDER_SEX.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)

OFFENDER_RACE_COMP = cl.CustomComparison(
    output_column_name="offender_race",
    comparison_levels=[
        NullComparisonLevel(
            "offender race NULL",
            ComparisonComp.OFFENDER_RACE_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match offender race",
            ComparisonComp.EXACT_OFFENDER_RACE.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)

OFFENDER_ETHNICITY_COMP = cl.CustomComparison(
    output_column_name="offender_ethnicity",
    comparison_levels=[
        NullComparisonLevel(
            "offender ethnicity NULL",
            ComparisonComp.OFFENDER_ETHNICITY_NULL.value
        ).to_dict(),
        ComparisonLevel(
            "exact match offender ethnicity",
            ComparisonComp.EXACT_OFFENDER_ETHNICITY.value
        ).to_dict(),
        cll.ElseLevel()
    ]
)



SUMMARY_COMP = cl.CustomComparison(
    output_column_name="summary_vec",
    comparison_levels=[
        SUMMARY_NULL_COMP.configure(
            label_for_charts="summary empty or 'No details'",
            is_null_level=True,
        ),
        cll.CosineSimilarityLevel(
            col_name="summary_vec",
            similarity_threshold=0.80
        ),
        cll.CosineSimilarityLevel(
            col_name="summary_vec",
            similarity_threshold=0.65
        ),
        cll.CosineSimilarityLevel(
            col_name="summary_vec",
            similarity_threshold=0.50
        ),
        cll.ElseLevel()
    ]
)
