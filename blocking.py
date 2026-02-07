"""
Classes and helper functions for defining blocking rules in Splink
"""

from enum import StrEnum

from splink import block_on
from splink.internals.blocking_rule_library import (
    And,
    Not,
    # Or,
    # CustomRule,
    # ExactMatchRule,
)
from splink.internals.blocking_rule_creator import BlockingRuleCreator

from pymonad import (
    Array,
    HashMap,
    TrainingBlockToComparisonLevelMap,
    comparison_level_keys,
)
from comparison import (
    _clause_from_comps,
    DATE_COMP,
    DATE_COMP_ORPHAN,
    DIST_COMP,
)

SAME_CITY = block_on("city_id")
def _and_city(cr: BlockingRuleCreator) -> BlockingRuleCreator:
    return And(cr, SAME_CITY)


DIFFERENT_ARTICLE = Not(block_on("exclusion_id"))
YEAR_MONTH = block_on("year", "month")
FULLNAME =  block_on("victim_fullname_concat")
AGE_SEX =   block_on("victim_age", "victim_sex")
OFFENDER_AGE_SEX = block_on("offender_age", "offender_sex")


class BlockComp(StrEnum):
    """
    Components that can be used in building blocking rules for deduplication.
    """

    SAME_CITY = "l.city_id = r.city_id"
    MIDPOINT_EXISTS = "l.midpoint_day IS NOT NULL AND r.midpoint_day IS NOT NULL"
    MIDPOINT_BLOCK_EXISTS = (
        "l.midpoint_day_block IS NOT NULL AND r.midpoint_day_block IS NOT NULL"
    )
    MIDPOINT_7MONTH = (
        "floor(l.midpoint_day/213) = floor(r.midpoint_day/213) "
        "AND floor((l.midpoint_day+106)/213) = floor((r.midpoint_day+106)/213)"
    )
    MIDPOINT_BLOCK_7MONTH = (
        "floor(l.midpoint_day_block/213) = floor(r.midpoint_day_block/213) "
        "AND floor((l.midpoint_day_block+106)/213) = "
        "floor((r.midpoint_day_block+106)/213)"
    )
    MIDPOINT_BLOCK_2MONTH = (
        "floor(l.midpoint_day_block/61) = floor(r.midpoint_day_block/61) "
        "AND floor((l.midpoint_day_block+30)/61) = "
        "floor((r.midpoint_day_block+30)/61)"
    )
    GEO_SHORT = "l.geo_address_short = r.geo_address_short"
    YEAR_BLOCK_1 = "abs(l.year_block - r.year_block) <= 1"
    YEAR_DIFF_2 = "abs(l.year - r.year) <= 2"
    EXACT_YEAR_BLOCK = "l.year_block = r.year_block"
    EXACT_YEAR = "l.year = r.year"
    EXACT_YEAR_MONTH = "l.year = r.year AND l.month = r.month"
    EXACT_YEAR_MONTH_DAY = "l.incident_date = r.incident_date"
    SAME_NAMES = (
        "l.victim_surname_norm = r.victim_surname_norm "
        "AND l.victim_forename_norm = r.victim_forename_norm"
    )
    SAME_FULLNAME = (
        "l.victim_fullname_concat = r.victim_fullname_concat")
    SAME_SURNAME_SOUNDEX = "l.victim_surname_soundex = r.victim_surname_soundex"
    SAME_FORENAME_SOUNDEX = "l.victim_forename_soundex = r.victim_forename_soundex"
    SAME_AGE_SEX = "l.victim_age = r.victim_age AND l.victim_sex = r.victim_sex"
    SAME_OFFENDER_AGE_SEX = (
        "l.offender_age = r.offender_age AND l.offender_sex = r.offender_sex"
    )
    OFFENDER_SEX = "l.offender_sex = r.offender_sex"
    AGE_DIFF2_SEX = (
        "abs(l.victim_age - r.victim_age) <= 2 "
        "AND l.victim_sex = r.victim_sex"
    )
    SAME_SEX = 'l.victim_sex = r.victim_sex'
    SAME_AGE = "l.victim_age = r.victim_age"
    SAME_WEAPON = "l.weapon = r.weapon"
    SAME_CIRCUMSTANCE = "l.circumstance = r.circumstance"
    FIREARM_HANDGUN = "l.weapon = 'firearm' AND r.weapon = 'handgun'"
    MIDPOINT_30DAYS = "abs(l.midpoint_day - r.midpoint_day) <= 30"
    MIDPOINT_90DAYS = "abs(l.midpoint_day - r.midpoint_day) <= 90"
    MONTH_PRECISION = (
        "(l.date_precision = 'month' OR r.date_precision = 'month') "
        "AND (l.date_precision <> 'year' AND r.date_precision <> 'year')"
    )
    YEAR_PRECISION = (
        "(l.date_precision = 'year' OR r.date_precision = 'year')"
    )
    WITHIN_YEAR = "abs(l.midpoint_day - r.midpoint_day) <= 370"
    DIFFERENT_ARTICLE = "l.exclusion_id <> r.exclusion_id"
    LONG_LAT_EXISTS = (
        "l.lat IS NOT NULL AND r.lat IS NOT NULL "
        "AND l.lon IS NOT NULL AND r.lon IS NOT NULL"
    )
    CLOSE_LONG_LAT = "abs(l.lat - r.lat) <= 0.0045 AND abs(l.lon - r.lon) <= 0.0055"
    CLOSE_SUMMARY = "array_cosine_similarity(l.summary_vec, r.summary_vec) >= 0.5"


def _add_city(*creators: BlockingRuleCreator) -> list[BlockingRuleCreator]:
    """
    Adds city check to each creator and returns as a list
        List is needed for settings
    """
    return list((_and_city & Array(creators)).a)


def _block_from_comps(
    *components: BlockComp, add_article_exclusion: bool = True
) -> str:
    comp_list = [BlockComp.SAME_CITY, *components]
    if add_article_exclusion:
        comp_list.append(BlockComp.DIFFERENT_ARTICLE)
    return _clause_from_comps(*comp_list)

def _train_block_from_comps(
    *components: BlockComp, add_article_exclusion: bool = True
) -> str:
    return _block_from_comps(*components, add_article_exclusion=add_article_exclusion)

class TrainBlockRule(StrEnum):
    """
    Predefined blocking rules for training (no article exclusion)
    """
    YEAR = _train_block_from_comps(BlockComp.EXACT_YEAR)
    YEAR_MONTH = _train_block_from_comps(BlockComp.EXACT_YEAR_MONTH)
    YEAR_BLOCK_1 = _train_block_from_comps(BlockComp.YEAR_BLOCK_1)
    YEAR_DIFF_2 = _train_block_from_comps(BlockComp.YEAR_DIFF_2)
    EXACT_YEAR_BLOCK = _train_block_from_comps(BlockComp.EXACT_YEAR_BLOCK)
    SAME_NAMES = _train_block_from_comps(BlockComp.SAME_NAMES)
    SAME_FULLNAME = _train_block_from_comps(BlockComp.SAME_FULLNAME)
    LOCATION = _train_block_from_comps(BlockComp.LONG_LAT_EXISTS,
                                       BlockComp.CLOSE_LONG_LAT)
    AGE_SEX = _train_block_from_comps(BlockComp.SAME_AGE_SEX)
    AGE_DIFF2_SEX = _train_block_from_comps(BlockComp.AGE_DIFF2_SEX)
    MIDPOINT_7MONTH = _train_block_from_comps(BlockComp.MIDPOINT_EXISTS,
                                              BlockComp.MIDPOINT_7MONTH)
    MIDPOINT_BLOCK_7MONTH = _train_block_from_comps(
        BlockComp.MIDPOINT_BLOCK_EXISTS,
        BlockComp.MIDPOINT_BLOCK_7MONTH
    )
    MIDPOINT_BLOCK_2MONTH = _train_block_from_comps(
        BlockComp.MIDPOINT_BLOCK_EXISTS,
        BlockComp.MIDPOINT_BLOCK_2MONTH
    )
    MIDPOINT_90DAYS_MONTH_PRECISION = _train_block_from_comps(
        BlockComp.MIDPOINT_EXISTS,
        BlockComp.MIDPOINT_90DAYS,
        BlockComp.MONTH_PRECISION
    )
    SUMMARY = _train_block_from_comps(BlockComp.CLOSE_SUMMARY)
    MONTH_AGE_SEX = _train_block_from_comps(
        BlockComp.EXACT_YEAR_MONTH,
        BlockComp.SAME_AGE_SEX
    )
    MONTH_AGE_SEX_WEAPON = _train_block_from_comps(
        BlockComp.EXACT_YEAR_MONTH,
        BlockComp.SAME_AGE_SEX,
        BlockComp.SAME_WEAPON
    )
    MONTH_AGE_SEX_FIREARM_HANDGUN = _train_block_from_comps(
        BlockComp.EXACT_YEAR_MONTH,
        BlockComp.SAME_AGE_SEX,
        BlockComp.FIREARM_HANDGUN
    )
    YEAR_CIRCUMSTANCE = _train_block_from_comps(
        BlockComp.EXACT_YEAR,
        BlockComp.SAME_CIRCUMSTANCE
    )
    YEAR_OFFENDER_AGE_SEX = _train_block_from_comps(
        BlockComp.EXACT_YEAR,
        BlockComp.EXACT_YEAR_MONTH,
        BlockComp.SAME_OFFENDER_AGE_SEX
    )
    OFFENDER_AGE_SEX = _train_block_from_comps(
        BlockComp.SAME_OFFENDER_AGE_SEX
    )
    OFFENDER_SEX_7MONTH = _train_block_from_comps(
        BlockComp.OFFENDER_SEX,
        BlockComp.MIDPOINT_7MONTH,
    )
    YEAR_AGE_SEX = _train_block_from_comps(
        BlockComp.EXACT_YEAR,
        BlockComp.SAME_AGE_SEX
    )
    GEO_SHORT = _train_block_from_comps(
        BlockComp.GEO_SHORT
    )
    AGE_GEO_SHORT = _train_block_from_comps(
        BlockComp.GEO_SHORT,
        BlockComp.SAME_AGE
    )
    SEX_GEO_SHORT = _train_block_from_comps(
        BlockComp.GEO_SHORT,
        BlockComp.SAME_SEX,
    )
    AGE_SEX_7MONTH = _train_block_from_comps(
        BlockComp.SAME_AGE_SEX,
        BlockComp.MIDPOINT_EXISTS,
        BlockComp.MIDPOINT_7MONTH,
        BlockComp.YEAR_PRECISION
    )
    SEX_7MONTH = _train_block_from_comps(
        BlockComp.SAME_SEX,
        BlockComp.MIDPOINT_EXISTS,
        BlockComp.MIDPOINT_7MONTH,
    )
    SEX_WITHIN_YEAR = _train_block_from_comps(
        BlockComp.SAME_SEX,
        BlockComp.WITHIN_YEAR
    )
    AGE_SEX_WITHIN_YEAR = _train_block_from_comps(
        BlockComp.SAME_AGE_SEX,
        BlockComp.WITHIN_YEAR
    )
    OFFENDER_SEX_WITHIN_YEAR = _train_block_from_comps(
        BlockComp.OFFENDER_SEX,
        BlockComp.WITHIN_YEAR
    )
    OFFENDER_AGE_SEX_WITHIN_YEAR = _train_block_from_comps(
        BlockComp.SAME_OFFENDER_AGE_SEX,
        BlockComp.WITHIN_YEAR
    )

class DedupBlockRule(StrEnum):
    """
    Predefined blocking rules for deduplication (with article exclusion)
    """
    EVERYTHING = _block_from_comps()
    YEAR_MONTH = _block_from_comps(BlockComp.EXACT_YEAR_MONTH)
    YEAR_MONTH_DAY = _block_from_comps(BlockComp.EXACT_YEAR_MONTH_DAY)
    DATE_LOCATION = _block_from_comps(BlockComp.MIDPOINT_EXISTS,
                                        BlockComp.MIDPOINT_7MONTH,
                                        BlockComp.LONG_LAT_EXISTS,
                                        BlockComp.CLOSE_LONG_LAT)
    SAME_NAMES = _block_from_comps(BlockComp.SAME_NAMES)
    SAME_NAMES_YEAR_DIFF2 = _block_from_comps(
        BlockComp.SAME_NAMES,
        BlockComp.YEAR_DIFF_2
    )
    SAME_FULLNAME = _block_from_comps(BlockComp.SAME_FULLNAME)
    YEAR_SAME_NAMES = _block_from_comps(
        BlockComp.EXACT_YEAR,
        BlockComp.SAME_NAMES
    )
    SURNAME_SOUNDEX = _block_from_comps(BlockComp.SAME_SURNAME_SOUNDEX)
    YEAR_SURNAME_SOUNDEX = _block_from_comps(
        BlockComp.EXACT_YEAR,
        BlockComp.SAME_SURNAME_SOUNDEX)
    FORENAME_SOUNDEX = _block_from_comps(BlockComp.SAME_FORENAME_SOUNDEX)
    YEAR_FORENAME_SOUNDEX = _block_from_comps(
        BlockComp.EXACT_YEAR,
        BlockComp.SAME_FORENAME_SOUNDEX)
    AGE_SEX = _block_from_comps(BlockComp.SAME_AGE_SEX)
    YEAR_AGE_SEX = _block_from_comps(
        BlockComp.EXACT_YEAR,
        BlockComp.SAME_AGE_SEX
    )
    SAME_NAMES_30DAYS = _block_from_comps(
        BlockComp.SAME_NAMES, BlockComp.MIDPOINT_30DAYS)
    DATE_LOCATION_AGE_SEX = _block_from_comps(
        BlockComp.EXACT_YEAR_MONTH_DAY,
        BlockComp.LONG_LAT_EXISTS,
        BlockComp.CLOSE_LONG_LAT,
        BlockComp.SAME_AGE_SEX)
    DATE_LOCATION_SEX = _block_from_comps(
        BlockComp.EXACT_YEAR_MONTH_DAY,
        BlockComp.LONG_LAT_EXISTS,
        BlockComp.CLOSE_LONG_LAT,
        BlockComp.SAME_SEX)
    OFFENDER_AGE_SEX = _block_from_comps(
        BlockComp.SAME_OFFENDER_AGE_SEX
    )
    AGE_WEAPON = _block_from_comps(
        BlockComp.SAME_AGE,
        BlockComp.SAME_WEAPON
    )


# Backwards-compatible aliases for refactors still in flight.


NAMED_VICTIM_BLOCKS = [
    DedupBlockRule.SAME_NAMES_YEAR_DIFF2,
    DedupBlockRule.YEAR_MONTH,
    DedupBlockRule.DATE_LOCATION,
    DedupBlockRule.YEAR_SURNAME_SOUNDEX,
    DedupBlockRule.YEAR_FORENAME_SOUNDEX,
    DedupBlockRule.YEAR_AGE_SEX
]

NAMED_VICTIM_BLOCKS_FOR_TRAINING = _add_city(
    YEAR_MONTH,
    FULLNAME,
    AGE_SEX,
    OFFENDER_AGE_SEX
)

NAMED_VICTIM_BLOCKS_FOR_TRAINING_OLD = [
    DedupBlockRule.YEAR_MONTH,
    DedupBlockRule.SAME_FULLNAME,
    DedupBlockRule.AGE_SEX,
    DedupBlockRule.OFFENDER_AGE_SEX
]

NAMED_VICTIM_DETERMINISTIC_BLOCKS = [
    DedupBlockRule.SAME_NAMES_30DAYS,
    DedupBlockRule.DATE_LOCATION_AGE_SEX
]

ORPHAN_WIDE_BLOCKS = [
    DedupBlockRule.EVERYTHING
]

ORPHAN_VICTIM_BLOCKS = [
    DedupBlockRule.YEAR_MONTH,
    DedupBlockRule.DATE_LOCATION,
    DedupBlockRule.AGE_SEX,
    DedupBlockRule.AGE_WEAPON,
    DedupBlockRule.OFFENDER_AGE_SEX
]

ORPHAN_DETERMINISTIC_BLOCKS = [
    DedupBlockRule.DATE_LOCATION_SEX
]

ORPHAN_TRAINING_BLOCKS = [
    TrainBlockRule.YEAR_MONTH,
    TrainBlockRule.AGE_SEX_WITHIN_YEAR,
    TrainBlockRule.SEX_GEO_SHORT,
    TrainBlockRule.OFFENDER_AGE_SEX_WITHIN_YEAR,
]

SHR_OVERALL_BLOCKS = [
    TrainBlockRule.MIDPOINT_7MONTH,
    TrainBlockRule.AGE_SEX,
]

SHR_DETERMINISTIC_BLOCKS = [
    TrainBlockRule.MONTH_AGE_SEX_WEAPON
]

SHR_TRAINING_BLOCKS = [
    TrainBlockRule.YEAR_MONTH,
    TrainBlockRule.AGE_SEX,
    TrainBlockRule.OFFENDER_AGE_SEX,
]

YEAR_MONTH_COMPARISON = [
    "exact date or month precision match",
    "exact yr/mon or within 2 days"
]

TRAINING_BLOCK_LEVEL_MAP: TrainingBlockToComparisonLevelMap = HashMap.make({
    DedupBlockRule.YEAR_MONTH: comparison_level_keys(
        DATE_COMP,
        YEAR_MONTH_COMPARISON,
    ),
    TrainBlockRule.YEAR_MONTH: comparison_level_keys(
        DATE_COMP_ORPHAN,
        [
            "exact date or month precision match",
            "exact yr/mon or year precision match",
        ],
    ),
    YEAR_MONTH: comparison_level_keys(
        DATE_COMP,
        YEAR_MONTH_COMPARISON,
    ),
    TrainBlockRule.SEX_WITHIN_YEAR: comparison_level_keys(
        DATE_COMP_ORPHAN,
        [
            "exact date or month precision match",
            "exact yr/mon or year precision match",
            "within a year",
        ],
    ),
    TrainBlockRule.GEO_SHORT: comparison_level_keys(
        DIST_COMP,
        [
            "exact location match",
            "within 0.4 km or similar address",
        ],
    ),
    TrainBlockRule.OFFENDER_SEX_WITHIN_YEAR: comparison_level_keys(
        DATE_COMP_ORPHAN,
        [
            "exact date or month precision match",
            "exact yr/mon or year precision match",
            "within a year",
        ],
    ),
})
