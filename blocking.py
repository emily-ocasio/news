"""
Classes and helper functions for defining blocking rules in Splink
"""

from enum import StrEnum

class BlockComp(StrEnum):
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

def _clause_from_comps(*components: StrEnum) -> str:
    return " AND ".join([component.value for component in components])

def _block_from_comps(
    *components: BlockComp, add_article_exclusion: bool = True
) -> str:
    comp_list = [BlockComp.SAME_CITY, *components]
    if add_article_exclusion:
        comp_list.append(BlockComp.DIFFERENT_ARTICLE)
    return _clause_from_comps(*comp_list)

def _train_block_from_comps(
    *components: BlockComp, add_article_exclusion: bool = False
) -> str:
    return _block_from_comps(*components, add_article_exclusion=add_article_exclusion)

class TrainBlockRule(StrEnum):
    """
    Predefined blocking rules for training
    """
    YEAR_MONTH = _train_block_from_comps(BlockComp.EXACT_YEAR_MONTH)
    SAME_NAMES = _train_block_from_comps(BlockComp.SAME_NAMES)
    LOCATION = _train_block_from_comps(BlockComp.LONG_LAT_EXISTS,
                                       BlockComp.CLOSE_LONG_LAT)
    AGE_SEX = _train_block_from_comps(BlockComp.SAME_AGE_SEX)

class DedupBlockRule(StrEnum):
    """
    Predefined blocking rules for deduplication
    """
    YEAR_MONTH = _block_from_comps(BlockComp.EXACT_YEAR_MONTH)
    YEAR_MONTH_DAY = _block_from_comps(BlockComp.EXACT_YEAR_MONTH_DAY)
    DATE_LOCATION = _block_from_comps(BlockComp.MIDPOINT_EXISTS,
                                        BlockComp.MIDPOINT_7MONTH,
                                        BlockComp.LONG_LAT_EXISTS,
                                        BlockComp.CLOSE_LONG_LAT)
    SAME_NAMES = _block_from_comps(BlockComp.SAME_NAMES)
    SURNAME_SOUNDEX = _block_from_comps(BlockComp.SAME_SURNAME_SOUNDEX)
    FORENAME_SOUNDEX = _block_from_comps(BlockComp.SAME_FORENAME_SOUNDEX)
    AGE_SEX = _block_from_comps(BlockComp.SAME_AGE_SEX)
    SAME_NAMES_30DAYS = _block_from_comps(
        BlockComp.SAME_NAMES, BlockComp.MIDPOINT_30DAYS)
    DATE_LOCATION_AGE_SEX = _block_from_comps(
        BlockComp.EXACT_YEAR_MONTH_DAY,
        BlockComp.LONG_LAT_EXISTS,
        BlockComp.CLOSE_LONG_LAT,
        BlockComp.SAME_AGE_SEX)

NAMED_VICTIM_BLOCKS = [
    DedupBlockRule.SAME_NAMES,
    DedupBlockRule.YEAR_MONTH,
    DedupBlockRule.DATE_LOCATION,
    DedupBlockRule.SURNAME_SOUNDEX,
    DedupBlockRule.FORENAME_SOUNDEX,
    DedupBlockRule.AGE_SEX
]

NAMED_VICTIM_BLOCKS_FOR_TRAINING = [
    TrainBlockRule.SAME_NAMES,
    TrainBlockRule.YEAR_MONTH,
    TrainBlockRule.LOCATION,
    TrainBlockRule.AGE_SEX
]

NAMED_VICTIM_DETERMINISTIC_BLOCKS = [
    DedupBlockRule.SAME_NAMES_30DAYS,
    DedupBlockRule.DATE_LOCATION_AGE_SEX
]
