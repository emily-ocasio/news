"""Splink runtime package."""

from .splink_api import (
    comparison_level_key,
    comparison_level_keys,
    splink_dedupe_job,
    splink_visualize_job,
)
from .splink_eliminator import run_splink
from .splink_types import (
    BlockingRuleLike,
    TrainingBlockToComparisonLevelMap,
    SplinkChartType,
    PredictionInputTableName,
    PredictionInputTableNames,
    PairsTableName,
    ClustersTableName,
    UniquePairsTableName,
    BlockedPairsTableName,
    DoNotLinkTableName,
    ComparisonLevelKey,
)

__all__ = [
    "comparison_level_key",
    "comparison_level_keys",
    "splink_dedupe_job",
    "splink_visualize_job",
    "run_splink",
    "BlockingRuleLike",
    "TrainingBlockToComparisonLevelMap",
    "SplinkChartType",
    "PredictionInputTableName",
    "PredictionInputTableNames",
    "PairsTableName",
    "ClustersTableName",
    "UniquePairsTableName",
    "BlockedPairsTableName",
    "DoNotLinkTableName",
    "ComparisonLevelKey",
]
