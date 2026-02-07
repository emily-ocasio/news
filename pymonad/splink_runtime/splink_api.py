"""Public Splink runtime API constructors and comparison-key helpers."""
# Run smart constructors intentionally call Run internals (_perform) to encode intents.
# pylint: disable=protected-access
from __future__ import annotations

from typing import Any, Sequence

from ..array import Array
from ..hashmap import HashMap
from ..run import Run
from .splink_types import (
    BlockingRuleCreator,
    ComparisonLevelKey,
    PredictionInputTableName,
    PredictionInputTableNames,
    PairsTableName,
    ClustersTableName,
    StringBlockingRule,
    StringBlockingRules,
    SplinkChartType,
    SplinkDedupeJob,
    SplinkVisualizeJob,
    TrainingBlockToComparisonLevelMap,
    UniquePairsTableName,
    BlockedPairsTableName,
    DoNotLinkTableName,
    BlockedIdLeftColumnName,
    BlockedIdRightColumnName,
    comparison_level_key,
    comparison_level_keys,
)


def splink_dedupe_job(
    input_table: PredictionInputTableName | PredictionInputTableNames | Sequence[str | PredictionInputTableName],
    settings: dict,
    predict_threshold: float = 0.05,
    cluster_threshold: float = 0,
    pairs_out: PairsTableName = PairsTableName("incidents_pairs"),
    clusters_out: ClustersTableName = ClustersTableName(""),
    train_first: bool = False,
    skip_u_estimation: bool = False,
    training_blocking_rules: StringBlockingRules | Sequence[StringBlockingRule] | Sequence[BlockingRuleCreator] | None = None,
    training_block_level_map: TrainingBlockToComparisonLevelMap = HashMap.empty(),
    deterministic_rules: StringBlockingRules | Sequence[StringBlockingRule] | None = None,
    deterministic_recall: float = 0.5,
    visualize: bool = False,
    unique_matching: bool = False,
    unique_pairs_table: UniquePairsTableName = UniquePairsTableName(""),
    em_max_runs: int = 3,
    em_min_runs: int = 1,
    em_stop_delta: float = 0.002,
    splink_key: Any = None,
    do_not_link_table: DoNotLinkTableName = DoNotLinkTableName(""),
    do_not_link_left_col: BlockedIdLeftColumnName = BlockedIdLeftColumnName("id_l"),
    do_not_link_right_col: BlockedIdRightColumnName = BlockedIdRightColumnName("id_r"),
    blocked_pairs_out: BlockedPairsTableName = BlockedPairsTableName(""),
) -> Run[tuple[Any, str, str]]:
    return Run(
        lambda self: self._perform(
            SplinkDedupeJob(
                PredictionInputTableNames.from_input(input_table),
                settings,
                predict_threshold,
                cluster_threshold,
                pairs_out,
                Array.make(tuple(deterministic_rules or ())),
                deterministic_recall,
                clusters_out,
                train_first,
                skip_u_estimation,
                Array.make(tuple(training_blocking_rules or ())),
                training_block_level_map,
                visualize,
                unique_matching,
                unique_pairs_table,
                em_max_runs,
                em_min_runs,
                em_stop_delta,
                splink_key,
                do_not_link_table,
                do_not_link_left_col,
                do_not_link_right_col,
                blocked_pairs_out,
            ),
            self,
        ),
        lambda i, c: c._perform(i, c),
    )


def splink_visualize_job(
    splink_key: Any,
    chart_type: SplinkChartType,
    left_midpoints: Sequence[int] | None = None,
    right_midpoints: Sequence[int] | None = None,
):
    return Run(
        lambda self: self._perform(
            SplinkVisualizeJob(
                splink_key=splink_key,
                chart_type=chart_type,
                left_midpoints=list(left_midpoints) if left_midpoints else None,
                right_midpoints=list(right_midpoints) if right_midpoints else None,
            ),
            self,
        ),
        lambda i, c: c._perform(i, c),
    )


__all__ = [
    "splink_dedupe_job",
    "splink_visualize_job",
    "comparison_level_key",
    "comparison_level_keys",
    "ComparisonLevelKey",
]
