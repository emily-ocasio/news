"""
Intent and smart constructors for Splink
Uses run_base_effect eliminator with REAL_DISPATCH
"""

from typing import Sequence

from splink.internals.blocking_rule_creator import BlockingRuleCreator

# pylint:disable=W0212
from .dispatch import SplinkDedupeJob
from .run import Run


def splink_dedupe_job(
    duckdb_path: str,
    input_table: str,
    settings: dict,
    predict_threshold: float,
    cluster_threshold: float,
    pairs_out: str = "incidents_pairs",
    clusters_out: str = "incidents_clusters",
    train_first: bool = False,
    training_blocking_rules: Sequence[str] | None = None,
    deterministic_rules: Sequence[str | BlockingRuleCreator] | None = None,
    deterministic_recall: float = 0.5
) -> Run[tuple[str, str]]:
    """
    Smart constructor for SplinkDedupeJob intent.
    Returns Run[(pairs_table_name, clusters_table_name)]
    """
    return Run(
        lambda self: self._perform(
            SplinkDedupeJob(
                duckdb_path,
                input_table,
                settings,
                predict_threshold,
                cluster_threshold,
                pairs_out,
                clusters_out,
                deterministic_rules or [],
                deterministic_recall,
                train_first,
                training_blocking_rules or [],
            ),
            self,
        ),
        lambda i, c: c._perform(i, c),
    )
