"""Splink runtime table assembly and naming helpers."""
from __future__ import annotations

from .splink_types import (
    BlockedPairsTableName,
    ClustersCountsTableName,
    ClustersTableName,
    DoNotLinkTableName,
    PairsCaptureTableName,
    PairsTableName,
    PairsTop1TableName,
    PredictionInputTableNames,
    SplinkLinkType,
    SplinkTableNames,
    UniquePairsTableName,
)
from ..run import ErrorPayload, Run, pure, throw
from ..monad import Unit, unit


def input_table_value(input_table: PredictionInputTableNames) -> str | list[str]:
    """Return a Splink input table as a single name or left/right table list."""
    left = input_table.left().value_or("")
    right = input_table.right()
    if right.is_present():
        return [left, right.value_or("")]
    return left


def add_all_tables(
    tables: SplinkTableNames,
    pairs_out: PairsTableName,
    input_tables: PredictionInputTableNames,
) -> SplinkTableNames:
    """Add core input and pairs table names to the table registry."""
    return tables.set(pairs_out).set(input_tables)


def add_link_type_tables(
    tables: SplinkTableNames,
    link_type: SplinkLinkType,
    pairs_out: PairsTableName,
) -> SplinkTableNames:
    """Add link-only derived table names when link mode requires them."""
    if link_type == SplinkLinkType.LINK_ONLY:
        return tables.set(PairsTop1TableName(f"{pairs_out}_top1"))
    return tables


def _add_capture_tables(
    tables: SplinkTableNames,
    capture_blocked_edges: bool,
    pairs_out: PairsTableName,
) -> SplinkTableNames:
    if capture_blocked_edges:
        return tables.set(PairsCaptureTableName(f"{pairs_out}_capture"))
    return tables


def _add_blocking_tables(
    tables: SplinkTableNames,
    do_not_link_table: DoNotLinkTableName,
    blocked_pairs_out: BlockedPairsTableName,
) -> SplinkTableNames:
    return tables.set(do_not_link_table).set(blocked_pairs_out)


def add_unique_matching_tables(
    tables: SplinkTableNames,
    unique_matching: bool,
    unique_pairs_table: UniquePairsTableName,
) -> SplinkTableNames:
    """Add unique-matching output table when unique matching is enabled."""
    if unique_matching:
        return tables.set(unique_pairs_table)
    return tables


def add_dedupe_tables(
    tables: SplinkTableNames,
    link_type: SplinkLinkType,
    clusters_out: ClustersTableName,
    do_not_link_table: DoNotLinkTableName,
    blocked_pairs_out: BlockedPairsTableName,
    pairs_out: PairsTableName,
) -> SplinkTableNames:
    """Add dedupe-only tables (clusters and blocking artifacts) to the registry."""
    if link_type != SplinkLinkType.DEDUPE_ONLY:
        return tables
    tables = tables.set(clusters_out).set(ClustersCountsTableName.from_clusters(clusters_out))
    tables = _add_blocking_tables(tables, do_not_link_table, blocked_pairs_out)
    tables = _add_capture_tables(tables, True, pairs_out)
    return tables


def validate_splink_dedupe_input_tables(
    input_tables: PredictionInputTableNames,
    link_type: SplinkLinkType,
    clusters_out: ClustersTableName,
    unique_matching: bool,
    unique_pairs_table: UniquePairsTableName,
    blocked_pairs_out: BlockedPairsTableName,
    do_not_link_table: DoNotLinkTableName,
) -> Run[Unit]:
    """Validate table-name and mode constraints for a dedupe job request."""
    if not input_tables.left().is_present():
        return throw(ErrorPayload("Input table name must be provided."))
    if link_type == SplinkLinkType.LINK_ONLY:
        if not input_tables.has_right():
            return throw(ErrorPayload(
                "Link-only mode requires exactly two input tables "
                f"(got {input_tables})."
            ))
        if clusters_out.is_present():
            return throw(ErrorPayload("Link-only mode requires clusters_out to be empty."))
        if blocked_pairs_out.is_present() or do_not_link_table.is_present():
            return throw(ErrorPayload("Link-only mode requires no blocked/exclusion tables."))
    else:
        if input_tables.has_right():
            return throw(ErrorPayload(
                "Dedupe mode requires a single input table "
                f"(got {input_tables})."
            ))
        if not clusters_out.is_present():
            return throw(ErrorPayload("Dedupe mode requires clusters_out to be set."))
        if not blocked_pairs_out.is_present() or not do_not_link_table.is_present():
            return throw(ErrorPayload("Dedupe mode requires blocked/exclusion table names."))
        if str(blocked_pairs_out) == str(do_not_link_table):
            return throw(ErrorPayload(
                "Dedupe mode requires blocked_pairs_out and do_not_link_table "
                "to be distinct."
            ))
    if unique_matching and not unique_pairs_table.is_present():
        return throw(ErrorPayload("unique_matching requires unique_pairs_table."))
    if not unique_matching and unique_pairs_table.is_present():
        return throw(ErrorPayload("unique_pairs_table is set but unique_matching is False."))
    return pure(unit)
