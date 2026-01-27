"""
Intent, eliminator, and smart constructors for Splink.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import logging
import json
import math
from pathlib import Path
from typing import Any, Sequence, TypeVar, cast
import uuid

import altair as alt
import duckdb
import networkx as nx
import pandas as pd
from pandas import DataFrame
from splink import Linker, DuckDBAPI, blocking_analysis
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.blocking_rule_creator_utils import to_blocking_rule_creator
from splink.internals.cache_dict_with_logging import CacheDictWithLogging
from splink.internals.charts import unlinkables_chart
from splink.internals.pipeline import CTEPipeline

# pylint:disable=W0212
from .environment import DbBackend, EnvKey, Environment
from .run import Run, ask, throw, ErrorPayload, pure, put_splink_linker, \
    get_splink_linker, get_splink_context, put_splink_context, HasSplinkLinker
from .runsql import SQL, sql_exec, sql_query, sql_register, with_duckdb
from .array import Array
from .monad import Unit, unit
from .hashmap import HashMap, flatten_map
from .string import String
from .maybe import Just, Nothing, Maybe, from_maybe

A = TypeVar("A")

@dataclass(frozen=True)
class SplinkVisualizeJob:
    """
    Splink visualization intent
    """
    splink_key: Any
    chart_type: SplinkChartType
    left_midpoints: Sequence[int] | None = None
    right_midpoints: Sequence[int] | None = None


class SplinkChartType(str, Enum):
    """
    Visualization types for Splink.
    """
    MODEL = "model"
    PARAMETER_ESTIMATE_COMPARISONS = "parameter_estimate_comparisons"
    WATERFALL = "waterfall"
    COMPARISON = "comparison"
    CLUSTER = "cluster"
    UNLINKABLES = "unlinkables"

@dataclass(frozen=True)
class ComparisonLevelKey:
    """Key for a specific comparison level within a Splink model."""
    comparison_name: str
    level_name: str

@dataclass(frozen=True)
class EmParam:
    """EM m/u probabilities for a comparison level (optional via Maybe)."""
    m: Maybe[float]
    u: Maybe[float]

EmParams = HashMap[ComparisonLevelKey, EmParam]

class SplinkPairsSchemaError(Exception):
    """Raised when Splink pairwise output does not include expected id columns."""


class SplinkPhase(str, Enum):
    """Lifecycle phase for the Splink workflow."""
    INIT = "init"
    PREPARE = "prepare"
    PREDICT = "predict"
    CLUSTER = "cluster"
    PERSIST = "persist"
    DONE = "done"

class SplinkLinkType(str, Enum):
    """Splink link type."""
    DEDUPE_ONLY = "dedupe_only"
    LINK_ONLY = "link_only"

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> SplinkLinkType:
        raw = str(settings.get("link_type", cls.DEDUPE_ONLY.value))
        return cls.LINK_ONLY if raw == cls.LINK_ONLY.value else cls.DEDUPE_ONLY


BlockingRule = str | BlockingRuleCreator


@dataclass(frozen=True)
class TableRef:
    """Base type for Splink table references."""


@dataclass(frozen=True)
class TableName(TableRef):
    """Single table name."""
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TableNames(TableRef):
    """Multiple table names."""
    names: tuple[str, ...]

    def __str__(self) -> str:
        return ", ".join(self.names)


class PredictionInputTableName(TableName):
    """Input table name used for prediction."""


class PredictionInputTableNames(TableNames):
    """Input table names used for prediction."""


class PairsTableName(TableName):
    """Pairs output table name."""


class PairsCaptureTableName(TableName):
    """Pairs table for capture-blocked-edges diagnostics."""


class PairsTop1TableName(TableName):
    """Top-1 pairs table for link-only mode."""


class ClustersTableName(TableName):
    """Clusters output table name."""


class ClustersCountsTableName(TableName):
    """Clusters counts output table name."""

    @classmethod
    def from_clusters(cls, clusters: ClustersTableName) -> ClustersCountsTableName:
        return cls(f"{clusters}_counts")


class UniquePairsTableName(TableName):
    """Unique matching pairs output table name."""


class BlockedPairsTableName(TableName):
    """Blocked edges output table name."""


class DoNotLinkTableName(TableName):
    """Exclusion list table name."""


class ResultPairsTableName(TableName):
    """Actual pairs table returned by the dedupe run."""


class ResultClustersTableName(TableName):
    """Actual clusters table returned by the dedupe run."""


class ColumnName(String):
    """Base type for Splink column names."""


class UniqueIdColumnName(ColumnName):
    """Unique id column name."""


class PairLeftIdColumnName(ColumnName):
    """Left id column name for pair tables."""


class PairRightIdColumnName(ColumnName):
    """Right id column name for pair tables."""


class BlockedIdLeftColumnName(ColumnName):
    """Left id column name for blocked edges."""


class BlockedIdRightColumnName(ColumnName):
    """Right id column name for blocked edges."""


class SplinkId(String):
    """Base type for Splink id values."""


class UniqueId(SplinkId):
    """Unique id value."""


class ExclusionId(SplinkId):
    """Exclusion id value."""


class PairLeftId(SplinkId):
    """Left-side id value for pair rows."""


class PairRightId(SplinkId):
    """Right-side id value for pair rows."""


TableNameType = type[TableRef]
TTableName = TypeVar("TTableName", bound=TableRef)


@dataclass(frozen=True)
class SplinkTableNames:
    """Typed registry for Splink table names."""
    tables: HashMap[TableNameType, TableRef]

    @classmethod
    def empty(cls) -> SplinkTableNames:
        return cls(HashMap.empty())

    def set(self, value: TableRef) -> SplinkTableNames:
        return SplinkTableNames(self.tables.set(type(value), value))

    def get(self, key: type[TTableName]) -> TTableName | None:
        value = self.tables.get(key)
        if value is None:
            return None
        return cast(TTableName, value) if isinstance(value, key) else None


@dataclass(frozen=True)
class SplinkContext:
    """
    Monadic Splink workflow context (dedupe).
    """
    phase: SplinkPhase = SplinkPhase.INIT
    tables: SplinkTableNames = field(default_factory=SplinkTableNames.empty)
    unique_id_col: UniqueIdColumnName = UniqueIdColumnName("unique_id")
    prediction_rules: Array[BlockingRule] = field(default_factory=Array.empty)
    training_rules: Array[BlockingRule] = field(default_factory=Array.empty)
    linker: Linker | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    predict_threshold: float = 0.05
    cluster_threshold: float = 0.0
    deterministic_rules: Array[BlockingRule] = field(default_factory=Array.empty)
    deterministic_recall: float = 0.5
    train_first: bool = False
    visualize: bool = False
    unique_matching: bool = False
    em_max_runs: int = 3
    em_min_runs: int = 1
    em_stop_delta: float = 0.002
    capture_blocked_edges: bool = False


@dataclass(frozen=True)
class SplinkClusterInputs:
    """
    Derived inputs for clustering (computed immediately prior to clustering).
    """
    nodes: Array[SplinkNode]
    edges: Array[SplinkEdge]
    unique_id_col: UniqueIdColumnName
    left_id_col: PairLeftIdColumnName
    right_id_col: PairRightIdColumnName
    pairs_table: PairsTableName
    clusters_out: ClustersTableName
    capture_blocked: bool
    blocked_pairs_out: BlockedPairsTableName
    blocked_id_col_l: BlockedIdLeftColumnName
    blocked_id_col_r: BlockedIdRightColumnName


@dataclass(frozen=True)
class SplinkNode:
    """Node descriptor for constrained clustering."""
    unique_id: UniqueId
    exclusion_id: ExclusionId


@dataclass(frozen=True)
class SplinkEdge:
    """Edge descriptor for constrained clustering."""
    left_id: PairLeftId
    right_id: PairRightId
    match_probability: float


@dataclass(frozen=True)
class SplinkPredictResult:
    """
    Outputs from prediction/pairs step needed for downstream steps.
    """
    linker: Linker
    input_table_for_prediction: TableName | TableNames
    unique_id_col: UniqueIdColumnName
    pairs_out: PairsTableName
    clusters_out: ClustersTableName
    do_not_link_table: DoNotLinkTableName | None
    do_not_link_left_col: BlockedIdLeftColumnName
    do_not_link_right_col: BlockedIdRightColumnName
    blocked_pairs_out: BlockedPairsTableName
    unique_pairs_table: UniquePairsTableName


@dataclass(frozen=True)
class SplinkDedupeJob:
    """
    Splink deduplication intent
    """
    input_table: PredictionInputTableName | PredictionInputTableNames
    settings: dict
    predict_threshold: float
    cluster_threshold: float
    pairs_out: PairsTableName
    deterministic_rules: Sequence[str | BlockingRuleCreator]
    deterministic_recall: float
    clusters_out: ClustersTableName = ClustersTableName("")
    train_first: bool = False
    training_blocking_rules: Sequence[str] | None = None
    visualize: bool = False
    unique_matching: bool = False
    unique_pairs_table: UniquePairsTableName = UniquePairsTableName("")
    em_max_runs: int = 3
    em_min_runs: int = 1
    em_stop_delta: float = 0.002
    splink_key: Any = None
    do_not_link_table: DoNotLinkTableName = DoNotLinkTableName("")
    do_not_link_left_col: BlockedIdLeftColumnName = field(
        default_factory=lambda: BlockedIdLeftColumnName("id_l")
    )
    do_not_link_right_col: BlockedIdRightColumnName = field(
        default_factory=lambda: BlockedIdRightColumnName("id_r")
    )
    capture_blocked_edges: bool = False
    blocked_pairs_out: BlockedPairsTableName = BlockedPairsTableName("")


def _require_splink_context() -> Run[SplinkContext]:
    return get_splink_context() >> (
        lambda ctx: throw(
            ErrorPayload("Splink context is not initialized.")
        ) if ctx is None else pure(ctx)
    )


def _update_splink_context(update_fn) -> Run[Unit]:
    return _require_splink_context() >> (
        lambda ctx: put_splink_context(update_fn(ctx)) ^ pure(unit)
    )


def _tables_get_required(
    tables: SplinkTableNames,
    key: type[TTableName],
) -> Run[TTableName]:
    value = tables.get(key)
    if value is None:
        return throw(ErrorPayload(f"Missing table name for {key.__name__}."))
    if not isinstance(value, key):
        return throw(
            ErrorPayload(f"Expected {key.__name__}, got {type(value).__name__}.")
        )
    return pure(value)


def _tables_get_optional(
    tables: SplinkTableNames,
    key: type[TTableName],
) -> Run[Maybe[TTableName]]:
    value = tables.get(key)
    if value is None:
        return pure(Nothing)
    if not isinstance(value, key):
        return throw(
            ErrorPayload(f"Expected {key.__name__}, got {type(value).__name__}.")
        )
    return pure(Just(value))


def _table_name_value(table: TableName) -> str:
    return table.name


def _table_names_value(tables: TableNames) -> list[str]:
    return list(tables.names)


def _input_table_value(
    input_table: PredictionInputTableName | PredictionInputTableNames,
) -> str | list[str]:
    if isinstance(input_table, PredictionInputTableName):
        return input_table.name
    return _table_names_value(input_table)

def _add_all_tables(
    tables: SplinkTableNames,
    pairs_out: PairsTableName,
    input_tables: PredictionInputTableName | PredictionInputTableNames,
) -> SplinkTableNames:
    return (
        tables
        .set(pairs_out)
        .set(input_tables)
    )


def _add_link_type_tables(
    tables: SplinkTableNames,
    link_type: SplinkLinkType,
    pairs_out: PairsTableName,
) -> SplinkTableNames:
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
    return (
        tables
        .set(do_not_link_table)
        .set(blocked_pairs_out)
    )


def _add_unique_matching_tables(
    tables: SplinkTableNames,
    unique_matching: bool,
    unique_pairs_table: UniquePairsTableName,
) -> SplinkTableNames:
    if unique_matching:
        return tables.set(cast(UniquePairsTableName, unique_pairs_table))
    return tables


def _add_dedupe_tables(
    tables: SplinkTableNames,
    link_type: SplinkLinkType,
    clusters_out: ClustersTableName,
    do_not_link_table: DoNotLinkTableName,
    blocked_pairs_out: BlockedPairsTableName,
    capture_blocked_edges: bool,
    pairs_out: PairsTableName,
) -> SplinkTableNames:
    if link_type != SplinkLinkType.DEDUPE_ONLY:
        return tables
    tables = (
        tables
        .set(clusters_out)
        .set(ClustersCountsTableName.from_clusters(clusters_out))
    )
    tables = _add_blocking_tables(tables, do_not_link_table, blocked_pairs_out)
    tables = _add_capture_tables(tables, capture_blocked_edges, pairs_out)
    return tables
def _validate_splink_dedupe_input_tables(
    input_tables: PredictionInputTableName | PredictionInputTableNames,
    link_type: SplinkLinkType,
    clusters_out: ClustersTableName,
    unique_matching: bool,
    unique_pairs_table: UniquePairsTableName,
    capture_blocked_edges: bool,
    blocked_pairs_out: BlockedPairsTableName,
    do_not_link_table: DoNotLinkTableName,
) -> Run[Unit]:
    if link_type == SplinkLinkType.LINK_ONLY:
        if not isinstance(input_tables, PredictionInputTableNames) or len(input_tables.names) != 2:
            return throw(
                ErrorPayload(
                    "Link-only mode requires exactly two input tables "
                    f"(got {input_tables})."
                )
            )
        if clusters_out.name != "":
            return throw(
                ErrorPayload("Link-only mode requires clusters_out to be empty.")
            )
        if capture_blocked_edges:
            return throw(
                ErrorPayload("Link-only mode does not support capture_blocked_edges.")
            )
        if blocked_pairs_out.name != "" or do_not_link_table.name != "":
            return throw(
                ErrorPayload("Link-only mode requires no blocked/exclusion tables.")
            )
    else:
        if isinstance(input_tables, PredictionInputTableNames):
            return throw(
                ErrorPayload(
                    "Dedupe mode requires a single input table "
                    f"(got {input_tables})."
                )
            )
        if clusters_out.name == "":
            return throw(
                ErrorPayload("Dedupe mode requires clusters_out to be set.")
            )
        if not capture_blocked_edges:
            return throw(
                ErrorPayload("Dedupe mode requires capture_blocked_edges to be True.")
            )
        if blocked_pairs_out.name == "" or do_not_link_table.name == "":
            return throw(
                ErrorPayload("Dedupe mode requires blocked/exclusion table names.")
            )
        if blocked_pairs_out.name == do_not_link_table.name:
            return throw(
                ErrorPayload(
                    "Dedupe mode requires blocked_pairs_out and do_not_link_table "
                    "to be distinct."
                )
            )
    if unique_matching and unique_pairs_table.name == "":
        return throw(
            ErrorPayload("unique_matching requires unique_pairs_table.")
        )
    if not unique_matching and unique_pairs_table.name != "":
        return throw(
            ErrorPayload("unique_pairs_table is set but unique_matching is False.")
        )
    return pure(unit)

def _extract_em_params(
        settings_dict: dict[str, Any]
        ) -> EmParams:
    """
    Flatten comparison level m/u probabilities to a stable HashMap
    keyed by (comparison_name, level_name).
    """
    def _level_entry(level: dict[str, Any]):
        raw_name = level.get("label_for_charts", "")
        level_name = str(raw_name).strip()
        maybe_key = Just(level_name) if level_name else Nothing
        return (
            maybe_key,
            EmParam(
                m=_maybe_float(level.get("m_probability")),
                u=_maybe_float(level.get("u_probability")),
            ),
        )

    def _comparison_map(
            comparison: dict[str, Any]
        ) -> tuple[Maybe[str], HashMap[str, EmParam]]:
        output_name = str(comparison.get("output_column_name", ""))
        description = str(comparison.get("comparison_description", ""))
        comp_name = f"{output_name}_{description}"
        level_map = HashMap.from_iterable(
            comparison.get("comparison_levels", []),
            _level_entry,
        )
        return Just(comp_name), level_map

    nested = HashMap.from_iterable(
        settings_dict.get("comparisons", []),
        _comparison_map,
    )
    return flatten_map(
        nested,
        lambda comp_name, level_name: ComparisonLevelKey(
            comparison_name=comp_name,
            level_name=level_name,
        ),
    )

def _maybe_float(value: Any) -> Maybe[float]:
    if value is None:
        return Nothing
    return Just(value)


def _resolve_pair_id_cols(
    pair_cols: set[str],
    unique_id_column_name: str,
    pairs_table: str,
) -> tuple[str, str]:
    """
    Determine the left/right id columns in a pairs table by inspecting its schema.

    Splink may emit either `{unique_id_column_name}_l/_r` or the legacy
    `unique_id_l/unique_id_r`. We resolve against the actual table columns to
    keep downstream queries robust across settings and Splink versions.
    """
    left_id_col = f"{unique_id_column_name}_l"
    right_id_col = f"{unique_id_column_name}_r"
    if left_id_col in pair_cols and right_id_col in pair_cols:
        return left_id_col, right_id_col
    if "unique_id_l" in pair_cols and "unique_id_r" in pair_cols:
        return "unique_id_l", "unique_id_r"
    raise SplinkPairsSchemaError(
        f"Constrained clustering cannot find id columns in {pairs_table}. "
        f"Expected ({left_id_col}, {right_id_col}) or (unique_id_l, unique_id_r)."
    )

def _with_temp_blocking_rules_on_linker(
    linker: Linker,
    rules: Sequence[str | BlockingRuleCreator],
    action,
):
    settings_obj = linker._settings_obj
    old_rules = settings_obj._blocking_rules_to_generate_predictions
    try:
        sql_dialect = cast(str, linker._db_api.sql_dialect.sql_dialect_str)
        rules_array = Array.make(tuple(rules))
        dialected = rules_array.map(
            lambda rule: to_blocking_rule_creator(rule).get_blocking_rule(sql_dialect)
        )
        blocking_rules_dialected = dialected.map_with_index(
            lambda idx, br: (
                br.add_preceding_rules(list(dialected.a[:idx])),
                br,
            )[1]
        )
        settings_obj._blocking_rules_to_generate_predictions = list(blocking_rules_dialected)
        return action()
    finally:
        settings_obj._blocking_rules_to_generate_predictions = old_rules

def _append_clause_to_rules(
    rules: Sequence[str | BlockingRuleCreator],
    clause: str,
) -> list[str | BlockingRuleCreator]:
    updated: list[str | BlockingRuleCreator] = []
    for rule in rules:
        if isinstance(rule, BlockingRuleCreator):
            updated.append(rule)
        else:
            updated.append(f"({rule}) AND {clause}")
    return updated


def _create_exclusion_list_table(
    *,
    input_table: str,
    output_table: str,
    do_not_link_table: str,
    unique_id_column_name: str,
    id_left_col: str,
    id_right_col: str,
    exec_fn,
) -> None:
    exec_fn(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH pairs AS (
          SELECT
            CAST({id_left_col} AS VARCHAR) AS victim_id,
            CAST({id_right_col} AS VARCHAR) AS other_id
          FROM {do_not_link_table}
          UNION ALL
          SELECT
            CAST({id_right_col} AS VARCHAR) AS victim_id,
            CAST({id_left_col} AS VARCHAR) AS other_id
          FROM {do_not_link_table}
        ),
        agg AS (
          SELECT
            victim_id,
            array_agg(DISTINCT other_id) AS exclusion_ids
          FROM pairs
          GROUP BY victim_id
        )
        SELECT
          v.*,
          COALESCE(a.exclusion_ids, []::VARCHAR[]) AS exclusion_ids
        FROM {input_table} v
        LEFT JOIN agg a
          ON CAST(v.{unique_id_column_name} AS VARCHAR) = a.victim_id
        """
    )

def _filter_pairs_table_do_not_link(
    *,
    pairs_table: str,
    do_not_link_table: str,
    id_left_col: str,
    id_right_col: str,
    unique_id_column_name: str,
    query_fn,
    exec_fn,
) -> None:
    pair_cols = {
        row["name"]
        for row in query_fn(
            f"SELECT name FROM pragma_table_info('{pairs_table}')"
        )
    }
    left_id_col, right_id_col = _resolve_pair_id_cols(
        pair_cols,
        unique_id_column_name,
        pairs_table,
    )
    exec_fn(
        f"""
        CREATE OR REPLACE TABLE {pairs_table} AS
        SELECT p.*
        FROM {pairs_table} p
        WHERE NOT EXISTS (
          SELECT 1
          FROM {do_not_link_table} d
          WHERE (
            CAST(d.{id_left_col} AS VARCHAR) = CAST(p.{left_id_col} AS VARCHAR)
            AND CAST(d.{id_right_col} AS VARCHAR) = CAST(p.{right_id_col} AS VARCHAR)
          ) OR (
            CAST(d.{id_left_col} AS VARCHAR) = CAST(p.{right_id_col} AS VARCHAR)
            AND CAST(d.{id_right_col} AS VARCHAR) = CAST(p.{left_id_col} AS VARCHAR)
          )
        )
        """
    )

def _collect_blocked_edges(
    *,
    pairs_table: str,
    do_not_link_table: str,
    id_left_col: str,
    id_right_col: str,
    unique_id_column_name: str,
    query_fn,
    exec_fn,
) -> None:
    pair_cols = {
        row["name"]
        for row in query_fn(
            f"SELECT name FROM pragma_table_info('{pairs_table}')"
        )
    }
    left_id_col, right_id_col = _resolve_pair_id_cols(
        pair_cols,
        unique_id_column_name,
        pairs_table,
    )
    if "exclusion_id_l" not in pair_cols or "exclusion_id_r" not in pair_cols:
        raise SplinkPairsSchemaError(
            f"Blocked edge capture requires exclusion_id_l/exclusion_id_r in {pairs_table}."
        )
    exec_fn(
        f"""
        CREATE OR REPLACE TABLE {do_not_link_table} AS
        SELECT DISTINCT
          CAST({left_id_col} AS VARCHAR) AS {id_left_col},
          CAST({right_id_col} AS VARCHAR) AS {id_right_col}
        FROM {pairs_table}
        WHERE CAST(exclusion_id_l AS VARCHAR) = CAST(exclusion_id_r AS VARCHAR)
          AND CAST(exclusion_id_l AS VARCHAR) NOT IN ('', 'None')
          AND CAST({left_id_col} AS VARCHAR) <> CAST({right_id_col} AS VARCHAR)
        """
    )

def _drop_all_splink_tables(exec_fn, query_fn) -> None:
    rows = query_fn(
        "SELECT table_name, table_type "
        "FROM information_schema.tables "
        "WHERE table_name LIKE '__splink__%'"
    )
    for row in rows:
        name = row["table_name"]
        table_type = str(row["table_type"]).upper()
        if table_type == "VIEW":
            exec_fn(f"DROP VIEW IF EXISTS {name}")
        else:
            exec_fn(f"DROP TABLE IF EXISTS {name}")


def _constrained_greedy_clusters(
    *,
    nodes: list[tuple[str, str]],
    edges: list[tuple[str, str, float]],
    unique_id_column_name: str,
    capture_blocked: bool = False,
    blocked_id_cols: tuple[str, str] = ("id_l", "id_r"),
) -> tuple[DataFrame, DataFrame | None]:
    """
    Build clusters using a greedy union-find over pairwise edges, with a hard constraint:
    no resulting cluster may contain two rows with the same exclusion id (e.g. article id).
    """
    unique_ids: list[str] = []
    exclusion_by_id: dict[str, str] = {}
    for uid, excl in nodes:
        uid_str = "" if uid is None else str(uid)
        excl_str = "" if excl is None else str(excl)
        unique_ids.append(uid_str)
        exclusion_by_id[uid_str] = excl_str

    parent: dict[str, str] = {uid: uid for uid in unique_ids}
    size: dict[str, int] = {uid: 1 for uid in unique_ids}
    exclusions_in_component: dict[str, set[str]] = {}
    for uid in unique_ids:
        excl = exclusion_by_id.get(uid, "")
        exclusions_in_component[uid] = set([excl]) if excl not in ("", "None") else set()

    def find(x: str) -> str:
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        sa = exclusions_in_component[ra]
        sb = exclusions_in_component[rb]
        return len(sa.intersection(sb)) == 0

    def union_status(a: str, b: str) -> tuple[bool, set[str], bool]:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False, set(), True
        shared = exclusions_in_component[ra].intersection(exclusions_in_component[rb])
        return len(shared) == 0, shared, False

    def union(a: str, b: str) -> bool:
        if not can_union(a, b):
            return False
        ra, rb = find(a), find(b)
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        exclusions_in_component[ra].update(exclusions_in_component[rb])
        return True

    # Ensure we don't error if an edge references a missing node (shouldn't happen).
    known = set(parent.keys())
    blocked_rows: list[dict[str, Any]] = []
    id_left_col, id_right_col = blocked_id_cols
    for uid_l, uid_r, prob in edges:
        if uid_l in known and uid_r in known:
            ok, shared, same_component = union_status(uid_l, uid_r)
            if ok:
                union(uid_l, uid_r)
            elif capture_blocked and not same_component and shared:
                print(
                    "Blocked union: "
                    f"{uid_l} vs {uid_r} "
                    f"shared_exclusion_ids={sorted(shared)} "
                    f"match_probability={prob}"
                )
                blocked_rows.append(
                    {
                        id_left_col: uid_l,
                        id_right_col: uid_r,
                        "match_probability": prob,
                        "shared_exclusion_ids": ",".join(sorted(shared)),
                    }
                )

    # Materialize components (including singletons) deterministically.
    components: dict[str, list[str]] = {}
    for uid in unique_ids:
        root = find(uid)
        components.setdefault(root, []).append(uid)
    members_sorted = []
    for members in components.values():
        members.sort()
        members_sorted.append(members)
    members_sorted.sort(key=lambda ms: ms[0] if ms else "")

    rows: list[dict[str, str]] = []
    for members in members_sorted:
        # Match Splink's default connected-components behavior where cluster_id is the
        # lexicographically smallest member id in the cluster.
        cluster_id = members[0] if members else ""
        for uid in members:
            rows.append({"cluster_id": cluster_id, unique_id_column_name: uid})

    clusters_df = DataFrame(rows, columns=["cluster_id", unique_id_column_name])
    blocked_df = None
    if capture_blocked:
        blocked_df = DataFrame(
            blocked_rows,
            columns=[id_left_col, id_right_col, "match_probability", "shared_exclusion_ids"],
        )
    return clusters_df, blocked_df


def splink_dedupe_job(
    input_table: PredictionInputTableName | PredictionInputTableNames,
    settings: dict,
    predict_threshold: float = 0.05,
    cluster_threshold: float = 0,
    pairs_out: PairsTableName = PairsTableName("incidents_pairs"),
    clusters_out: ClustersTableName = ClustersTableName(""),
    train_first: bool = False,
    training_blocking_rules: Sequence[str] | None = None,
    deterministic_rules: Sequence[str | BlockingRuleCreator] | None = None,
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
    capture_blocked_edges: bool = False,
    blocked_pairs_out: BlockedPairsTableName = BlockedPairsTableName("")


) -> Run[tuple[Any, str, str]]:
    """
    Smart constructor for SplinkDedupeJob intent.
    Returns Run[(linker, pairs_table_name, clusters_table_name)]
    """
    return Run(
        lambda self: self._perform(
            SplinkDedupeJob(
                input_table,
                settings,
                predict_threshold,
                cluster_threshold,
                pairs_out,
                deterministic_rules or [],
                deterministic_recall,
                clusters_out,
                train_first,
                training_blocking_rules or [],
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
                capture_blocked_edges,
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
) -> Run[Unit]:
    """
    Smart constructor for SplinkVisualizeJob intent.
    """
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


def _run_splink_visualize(linker: Linker, job: SplinkVisualizeJob) -> None:
    alt.renderers.enable("browser")
    settings = linker.misc.save_model_to_json(out_path=None)
    link_type = SplinkLinkType.from_settings(settings)
    unique_id_col = settings.get("unique_id_column_name", "unique_id")
    left_id_col: str | None = f"{unique_id_col}_l"
    right_id_col: str | None = f"{unique_id_col}_r"
    df_clustered = None
    clustered_df = None
    cluster_id_map: dict[Any, Any] = {}

    def _midpoint_blocking_rule(
        left_midpoints: Sequence[int] | None,
        right_midpoints: Sequence[int] | None,
    ) -> str | None:
        clauses: list[str] = []
        if left_midpoints:
            left_vals = ",".join(str(int(v)) for v in left_midpoints)
            clauses.append(f"l.midpoint_day in ({left_vals})")
        if right_midpoints:
            right_vals = ",".join(str(int(v)) for v in right_midpoints)
            clauses.append(f"r.midpoint_day in ({right_vals})")
        if not clauses:
            return None
        return " and ".join(clauses)

    def _sql_literal(value: Any) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def _distinct_sources(self_link_table: str, source_col: str) -> list[Any]:
        source_col_sql = f'"{source_col}"'
        suffix = abs(hash((self_link_table, source_col))) % 1000000
        pipeline = CTEPipeline()
        sql = (
            f"select distinct {source_col_sql} as source_dataset "
            f"from {self_link_table} where {source_col_sql} is not null"
        )
        pipeline.enqueue_sql(sql, f"__splink__unlinkable_sources_{suffix}")
        data = linker._db_api.sql_pipeline_to_splink_dataframe(pipeline, use_cache=False)
        rows = data.as_record_dict()
        data.drop_table_from_database_and_remove_from_cache()
        return [row["source_dataset"] for row in rows]

    def _unlinkables_records(
        self_link_table: str,
        where_clause: str | None = None,
    ) -> list[dict[str, Any]]:
        suffix = abs(hash((self_link_table, where_clause))) % 1000000
        round_table = f"__splink__df_round_self_link_{suffix}"
        prop_table = f"__splink__df_unlinkables_proportions_{suffix}"
        cum_table = f"__splink__df_unlinkables_proportions_cumulative_{suffix}"
        pipeline = CTEPipeline()
        where_sql = f"where {where_clause}" if where_clause else ""
        sql = f"""
            select
            round(match_weight, 2) as match_weight,
            round(match_probability, 5) as match_probability
            from {self_link_table}
            {where_sql}
        """
        pipeline.enqueue_sql(sql, round_table)
        sql = f"""
            select
            max(match_weight) as match_weight,
            match_probability,
            count(*) / cast( sum(count(*)) over () as float) as prop
            from {round_table}
            group by match_probability
            order by match_probability
        """
        pipeline.enqueue_sql(sql, prop_table)
        sql = f"""
            select *,
            sum(prop) over(order by match_probability) as cum_prop
            from {prop_table}
            where match_probability < 1
        """
        pipeline.enqueue_sql(sql, cum_table)
        data = linker._db_api.sql_pipeline_to_splink_dataframe(pipeline, use_cache=False)
        records = data.as_record_dict()
        data.drop_table_from_database_and_remove_from_cache()
        return records

    def _unlinkables_records_for_source(
        self_link_table: str,
        source_col: str,
        source_value: Any,
    ) -> list[dict[str, Any]]:
        source_col_sql = f'"{source_col}"'
        source_literal = _sql_literal(source_value)
        where_clause = f"{source_col_sql} = {source_literal}"
        return _unlinkables_records(self_link_table, where_clause)

    def _with_cache_uid(new_uid: str, action):
        old_uid = linker._db_api._cache_uid
        linker._db_api._cache_uid = new_uid
        try:
            return action()
        finally:
            linker._db_api._cache_uid = old_uid

    def _with_isolated_cache(action):
        db_api = linker._db_api
        old_cache = db_api._intermediate_table_cache
        new_cache = CacheDictWithLogging()
        db_api._intermediate_table_cache = new_cache
        try:
            return action(new_cache)
        finally:
            db_api._intermediate_table_cache = old_cache

    if job.chart_type == SplinkChartType.MODEL:
        chart = linker.visualisations.match_weights_chart()
        chart.show()  # type: ignore
        chart = linker.visualisations.m_u_parameters_chart()
        chart.show()  # type: ignore
        return
    if job.chart_type == SplinkChartType.PARAMETER_ESTIMATE_COMPARISONS:
        chart = linker.visualisations.parameter_estimate_comparisons_chart()
        chart.show()  # type: ignore
        return
    if job.chart_type == SplinkChartType.UNLINKABLES:
        print("Generating unlinkables chart…")
        if link_type == SplinkLinkType.LINK_ONLY.value and len(linker._input_tables_dict) == 2:
            self_link_df = _with_cache_uid(
                f"unlinkables_{uuid.uuid4().hex[:8]}",
                linker._self_link,
            )
            col_names = [col.name for col in self_link_df.columns]
            source_col = next(
                (
                    col
                    for col in ("source_dataset_l", "source_dataset", "source_dataset_r")
                    if col in col_names
                ),
                None,
            )
            if source_col:
                sources = _distinct_sources(self_link_df.physical_name, source_col)
                if sources:
                    for source in sources:
                        print(f"Generating unlinkables chart for {source}…")
                        records = _unlinkables_records_for_source(
                            self_link_df.physical_name,
                            source_col,
                            source,
                        )
                        chart = unlinkables_chart(records, "match_weight", str(source))
                        chart.show()  # type: ignore
                    self_link_df.drop_table_from_database_and_remove_from_cache()
                    return
            print("Source dataset column not found; using combined unlinkables chart.")
            for alias, df in linker._input_tables_dict.items():
                label = df.physical_name
                print(f"Generating unlinkables chart for {label}…")
                single_settings = dict(settings)
                single_settings["link_type"] = "dedupe_only"
                single_settings.pop("source_dataset_column_name", None)
                single_settings["linker_uid"] = f"{label}_{uuid.uuid4().hex[:8]}"
                temp_linker = Linker(
                    df.physical_name,
                    single_settings,
                    db_api=linker._db_api,
                )
                def _run_self_link(cache):
                    temp_linker._intermediate_table_cache = cache
                    return _with_cache_uid(
                        f"unlinkables_{uuid.uuid4().hex[:8]}",
                        temp_linker._self_link,
                    )
                self_link_df = _with_isolated_cache(
                    _run_self_link
                )
                records = _unlinkables_records(self_link_df.physical_name)
                chart = unlinkables_chart(records, "match_weight", str(label))
                chart.show()  # type: ignore
                self_link_df.drop_table_from_database_and_remove_from_cache()
            return
        chart = linker.evaluation.unlinkables_chart()
        chart.show()  # type: ignore
        return

    if job.chart_type == SplinkChartType.COMPARISON:
        def _query_dicts(sql: str) -> list[dict]:
            rel = linker._db_api._execute_sql_against_backend(sql)
            cols = [d[0] for d in (rel.description or [])]
            rows = rel.fetchall()
            return [dict(zip(cols, row)) for row in rows]
        try:
            linker.table_management.invalidate_cache()
            _drop_all_splink_tables(linker._db_api._execute_sql_against_backend, _query_dicts)
        except Exception: #pylint: disable=W0718
            pass

    use_midpoint_blocking = job.chart_type in (
        SplinkChartType.WATERFALL,
        SplinkChartType.CLUSTER,
    ) and (job.left_midpoints or job.right_midpoints)
    if use_midpoint_blocking:
        rule = _midpoint_blocking_rule(job.left_midpoints, job.right_midpoints)
        if rule:
            print("Using midpoint-only blocking for prediction.")
            df_pairs = _with_temp_blocking_rules_on_linker(
                linker,
                [rule],
                lambda: linker.inference.predict(threshold_match_probability=0),
            )
        else:
            df_pairs = linker.inference.predict(threshold_match_probability=0)
    else:
        df_pairs = linker.inference.predict(threshold_match_probability=0)
    pd_pairs = df_pairs.as_pandas_dataframe()

    inspect_df = pd_pairs
    inspect_ids: set[Any] = set()
    if job.chart_type in (SplinkChartType.WATERFALL, SplinkChartType.CLUSTER):
        if job.left_midpoints:
            if "midpoint_day_l" in inspect_df.columns:
                inspect_df = inspect_df[inspect_df["midpoint_day_l"].isin(job.left_midpoints)]
            else:
                print("midpoint_day_l column not found; skipping left midpoint filter.")
        if job.right_midpoints:
            if "midpoint_day_r" in inspect_df.columns:
                inspect_df = inspect_df[inspect_df["midpoint_day_r"].isin(job.right_midpoints)]
            else:
                print("midpoint_day_r column not found; skipping right midpoint filter.")

        print(f"Total predictions within threshold: {len(pd_pairs)}")
        print(f"Number of records in selection: {len(inspect_df)}\n")
        if job.chart_type == SplinkChartType.WATERFALL:
            print("Waterfall chart members:")
        else:
            print("Cluster chart members:")

        if left_id_col not in inspect_df.columns or right_id_col not in inspect_df.columns:
            if "unique_id_l" in inspect_df.columns and "unique_id_r" in inspect_df.columns:
                left_id_col, right_id_col = "unique_id_l", "unique_id_r"
            else:
                left_id_col, right_id_col = None, None

        if left_id_col and left_id_col in inspect_df.columns:
            inspect_ids.update(inspect_df[left_id_col].dropna().tolist())
        if right_id_col and right_id_col in inspect_df.columns:
            inspect_ids.update(inspect_df[right_id_col].dropna().tolist())

        if job.chart_type == SplinkChartType.CLUSTER and left_id_col:
            try:
                if df_clustered is None:
                    df_clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
                        df_pairs, 0.01
                    )
                clustered_df = df_clustered.as_pandas_dataframe()
                cluster_id_map = dict(
                    zip(clustered_df[unique_id_col], clustered_df["cluster_id"])
                )
                cluster_id_series = None
                if left_id_col in inspect_df.columns:
                    cluster_id_series = inspect_df[left_id_col].map(cluster_id_map)
                if right_id_col and right_id_col in inspect_df.columns:
                    right_cluster = inspect_df[right_id_col].map(cluster_id_map)
                    if cluster_id_series is None:
                        cluster_id_series = right_cluster
                    else:
                        cluster_id_series = cluster_id_series.combine_first(right_cluster)
                if cluster_id_series is not None:
                    inspect_df = inspect_df.assign(cluster_id=cluster_id_series)
            except Exception as exc: #pylint: disable=W0718
                print(f"Unable to attach cluster_id to members table: {exc}")

        display_cols = [
            col for col in [
                "cluster_id",
                "match_probability",
                left_id_col,
                right_id_col,
                "midpoint_day_l",
                "midpoint_day_r",
            ]
            if col and col in inspect_df.columns
        ]
        if display_cols:
            print_df = inspect_df[display_cols].reset_index()
            print(print_df)
        else:
            print(inspect_df.reset_index())

    if job.chart_type == SplinkChartType.WATERFALL:
        if len(inspect_df) > 0:
            inspect_dict = cast(list[dict[str, Any]], inspect_df.to_dict(orient="records"))
            waterfall = linker.visualisations.waterfall_chart(inspect_dict)
            waterfall.show()  # type: ignore
        else:
            print("No records match the requested midpoint filters; skipping waterfall chart.")
        return

    if job.chart_type == SplinkChartType.COMPARISON:
        print("\nGenerating comparison viewer dashboard…")
        linker.visualisations.comparison_viewer_dashboard(
            df_pairs,
            "comparison_viewer.html",
            overwrite=True,
            num_example_rows=5,
        )
        print("Comparison viewer dashboard written to comparison_viewer.html")
        return

    if job.chart_type != SplinkChartType.CLUSTER:
        return

    if link_type != SplinkLinkType.DEDUPE_ONLY:
        print("Cluster studio charts are only available for dedupe models.")
        return

    if df_clustered is None:
        print("\nGenerating unconstrained clusters for charting…")
        df_clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
            df_pairs, 0.01
        )
    else:
        print("\nUsing cached unconstrained clusters for charting…")
    print("\nGenerating Cluster Studio dashboard…")
    try:
        if clustered_df is None:
            clustered_df = df_clustered.as_pandas_dataframe()
        if inspect_ids:
            filtered = clustered_df[clustered_df[unique_id_col].isin(inspect_ids)]
        else:
            filtered = clustered_df.iloc[0:0]
        cluster_ids = sorted(filtered["cluster_id"].dropna().unique().tolist())
    except Exception as exc: #pylint: disable=W0718
        print(f"Unable to compute cluster_ids for dashboard: {exc}")
        cluster_ids = []
        return
    if not cluster_ids:
        print("No clusters found for the selected rows; skipping dashboard.")
        return
    try:
        pairs_df = df_pairs.as_pandas_dataframe()
        mask = pd.Series(False, index=pairs_df.index)
        if left_id_col and left_id_col in pairs_df.columns:
            mask |= pairs_df[left_id_col].isin(inspect_ids)
        if right_id_col and right_id_col in pairs_df.columns:
            mask |= pairs_df[right_id_col].isin(inspect_ids)
        filtered_pairs = pairs_df.loc[mask]
        filtered_clusters = clustered_df[clustered_df[unique_id_col].isin(inspect_ids)]
        df_pairs_filtered = linker._db_api.register_table(
            filtered_pairs,
            f"__splink__df_pairs_inspect_{id(filtered_pairs)}",
        )
        df_clustered_filtered = linker._db_api.register_table(
            filtered_clusters,
            f"__splink__df_clustered_inspect_{id(filtered_clusters)}",
        )
    except Exception as exc: #pylint: disable=W0718
        print(f"Unable to filter dashboard inputs: {exc}")
        return
    linker.visualisations.cluster_studio_dashboard(
        df_pairs_filtered,
        df_clustered_filtered,
        "cluster_studio.html",
        cluster_ids=cluster_ids,
        overwrite=True,
    )
    print("Cluster studio dashboard written to cluster_studio.html")


def _run_splink_predict_pairs_with_conn(
    job: SplinkDedupeJob,
    con: duckdb.DuckDBPyConnection,
    current: Run[Any],
) -> SplinkPredictResult:
    def _duckdb_exec(sql: str) -> None:
        with_duckdb(sql_exec(SQL(sql)))._step(current)

    def _duckdb_query(sql: str) -> Array:
        return with_duckdb(sql_query(SQL(sql)))._step(current)

    def _duckdb_register(name: str, df: DataFrame) -> None:
        with_duckdb(sql_register(name, df))._step(current)

    db_api = DuckDBAPI(connection=con)
    unique_id_col = cast(str, job.settings.get("unique_id_column_name", "unique_id"))
    base_prediction_rules = list(job.settings.get("blocking_rules_to_generate_predictions", []))
    base_training_rules = (
        list(job.training_blocking_rules)
        if job.training_blocking_rules
        else base_prediction_rules
    )
    input_table = _input_table_value(job.input_table)
    pairs_out = _table_name_value(job.pairs_out)
    clusters_out = _table_name_value(job.clusters_out)
    unique_pairs_table = _table_name_value(job.unique_pairs_table)
    blocked_pairs_out = _table_name_value(job.blocked_pairs_out)
    do_not_link_table = _table_name_value(job.do_not_link_table)
    do_not_link_left_col = str(job.do_not_link_left_col)
    do_not_link_right_col = str(job.do_not_link_right_col)

    def _build_linker(
        prediction_rules: Sequence[str | BlockingRuleCreator],
        input_table: str | list[str],
        extra_columns_to_retain: Sequence[str] | None = None,
    ) -> Linker:
        settings = dict(job.settings)
        if extra_columns_to_retain:
            existing = list(settings.get("additional_columns_to_retain", []))
            merged = list(dict.fromkeys(existing + list(extra_columns_to_retain)))
            settings["additional_columns_to_retain"] = merged
        return Linker(
            input_table,
            settings
            | {
                "retain_intermediate_calculation_columns": True,
                "max_iterations": 100,
                "blocking_rules_to_generate_predictions": list(prediction_rules),
            },
            db_api=db_api,
        )

    def _print_prediction_counts(
        prediction_rules: Sequence[str | BlockingRuleCreator],
        input_table: str | list[str],
    ) -> None:
        if not prediction_rules:
            return
        try:
            tables = input_table if isinstance(input_table, list) else [input_table]
            counts_df = blocking_analysis.cumulative_comparisons_to_be_scored_from_blocking_rules_data(
                table_or_tables=tables,
                blocking_rules=prediction_rules,
                link_type=job.settings.get("link_type", "dedupe_only"),
                db_api=db_api,
                unique_id_column_name=job.settings.get("unique_id_column_name", "unique_id"),
                source_dataset_column_name=job.settings.get("source_dataset_column_name"),
            )
            total_comparisons = int(counts_df["row_count"].sum())
            print(f"Total comparisons to be scored (pre-threshold): {total_comparisons}")
        except Exception as exc: #pylint: disable=W0718
            print(f"Skipping blocking count analysis: {exc}")

    def _train_linker(
        linker: Linker,
        training_rules: Sequence[str | BlockingRuleCreator],
        input_table: str | list[str],
    ) -> None:
        linker.training.estimate_probability_two_random_records_match(
            list(job.deterministic_rules), recall=job.deterministic_recall
        )
        linker.training.estimate_u_using_random_sampling(1e8)

        if not job.train_first:
            return
        prev_params: EmParams | None = None
        for run_idx in range(job.em_max_runs):
            prev_block_params: EmParams | None = None
            current_params: EmParams | None = None
            for training_rule in training_rules:
                em_session = linker.training.estimate_parameters_using_expectation_maximisation(
                    blocking_rule=training_rule,
                    fix_probability_two_random_records_match=False
                )
                try:
                    lambda_history = em_session._lambda_history_records
                except AttributeError:
                    lambda_history = []
                if lambda_history:
                    print("EM lambda history (probability_two_random_records_match):")
                    for record in lambda_history:
                        print(
                            f"  iter {record['iteration']}: "
                            f"{record['probability_two_random_records_match']:.6f}"
                        )
                current_settings = linker.misc.save_model_to_json(out_path=None)
                current_params = _extract_em_params(current_settings)
                deltas: list[float] = []
                if prev_block_params is not None:
                    max_key: ComparisonLevelKey | None = None
                    max_delta_local = -1.0
                    for key, now in current_params.items():
                        if key not in prev_block_params:
                            continue
                        prev = prev_block_params[key]
                        match now.m, now.u, prev.m, prev.u:
                            case Just(m_now), Just(u_now), Just(m_prev), Just(u_prev):
                                if u_now != 0 and u_prev != 0:
                                    delta = abs(
                                        math.log2(m_now / u_now)
                                        - math.log2(m_prev / u_prev)
                                    )
                                    deltas.append(delta)
                                    if delta > max_delta_local:
                                        max_key = key
                                        max_delta_local = delta
                    max_delta = max(deltas) if deltas else 1.0
                    print(
                        f"EM run {run_idx + 1}/{job.em_max_runs} block drift: "
                        f"max_delta={max_delta:.6f} (block to block)"
                    )
                    if max_key is not None:
                        print(
                            f"  max_delta_key={max_key.comparison_name}::"
                            f"{max_key.level_name}"
                        )
                prev_block_params = current_params
            if current_params is None:
                break
            if prev_params is not None:
                deltas = []
                max_key = None
                max_delta_local = -1.0
                for key, now in current_params.items():
                    if key not in prev_params:
                        continue
                    prev = prev_params[key]
                    match now.m, now.u, prev.m, prev.u:
                        case Just(m_now), Just(u_now), Just(m_prev), Just(u_prev):
                            if u_now != 0 and u_prev != 0:
                                delta = abs(
                                    math.log2(m_now / u_now)
                                    - math.log2(m_prev / u_prev)
                                )
                                deltas.append(delta)
                                if delta > max_delta_local:
                                    max_key = key
                                    max_delta_local = delta
                max_delta = max(deltas) if deltas else 1.0
                print(
                    f"EM run {run_idx + 1}/{job.em_max_runs}: "
                    f"max_delta={max_delta:.6f} "
                    f"(stop if < {job.em_stop_delta:.6f} after {job.em_min_runs} runs)"
                )
                if max_key is not None:
                    print(
                        f"  max_delta_key={max_key.comparison_name}::"
                        f"{max_key.level_name}"
                    )
                if run_idx + 1 >= job.em_min_runs and max_delta < job.em_stop_delta:
                    break
            prev_params = current_params

    def _predict_to_table(linker: Linker, table_name: str) -> None:
        df_pairs = linker.inference.predict(
            threshold_match_probability=job.predict_threshold
        )
        _duckdb_exec(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM {df_pairs.physical_name}"
        )

    do_not_link_table_value = do_not_link_table if do_not_link_table != "" else None
    input_table_for_prediction = input_table
    if job.capture_blocked_edges:
        linker = _build_linker(base_prediction_rules, input_table)
        _print_prediction_counts(base_prediction_rules, input_table)
        _train_linker(linker, base_training_rules, input_table)
        capture_rules: list[str | BlockingRuleCreator] = []
        seen_rules: set[str] = set()
        for rule in base_training_rules + base_prediction_rules:
            if isinstance(rule, BlockingRuleCreator):
                capture_rules.append(rule)
                continue
            if rule in seen_rules:
                continue
            capture_rules.append(rule)
            seen_rules.add(rule)
        capture_pairs_table = f"{pairs_out}_capture"
        _with_temp_blocking_rules_on_linker(
            linker,
            capture_rules,
            lambda: _predict_to_table(linker, capture_pairs_table),
        )
        _collect_blocked_edges(
            pairs_table=capture_pairs_table,
            do_not_link_table=do_not_link_table_value or "do_not_link",
            id_left_col=do_not_link_left_col,
            id_right_col=do_not_link_right_col,
            unique_id_column_name=unique_id_col,
            query_fn=_duckdb_query,
            exec_fn=_duckdb_exec,
        )
        try:
            linker.table_management.invalidate_cache()
            _drop_all_splink_tables(_duckdb_exec, _duckdb_query)
        except Exception: #pylint: disable=W0718
            pass

        prediction_rules = base_prediction_rules
        training_rules = base_training_rules
        if do_not_link_table_value and isinstance(input_table, str):
            input_table_for_prediction = f"{input_table}_exc"
            _create_exclusion_list_table(
                input_table=input_table,
                output_table=input_table_for_prediction,
                do_not_link_table=do_not_link_table_value,
                unique_id_column_name=unique_id_col,
                id_left_col=do_not_link_left_col,
                id_right_col=do_not_link_right_col,
                exec_fn=_duckdb_exec,
            )
            exclusion_clause = (
                f"NOT (list_contains(l.exclusion_ids, r.{unique_id_col}) "
                f"OR list_contains(r.exclusion_ids, l.{unique_id_col}))"
            )
            prediction_rules = _append_clause_to_rules(prediction_rules, exclusion_clause)
            training_rules = _append_clause_to_rules(training_rules, exclusion_clause)
        extra_cols = ["exclusion_ids"] if do_not_link_table_value else None
        linker = _build_linker(prediction_rules, input_table_for_prediction, extra_cols)
        _print_prediction_counts(prediction_rules, input_table_for_prediction)
        _train_linker(linker, training_rules, input_table_for_prediction)
    else:
        prediction_rules = base_prediction_rules
        training_rules = base_training_rules
        linker = _build_linker(prediction_rules, input_table)
        _print_prediction_counts(prediction_rules, input_table)
        _train_linker(linker, training_rules, input_table)
    df_pairs = linker.inference.predict(
        threshold_match_probability=job.predict_threshold
    )

    _duckdb_exec(
        f"CREATE OR REPLACE TABLE {pairs_out} AS "
        f"SELECT * FROM {df_pairs.physical_name}"
    )
    if do_not_link_table_value:
        _filter_pairs_table_do_not_link(
            pairs_table=pairs_out,
            do_not_link_table=do_not_link_table_value,
            id_left_col=do_not_link_left_col,
            id_right_col=do_not_link_right_col,
            unique_id_column_name=unique_id_col,
            query_fn=_duckdb_query,
            exec_fn=_duckdb_exec,
        )
    if job.settings.get("link_type", SplinkLinkType.DEDUPE_ONLY.value) \
        == SplinkLinkType.LINK_ONLY.value:
        _duckdb_exec(f"""
            CREATE OR REPLACE TABLE {pairs_out}_top1 AS
            WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                PARTITION BY unique_id_r
                ORDER BY
                    match_probability DESC,
                    -- stable tie-breakers so results are deterministic:
                    COALESCE(match_weight, 0) DESC,
                    CAST(unique_id_l AS VARCHAR)
                ) AS rn
            FROM {pairs_out}
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1;
            """)

    prediction_table: TableName | TableNames
    if isinstance(input_table_for_prediction, str):
        prediction_table = TableName(input_table_for_prediction)
    else:
        prediction_table = TableNames(tuple(input_table_for_prediction))
    return SplinkPredictResult(
        linker=linker,
        input_table_for_prediction=prediction_table,
        unique_id_col=UniqueIdColumnName(unique_id_col),
        pairs_out=job.pairs_out,
        clusters_out=job.clusters_out,
        do_not_link_table=job.do_not_link_table if do_not_link_table_value else None,
        do_not_link_left_col=job.do_not_link_left_col,
        do_not_link_right_col=job.do_not_link_right_col,
        blocked_pairs_out=job.blocked_pairs_out,
        unique_pairs_table=job.unique_pairs_table,
    )

def _run_splink_unique_matching_and_cluster_with_conn(
    job: SplinkDedupeJob,
    predict: SplinkPredictResult,
    con: duckdb.DuckDBPyConnection,
    current: Run[Any],
) -> tuple[str, str]:
    _ = con

    def _duckdb_exec(sql: str) -> None:
        with_duckdb(sql_exec(SQL(sql)))._step(current)

    def _duckdb_query(sql: str) -> Array:
        return with_duckdb(sql_query(SQL(sql)))._step(current)

    def _duckdb_register(name: str, df: DataFrame) -> None:
        with_duckdb(sql_register(name, df))._step(current)

    input_table_for_prediction = (
        _table_name_value(predict.input_table_for_prediction)
        if isinstance(predict.input_table_for_prediction, TableName)
        else _table_names_value(predict.input_table_for_prediction)
    )
    unique_id_col = str(predict.unique_id_col)
    pairs_out = _table_name_value(predict.pairs_out)
    clusters_out = _table_name_value(predict.clusters_out)
    do_not_link_table_value = (
        _table_name_value(predict.do_not_link_table)
        if predict.do_not_link_table is not None
        else None
    )
    do_not_link_left_col = str(predict.do_not_link_left_col)
    do_not_link_right_col = str(predict.do_not_link_right_col)
    blocked_pairs_out = _table_name_value(predict.blocked_pairs_out)
    unique_pairs_table = _table_name_value(predict.unique_pairs_table)

    if job.unique_matching and unique_pairs_table != "":
        pairs = _duckdb_query(
            f"""
            SELECT unique_id_l, unique_id_r, match_probability 
            FROM {pairs_out} WHERE match_probability > 0
            """
        )
        G: nx.Graph = nx.Graph() #pylint: disable=C0103
        for row in pairs:
            G.add_edge(
                f"l_{row['unique_id_l']}",
                f"r_{row['unique_id_r']}",
                weight=row["match_probability"]
            )
        matching = nx.max_weight_matching(G)
        matched_pairs = []
        seen = set()
        for u, v in matching:
            if u not in seen and v not in seen:
                weight = G[u][v]['weight']
                if u.startswith("r_"):
                    u, v = v, u
                matched_pairs.append((u[2:], v[2:], weight))
                seen.add(u)
                seen.add(v)
        if matched_pairs:
            values_str = ', '.join(f"({repr(u)}, {repr(v)}, {w})" for u, v, w in matched_pairs)
            _duckdb_exec(
                f"CREATE OR REPLACE TABLE {unique_pairs_table} "
                f"AS SELECT * FROM (VALUES {values_str}) "
                "AS t(unique_id_l, unique_id_r, match_probability)"
            )
        else:
            _duckdb_exec(f"""--sql
                CREATE OR REPLACE TABLE {unique_pairs_table}
                (
                    unique_id_l VARCHAR, 
                    unique_id_r VARCHAR, 
                    match_probability DOUBLE
                )
                """)

    if job.settings.get("link_type", SplinkLinkType.DEDUPE_ONLY.value) != SplinkLinkType.LINK_ONLY.value:
        if not isinstance(input_table_for_prediction, str):
            raise RuntimeError(
                "Constrained clustering currently requires a single input table name "
                f"(got {type(input_table_for_prediction).__name__})."
            )

        nodes_rows = _duckdb_query(
            f"""
            SELECT
              CAST({unique_id_col} AS VARCHAR) AS unique_id,
              CAST(exclusion_id AS VARCHAR) AS exclusion_id
            FROM {input_table_for_prediction}
            """
        )
        nodes = [
            (row["unique_id"], row["exclusion_id"])
            for row in nodes_rows
        ]
        pair_cols = {
            row["name"]
            for row in _duckdb_query(
                f"SELECT name FROM pragma_table_info('{pairs_out}')"
            )
        }
        left_id_col, right_id_col = _resolve_pair_id_cols(
            pair_cols,
            unique_id_col,
            pairs_out,
        )
        edges_rows = _duckdb_query(
            f"""
            SELECT
              CAST({left_id_col} AS VARCHAR) AS uid_l,
              CAST({right_id_col} AS VARCHAR) AS uid_r,
              match_probability
            FROM {pairs_out}
            WHERE match_probability >= {job.cluster_threshold}
            ORDER BY
              match_probability DESC,
              CAST({left_id_col} AS VARCHAR),
              CAST({right_id_col} AS VARCHAR)
            """
        )
        edges = [
            (row["uid_l"], row["uid_r"], row["match_probability"])
            for row in edges_rows
        ]
        capture_blocked = (
            job.capture_blocked_edges
            and bool(blocked_pairs_out)
            and blocked_pairs_out != do_not_link_table_value
        )
        clusters_df, blocked_df = _constrained_greedy_clusters(
            nodes=nodes,
            edges=edges,
            unique_id_column_name=unique_id_col,
            capture_blocked=capture_blocked,
            blocked_id_cols=(do_not_link_left_col, do_not_link_right_col),
        )
        _duckdb_register("_constrained_clusters_df", clusters_df)
        _duckdb_exec(
            f"CREATE OR REPLACE TABLE {clusters_out} AS "
            "SELECT * FROM _constrained_clusters_df"
        )
        if blocked_df is not None and capture_blocked and blocked_pairs_out:
            _duckdb_register("_blocked_edges_df", blocked_df)
            _duckdb_exec(
                f"CREATE OR REPLACE TABLE {blocked_pairs_out} AS "
                "SELECT * FROM _blocked_edges_df"
            )

        counts_table = f"{clusters_out}_counts"
        existing = _duckdb_query(
            f"""
            SELECT table_type
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = '{counts_table}'
            """
        )
        if existing:
            table_type = str(existing[0]["table_type"]).upper()
            if table_type == "VIEW":
                _duckdb_exec(f"DROP VIEW IF EXISTS {counts_table}")
            else:
                _duckdb_exec(f"DROP TABLE IF EXISTS {counts_table}")
        _duckdb_exec(
            f"""
            CREATE OR REPLACE TABLE {counts_table} AS
            SELECT cluster_id, COUNT(*)::BIGINT AS member_count
            FROM {clusters_out}
            GROUP BY cluster_id
            """
        )
    return (
        unique_pairs_table if job.unique_matching and unique_pairs_table != "" else pairs_out,
        clusters_out if job.settings.get("link_type", SplinkLinkType.DEDUPE_ONLY.value) != SplinkLinkType.LINK_ONLY.value else ""
    )

def _run_splink_dedupe_with_conn(
    job: SplinkDedupeJob,
    con: duckdb.DuckDBPyConnection,
    current: Run[Any],
) -> tuple[Linker, str, str]:
    def _configure_splink_logger(name: str) -> None:
        logger = logging.getLogger(name)
        logger.setLevel(15)
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(15)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.propagate = False

    _configure_splink_logger("splink")
    _configure_splink_logger("splink.internals")

    predict = _run_splink_predict_pairs_with_conn(job, con, current)
    pairs_table, clusters_table = _run_splink_unique_matching_and_cluster_with_conn(
        job,
        predict,
        con,
        current,
    )
    return (predict.linker, pairs_table, clusters_table)


def _init_splink_dedupe_context(job: SplinkDedupeJob) -> Run[Unit]:
    unique_id_col = UniqueIdColumnName(
        cast(str, job.settings.get("unique_id_column_name", "unique_id"))
    )
    prediction_rules = Array.make(
        tuple(job.settings.get("blocking_rules_to_generate_predictions", []))
    )
    training_rules = Array.make(
        tuple(job.training_blocking_rules
            if job.training_blocking_rules
            else prediction_rules
        )
    )
    link_type = SplinkLinkType.from_settings(job.settings)
    def _build_tables() -> SplinkTableNames:
        tables = _add_all_tables(SplinkTableNames.empty(), job.pairs_out, job.input_table)
        tables = _add_dedupe_tables(
            tables,
            link_type,
            job.clusters_out,
            job.do_not_link_table,
            job.blocked_pairs_out,
            job.capture_blocked_edges,
            job.pairs_out,
        )
        tables = _add_unique_matching_tables(
            tables,
            job.unique_matching,
            job.unique_pairs_table,
        )
        tables = _add_link_type_tables(tables, link_type, job.pairs_out)
        return tables

    def _finish(tables: SplinkTableNames) -> Run[Unit]:
        ctx = SplinkContext(
            phase=SplinkPhase.INIT,
            tables=tables,
            unique_id_col=unique_id_col,
            prediction_rules=prediction_rules,
            training_rules=training_rules,
            settings=job.settings,
            predict_threshold=job.predict_threshold,
            cluster_threshold=job.cluster_threshold,
            deterministic_rules=Array.make(tuple(job.deterministic_rules)),
            deterministic_recall=job.deterministic_recall,
            train_first=job.train_first,
        visualize=job.visualize,
        unique_matching=job.unique_matching,
            em_max_runs=job.em_max_runs,
            em_min_runs=job.em_min_runs,
            em_stop_delta=job.em_stop_delta,
            capture_blocked_edges=job.capture_blocked_edges,
        )
        return put_splink_context(ctx) ^ pure(unit)

    return _validate_splink_dedupe_input_tables(
        input_tables=job.input_table,
        link_type=link_type,
        clusters_out=job.clusters_out,
        unique_matching=job.unique_matching,
        unique_pairs_table=job.unique_pairs_table,
        capture_blocked_edges=job.capture_blocked_edges,
        blocked_pairs_out=job.blocked_pairs_out,
        do_not_link_table=job.do_not_link_table,
    ) ^ _finish(_build_tables())


def _configure_splink_logger_step() -> Run[Unit]:
    def _configure(name: str) -> None:
        logger = logging.getLogger(name)
        logger.setLevel(15)
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(15)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.propagate = False

    def _run(_: Any) -> Run[Unit]:
        _configure("splink")
        _configure("splink.internals")
        return pure(unit)

    return pure(unit) >> _run


def _load_splink_duckdb_step() -> Run[Unit]:
    def _with_env(env: Environment) -> Run[Unit]:
        con = env["connections"].get(DbBackend.DUCKDB)
        if con is None:
            return throw(ErrorPayload("Splink requires a DuckDB connection."))

        def _update(ctx: SplinkContext) -> SplinkContext:
            return replace(ctx, phase=SplinkPhase.PREPARE)

        return _update_splink_context(_update)

    return ask() >> _with_env


def _splink_dedupe_predict_pairs_step(
    job: SplinkDedupeJob,
) -> Run[SplinkPredictResult]:
    def _run(ctx: SplinkContext) -> Run[SplinkPredictResult]:
        def _with_env(env: Environment) -> Run[SplinkPredictResult]:
            con = env["connections"].get(DbBackend.DUCKDB)
            if con is None:
                return throw(ErrorPayload("Splink requires a DuckDB connection."))

            def _step(self_run: Run[Any]) -> SplinkPredictResult:
                result = _run_splink_predict_pairs_with_conn(job, con, self_run)
                updated = replace(
                    ctx,
                    linker=result.linker,
                    phase=SplinkPhase.PREDICT,
                )
                put_splink_context(updated)._step(self_run)
                return result

            return Run(_step, lambda i, c: c._perform(i, c))

        return ask() >> _with_env

    return _require_splink_context() >> _run

def _splink_dedupe_unique_matching_and_cluster_step(
    job: SplinkDedupeJob,
    predict: SplinkPredictResult,
) -> Run[Unit]:
    def _run(ctx: SplinkContext) -> Run[Unit]:
        def _with_env(env: Environment) -> Run[Unit]:
            con = env["connections"].get(DbBackend.DUCKDB)
            if con is None:
                return throw(ErrorPayload("Splink requires a DuckDB connection."))

            def _step(self_run: Run[Any]) -> Unit:
                pairs_table, clusters_table = _run_splink_unique_matching_and_cluster_with_conn(
                    job,
                    predict,
                    con,
                    self_run,
                )
                tables = ctx.tables.set(ResultPairsTableName(pairs_table))
                if clusters_table:
                    tables = tables.set(ResultClustersTableName(clusters_table))
                updated = replace(
                    ctx,
                    tables=tables,
                    phase=SplinkPhase.PERSIST,
                )
                put_splink_context(updated)._step(self_run)
                return unit

            return Run(_step, lambda i, c: c._perform(i, c))

        return ask() >> _with_env

    return _require_splink_context() >> _run


def _splink_dedupe_finalize() -> Run[tuple[Linker, str, str]]:
    def _finish(ctx: SplinkContext) -> Run[tuple[Linker, str, str]]:
        linker = ctx.linker
        if linker is None:
            return throw(ErrorPayload("Splink linker was not created."))
        return _tables_get_required(ctx.tables, ResultPairsTableName) >> (
            lambda pairs_table:
            _tables_get_optional(ctx.tables, ResultClustersTableName)
            >> (lambda clusters_table: pure((pairs_table, clusters_table)))
        ) >> (
            lambda tables:
            put_splink_context(replace(ctx, phase=SplinkPhase.DONE)) >> (
                lambda _: pure(
                    (
                        linker,
                        str(tables[0]),
                        str(from_maybe(ResultClustersTableName(""), tables[1])),
                    )
                )
            )
        )
    return _require_splink_context() >> _finish


def run_splink_dedupe_monadic(job: SplinkDedupeJob) -> Run[tuple[Linker, str, str]]:
    """
    New monadic handler for SplinkDedupeJob (not wired into perform yet).
    """
    chain = (
        _init_splink_dedupe_context(job)
        >> (lambda _: _configure_splink_logger_step())
        >> (lambda _: _load_splink_duckdb_step())
        >> (lambda _: _splink_dedupe_predict_pairs_step(job))
        >> (lambda predict: _splink_dedupe_unique_matching_and_cluster_step(job, predict))
        >> (lambda _: _splink_dedupe_finalize())
    )
    return with_duckdb(chain)

def _splink_model_paths(splink_key: Any) -> tuple[Path, Path]:
    key_str = str(splink_key).replace("/", "_")
    base_dir = Path("splink_models")
    model_path = base_dir / f"splink_model_{key_str}.json"
    meta_path = base_dir / f"splink_model_{key_str}.meta.json"
    return model_path, meta_path


def _save_splink_model(
    splink_key: Any,
    linker: Linker,
    input_table: PredictionInputTableName | PredictionInputTableNames,
) -> None:
    if splink_key is None:
        return
    model_path, meta_path = _splink_model_paths(splink_key)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    linker.misc.save_model_to_json(str(model_path), overwrite=True)
    actual_input_table: str | list[str] = _input_table_value(input_table)
    try:
        input_tables = list(linker._input_tables_dict.values())
        if len(input_tables) == 1:
            actual_input_table = input_tables[0].physical_name
        elif len(input_tables) > 1:
            actual_input_table = [df.physical_name for df in input_tables]
    except Exception: #pylint: disable=W0718
        actual_input_table = _input_table_value(input_table)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"input_table": actual_input_table}, f, indent=2)


def _load_splink_model(
    splink_key: Any,
    con: duckdb.DuckDBPyConnection,
) -> Linker | None:
    if splink_key is None:
        return None
    model_path, meta_path = _splink_model_paths(splink_key)
    if not model_path.is_file() or not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        input_table = meta.get("input_table")
        if input_table is None:
            return None
        db_api = DuckDBAPI(connection=con)
        return Linker(input_table, str(model_path), db_api=db_api)
    except Exception: #pylint: disable=W0718
        return None


def run_splink(prog: Run[A]) -> Run[A]:
    """
    Eliminator for Splink intents.
    """
    def step(self_run: Run[Any]) -> A:
        parent = self_run._perform

        def perform(intent: Any, current: Run[Any]) -> Any:
            match intent:
                case SplinkDedupeJob():
                    env = cast(Environment, ask()._step(current))
                    con = env["connections"].get(DbBackend.DUCKDB)
                    if con is None:
                        return throw(
                            ErrorPayload("Splink requires a DuckDB connection.")
                        )._step(current)
                    use_monadic = bool(
                        env.get("extras", {}).get(EnvKey("splink_use_monadic"), False)
                    )
                    if use_monadic:
                        out = run_splink_dedupe_monadic(intent)._step(current)
                    else:
                        out = _run_splink_dedupe_with_conn(intent, con, current)
                    put_splink_linker(intent.splink_key, out[0])._step(current)
                    _save_splink_model(intent.splink_key, out[0], intent.input_table)
                    return out
                case SplinkVisualizeJob():
                    linker = get_splink_linker(intent.splink_key)._step(current)
                    if linker is None:
                        return throw(
                            ErrorPayload("No Splink linker stored for visualization.")
                        )._step(current)
                    return _run_splink_visualize(linker, intent)
                case HasSplinkLinker(key):
                    linker = get_splink_linker(key)._step(current)
                    if linker is not None:
                        return True
                    env = cast(Environment, ask()._step(current))
                    con = env["connections"].get(DbBackend.DUCKDB)
                    if con is None:
                        return False
                    reloaded = _load_splink_model(key, con)
                    if reloaded is None:
                        return False
                    put_splink_linker(key, reloaded)._step(current)
                    return True
                case _:
                    return parent(intent, current)

        inner = Run(prog._step, perform)
        return inner._step(inner)

    return Run(step, lambda i, c: c._perform(i, c))
