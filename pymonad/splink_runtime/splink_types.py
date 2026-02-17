"""Core Splink runtime types and shared value objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Sequence, TypeVar, cast

from pandas import DataFrame
from splink import Linker, DuckDBAPI
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.comparison import Comparison as SplinkComparison
from splink.internals.comparison_creator import ComparisonCreator

from ..array import Array
from ..hashmap import HashMap
from ..maybe import Just, Nothing, Maybe, from_maybe, nothing
from ..string import String
from ..tuple import Tuple, Threeple

A = TypeVar("A")
type StringBlockingRule = StrEnum
type StringBlockingRules = Array[StringBlockingRule]
type BlockingRuleCreators = Array[BlockingRuleCreator]


class CustomStringBlockingRule(String):
    """Dynamically constructed blocking rule (string wrapper)."""


type CustomStringBlockingRules = Array[CustomStringBlockingRule]


class SplinkChartType(str, Enum):
    """Visualization types for Splink."""
    MODEL = "model"
    PARAMETER_ESTIMATE_COMPARISONS = "parameter_estimate_comparisons"
    WATERFALL = "waterfall"
    COMPARISON = "comparison"
    CLUSTER = "cluster"
    UNLINKABLES = "unlinkables"


@dataclass(frozen=True)
class SplinkVisualizeJob:
    """Splink visualization intent."""
    splink_key: Any
    chart_type: SplinkChartType
    left_midpoints: Sequence[int] | None = None
    right_midpoints: Sequence[int] | None = None


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
        """Parse link type from settings, defaulting to dedupe mode."""
        raw = str(settings.get("link_type", cls.DEDUPE_ONLY.value))
        return cls.LINK_ONLY if raw == cls.LINK_ONLY.value else cls.DEDUPE_ONLY


def comparison_level_key(
    comparison: ComparisonCreator | SplinkComparison,
    level_label: str,
    *,
    sql_dialect: str = "duckdb",
) -> ComparisonLevelKey:
    """Build a stable comparison-level key from a comparison and level label."""
    if isinstance(comparison, ComparisonCreator):
        comparison = comparison.get_comparison(sql_dialect)
    comparison_name = f"{comparison.output_column_name}_{comparison.comparison_description}"
    return ComparisonLevelKey(comparison_name=comparison_name, level_name=level_label)


def comparison_level_keys(
    comparison: ComparisonCreator | SplinkComparison,
    level_labels: Sequence[str],
    *,
    sql_dialect: str = "duckdb",
) -> Array[ComparisonLevelKey]:
    """Build comparison-level keys for each provided level label."""
    return Array.make(
        tuple(
            comparison_level_key(
                comparison,
                level_label,
                sql_dialect=sql_dialect,
            )
            for level_label in level_labels
        )
    )


type BlockingRuleLike = (
    StringBlockingRule | CustomStringBlockingRule | BlockingRuleCreator
)
BlockingRuleLikes = (
    StringBlockingRules
    | CustomStringBlockingRules
    | BlockingRuleCreators
    | Array[BlockingRuleLike]
)

type TrainingBlockToComparisonLevelMap = HashMap[BlockingRuleLike, Array[ComparisonLevelKey]]


def _concat_blocking_rule_likes(
    left: BlockingRuleLikes,
    right: BlockingRuleLikes,
) -> Array[BlockingRuleLike]:
    """Concatenate two blocking-rule arrays preserving order."""
    return Array.make(left.a + right.a)


@dataclass(frozen=True)
class TableRef:
    """Base type for Splink table references."""


def _normalize_table_name(name: str | None) -> Maybe[str]:
    """Normalize nullable table name into Maybe[str]."""
    if name is None:
        return Nothing
    value = str(name)
    if value == "":
        return Nothing
    return Just(value)


@dataclass(frozen=True, init=False)
class TableName(TableRef):
    """Single table name."""
    name: Maybe[str]

    def __init__(self, name: str | None = "") -> None:
        """Initialize a table name wrapper from an optional string."""
        object.__setattr__(self, "name", _normalize_table_name(name))

    def is_present(self) -> bool:
        """Return True when this table name is non-empty."""
        return isinstance(self.name, Just)

    def value_or(self, default: str = "") -> str:
        """Return contained name string or provided default."""
        return from_maybe(default, self.name)

    def __str__(self) -> str:
        """Render the normalized table name as a string."""
        return self.value_or("")


class PredictionInputTableName(TableName):
    """Input table name used for prediction."""


def _to_prediction_input_name(
    value: PredictionInputTableName | str | None,
) -> PredictionInputTableName:
    """Normalize an input table value to PredictionInputTableName."""
    if value is None:
        return PredictionInputTableName("")
    if isinstance(value, PredictionInputTableName):
        return value
    return PredictionInputTableName(value)


@dataclass(frozen=True, init=False)
class PredictionInputTableNames(TableRef):
    """Input table names used for prediction."""
    tables: Tuple[PredictionInputTableName, PredictionInputTableName]

    def __init__(
        self,
        left: PredictionInputTableName | Sequence[str | PredictionInputTableName] | str,
        right: PredictionInputTableName | str | None = None,
    ) -> None:
        """Initialize left/right prediction input names from scalar or sequence input."""
        match left:
            case PredictionInputTableName() | str():
                left_entry = _to_prediction_input_name(left)
                right_entry = _to_prediction_input_name(right)
            case [*_]:
                match right:
                    case None:
                        entries = list(left)
                        left_entry = (
                            _to_prediction_input_name(entries[0])
                            if entries else PredictionInputTableName("")
                        )
                        right_entry = (
                            _to_prediction_input_name(entries[1])
                            if len(entries) > 1 else _to_prediction_input_name(None)
                        )
                    case _:
                        raise TypeError(
                            "PredictionInputTableNames expects a sequence alone "
                        )
            case _:
                raise TypeError(
                    "PredictionInputTableNames expects either a sequence alone "
                    "or explicit left/right values."
                )

        object.__setattr__(self, "tables", Tuple(left_entry, right_entry))

    @classmethod
    def from_input(
        cls,
        input_tables: PredictionInputTableName | PredictionInputTableNames | Sequence[str | PredictionInputTableName],
    ) -> PredictionInputTableNames:
        """Normalize supported input representations into PredictionInputTableNames."""
        if isinstance(input_tables, PredictionInputTableNames):
            return input_tables
        return cls(input_tables)

    def left(self) -> PredictionInputTableName:
        """Return the left prediction input table name."""
        return self.tables.fst

    def right(self) -> PredictionInputTableName:
        """Return the right prediction input table name."""
        return self.tables.snd

    def has_right(self) -> bool:
        """Return True when a right input table is present."""
        return self.tables.snd.is_present()

    def __str__(self) -> str:
        """Render prediction input names as one or two comma-separated names."""
        right = self.tables.snd
        if right.is_present():
            return f"{self.tables.fst}, {right}"
        return str(self.tables.fst)


class PairsTableName(TableName):
    """Pairs output table name."""


class PairsSourceTableName(TableName):
    """Physical pairs source table name produced by Splink."""


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
        """Create counts table name derived from a clusters table name."""
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


class ExclusionInputTableName(TableName):
    """Input table name used after exclusion enrichment."""


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


class RetainColumnName(ColumnName):
    """Column name retained for prediction output."""


type RetainColumnNames = Array[RetainColumnName]


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


class ClusterPairsTableName(TableName):
    """Active pairs table for clustering."""


class ClusterNodeId(SplinkId):
    """Node id value for clustering."""


class ClusterEdgeLeftId(ClusterNodeId):
    """Left node id for a clustering edge."""


class ClusterEdgeRightId(ClusterNodeId):
    """Right node id for a clustering edge."""


class ClusterEdgeWeight(float):
    """Match probability weight for a clustering edge."""


type ClusterNode = Tuple[ClusterNodeId, ExclusionId]
type ClusterNodes = Array[ClusterNode]
type ClusterEdge = Threeple[ClusterEdgeLeftId, ClusterEdgeRightId, ClusterEdgeWeight]
type ClusterEdges = Array[ClusterEdge]
type PairIdCols = Tuple[PairLeftIdColumnName, PairRightIdColumnName]


@dataclass(frozen=True)
class ClusteredRows:
    """Clusters dataframe wrapper."""
    df: DataFrame


@dataclass(frozen=True)
class BlockedEdgesRows:
    """Blocked edges dataframe wrapper."""
    df: DataFrame


@dataclass(frozen=True)
class ClusterResult:
    """Outputs from constrained clustering."""
    clusters: ClusteredRows
    blocked: Maybe[BlockedEdgesRows]


TableNameType = type[TableRef]
TTableName = TypeVar("TTableName", bound=TableRef)


@dataclass(frozen=True)
class SplinkTableNames:
    """Typed registry for Splink table names."""
    tables: HashMap[TableNameType, TableRef]

    @classmethod
    def empty(cls) -> SplinkTableNames:
        """Create an empty typed table-name registry."""
        return cls(HashMap.empty())

    def set(self, value: TableRef) -> SplinkTableNames:
        """Return a new registry with a typed table reference inserted."""
        return SplinkTableNames(self.tables.set(type(value), value))

    def get(self, key: type[TTableName]) -> TTableName | None:
        """Return an optional typed table reference from the registry."""
        value = self.tables.get(key)
        if value is None:
            return None
        if not isinstance(value, key):
            return None
        if isinstance(value, TableName) and not value.is_present():
            return None
        return cast(TTableName, value)

    def get_required(self, key: type[TTableName]) -> TTableName:
        """Return a required typed table reference or raise KeyError."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Missing table name for {key.__name__}.")
        return value


@dataclass(frozen=True)
class SplinkContext:
    """Monadic Splink workflow context (dedupe)."""
    phase: SplinkPhase = SplinkPhase.INIT
    tables: SplinkTableNames = field(default_factory=SplinkTableNames.empty)
    unique_id_col: UniqueIdColumnName = UniqueIdColumnName("unique_id")
    prediction_rules: BlockingRuleLikes = field(default_factory=Array.empty)
    training_rules: BlockingRuleLikes = field(default_factory=Array.empty)
    training_block_level_map: TrainingBlockToComparisonLevelMap = field(
        default_factory=HashMap.empty
    )
    linker: Maybe[Linker] = field(default_factory=nothing)
    db_api: Maybe[DuckDBAPI] = field(default_factory=nothing)
    predict_plan: Maybe[PredictPlan] = field(default_factory=nothing)
    settings: dict[str, Any] = field(default_factory=dict)
    predict_threshold: float = 0.05
    cluster_threshold: float = 0.0
    cluster_pairs_table: ClusterPairsTableName = field(default_factory=ClusterPairsTableName)
    pair_id_cols: Maybe[PairIdCols] = field(default_factory=nothing)
    cluster_nodes: ClusterNodes = field(default_factory=Array.empty)
    cluster_edges: ClusterEdges = field(default_factory=Array.empty)
    cluster_result: Maybe[ClusterResult] = field(default_factory=nothing)
    deterministic_rules: StringBlockingRules = field(default_factory=Array.empty)
    deterministic_recall: float = 0.5
    train_first: bool = False
    skip_u_estimation: bool = False
    visualize: bool = False
    unique_matching: bool = False
    em_max_runs: int = 3
    em_min_runs: int = 1
    em_stop_delta: float = 0.002
    capture_blocked_edges: bool = False
    do_not_link_left_col: BlockedIdLeftColumnName = BlockedIdLeftColumnName("id_l")
    do_not_link_right_col: BlockedIdRightColumnName = BlockedIdRightColumnName("id_r")


@dataclass(frozen=True)
class SplinkClusterInputs:
    """Derived inputs for clustering."""
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
    """Outputs from prediction/pairs step needed for downstream steps."""
    linker: Linker
    input_table_for_prediction: PredictionInputTableNames
    unique_id_col: UniqueIdColumnName
    pairs_out: PairsTableName
    clusters_out: ClustersTableName
    do_not_link_table: DoNotLinkTableName | None
    do_not_link_left_col: BlockedIdLeftColumnName
    do_not_link_right_col: BlockedIdRightColumnName
    blocked_pairs_out: BlockedPairsTableName
    unique_pairs_table: UniquePairsTableName


@dataclass(frozen=True)
class PredictPlan:
    """Derived inputs for the predict/pairs step."""
    prediction_rules: BlockingRuleLikes
    training_rules: BlockingRuleLikes
    input_table_for_prediction: PredictionInputTableNames
    extra_columns_to_retain: RetainColumnNames


@dataclass(frozen=True)
class SplinkDedupeJob:
    """Splink deduplication intent."""
    input_table: PredictionInputTableNames
    settings: dict
    predict_threshold: float
    cluster_threshold: float
    pairs_out: PairsTableName
    deterministic_rules: StringBlockingRules
    deterministic_recall: float
    clusters_out: ClustersTableName = ClustersTableName("")
    train_first: bool = False
    skip_u_estimation: bool = False
    training_blocking_rules: StringBlockingRules | BlockingRuleCreators | None = None
    training_block_level_map: TrainingBlockToComparisonLevelMap = field(
        default_factory=HashMap.empty
    )
    visualize: bool = False
    unique_matching: bool = False
    unique_pairs_table: UniquePairsTableName = UniquePairsTableName("")
    em_max_runs: int = 3
    em_min_runs: int = 1
    em_stop_delta: float = 0.002
    splink_key: Any = None
    do_not_link_table: DoNotLinkTableName = DoNotLinkTableName("")
    do_not_link_left_col: BlockedIdLeftColumnName = BlockedIdLeftColumnName("id_l")
    do_not_link_right_col: BlockedIdRightColumnName = BlockedIdRightColumnName("id_r")
    blocked_pairs_out: BlockedPairsTableName = BlockedPairsTableName("")
    capture_blocked_edges: bool = True
