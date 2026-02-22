"""Splink runtime clustering and pair post-processing helpers."""
from __future__ import annotations

from typing import Any

import networkx as nx
from pandas import DataFrame

from ..array import Array
from ..hashset import HashSet
from ..maybe import Just, nothing
from ..monad import Unit, unit
from ..run import ErrorPayload, Run, pure, throw
from ..runsql import SQL, sql_exec, sql_query, sql_register
from ..traverse import array_traverse_run
from ..tuple import Threeple, Tuple
from .splink_context import (
    context_replace,
    tables_get_optional,
    with_splink_context,
    with_splink_context_linker,
)
from .splink_types import (
    BlockedEdgesRows,
    BlockedPairsTableName,
    ClusterNodeId,
    ClusterEdgeLeftId,
    ClusterEdgeRightId,
    ClusterEdgeWeight,
    ClusterPairsTableName,
    ClusterResult,
    ClusteredRows,
    ClustersCountsTableName,
    ClustersTableName,
    ColumnName,
    DoNotLinkTableName,
    ExclusionId,
    PairLeftIdColumnName,
    PairRightIdColumnName,
    PairsTableName,
    ResultClustersTableName,
    ResultPairsTableName,
    SplinkContext,
    SplinkPairsSchemaError,
    SplinkPhase,
    SplinkLinkType,
    UniqueIdColumnName,
    UniquePairsTableName,
)


def _set_cluster_pairs_table_from_pairs(ctx: SplinkContext) -> Run[Unit]:
    pairs_table = ctx.tables.get_required(PairsTableName)
    return context_replace(cluster_pairs_table=ClusterPairsTableName(str(pairs_table)))


def _resolve_pair_id_cols_typed(
    pair_cols: HashSet[ColumnName],
    unique_id_column_name: UniqueIdColumnName,
    pairs_table: PairsTableName,
) -> Tuple[PairLeftIdColumnName, PairRightIdColumnName]:
    left_id_col = PairLeftIdColumnName(f"{unique_id_column_name}_l")
    right_id_col = PairRightIdColumnName(f"{unique_id_column_name}_r")
    if left_id_col in pair_cols and right_id_col in pair_cols:
        return Tuple(left_id_col, right_id_col)
    if ColumnName("unique_id_l") in pair_cols and ColumnName("unique_id_r") in pair_cols:
        return Tuple(PairLeftIdColumnName("unique_id_l"), PairRightIdColumnName("unique_id_r"))
    raise SplinkPairsSchemaError(
        f"Constrained clustering cannot find id columns in {pairs_table}. "
        f"Expected ({left_id_col}, {right_id_col}) or (unique_id_l, unique_id_r)."
    )


def resolve_pair_id_cols_from_table_step(
    pairs_table: PairsTableName,
    unique_id_column_name: UniqueIdColumnName,
) -> Run[Tuple[HashSet[ColumnName], Tuple[PairLeftIdColumnName, PairRightIdColumnName]]]:
    """Resolve pair id column names from table schema and return typed columns."""
    def _with_cols(rows: Array) -> Run[Tuple[HashSet[ColumnName], Tuple[PairLeftIdColumnName, PairRightIdColumnName]]]:
        cols_array = (lambda row: ColumnName(str(row["name"]))) & rows
        pair_cols = HashSet.fromArray(cols_array)
        id_cols = _resolve_pair_id_cols_typed(pair_cols, unique_id_column_name, pairs_table)
        return pure(Tuple(pair_cols, id_cols))

    return sql_query(SQL(f"SELECT name FROM pragma_table_info('{pairs_table}')")) >> _with_cols


def _set_pair_id_cols(ctx: SplinkContext) -> Run[Unit]:
    pairs_table = ctx.cluster_pairs_table
    if not pairs_table.is_present():
        return throw(ErrorPayload("Cluster pairs table is not initialized."))
    return resolve_pair_id_cols_from_table_step(
        pairs_table=PairsTableName(str(pairs_table)),
        unique_id_column_name=ctx.unique_id_col,
    ) >> (lambda resolved: context_replace(pair_id_cols=Just(Tuple.make(resolved.snd.fst, resolved.snd.snd))))


def _set_cluster_nodes(ctx: SplinkContext) -> Run[Unit]:
    input_tables = ctx.tables.get_required(PredictionInputTableNames)
    input_table = input_tables.left().value_or("")

    def _with_rows(rows: Array) -> Run[Unit]:
        nodes = (lambda row: Tuple.make(
            ClusterNodeId(str(row["unique_id"])),
            ExclusionId(str(row["exclusion_id"])),
        )) & rows
        return context_replace(cluster_nodes=nodes)

    return sql_query(SQL(f"""
        SELECT CAST({ctx.unique_id_col} AS VARCHAR) AS unique_id,
               CAST(exclusion_id AS VARCHAR) AS exclusion_id
        FROM {input_table}
    """)) >> _with_rows


def _set_cluster_edges(ctx: SplinkContext) -> Run[Unit]:
    pairs_table = ctx.cluster_pairs_table
    if not pairs_table.is_present():
        return throw(ErrorPayload("Cluster pairs table is not initialized."))
    match ctx.pair_id_cols:
        case Just(cols):
            left_id_col = cols.fst
            right_id_col = cols.snd
        case _:
            return throw(ErrorPayload("Pair id columns are not initialized."))

    def _with_rows(rows: Array) -> Run[Unit]:
        edges = (lambda row: Threeple.make(
            ClusterEdgeLeftId(str(row["uid_l"])),
            ClusterEdgeRightId(str(row["uid_r"])),
            ClusterEdgeWeight(float(row["match_probability"])),
        )) & rows
        return context_replace(cluster_edges=edges)

    return sql_query(SQL(f"""
        SELECT CAST({left_id_col} AS VARCHAR) AS uid_l,
               CAST({right_id_col} AS VARCHAR) AS uid_r,
               match_probability
        FROM {pairs_table}
        WHERE match_probability >= {ctx.cluster_threshold}
        ORDER BY match_probability DESC,
                 CAST({left_id_col} AS VARCHAR),
                 CAST({right_id_col} AS VARCHAR)
    """)) >> _with_rows


def _constrained_greedy_clusters(
    *,
    nodes: list[tuple[str, str]],
    edges: list[tuple[str, str, float]],
    unique_id_column_name: str,
    capture_blocked: bool = False,
    blocked_id_cols: tuple[str, str] = ("id_l", "id_r"),
) -> tuple[DataFrame, DataFrame | None]:
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
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_status(a: str, b: str) -> tuple[bool, set[str], bool]:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False, set(), True
        shared = exclusions_in_component[ra].intersection(exclusions_in_component[rb])
        return len(shared) == 0, shared, False

    def can_union(a: str, b: str) -> bool:
        ok, _, _ = union_status(a, b)
        return ok

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
                blocked_rows.append({
                    id_left_col: uid_l,
                    id_right_col: uid_r,
                    "match_probability": prob,
                    "shared_exclusion_ids": ",".join(sorted(shared)),
                })

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


def _set_cluster_result(ctx: SplinkContext) -> Run[Unit]:
    def _to_nodes() -> list[tuple[str, str]]:
        return [(str(n.fst), str(n.snd)) for n in ctx.cluster_nodes]

    def _to_edges() -> list[tuple[str, str, float]]:
        return [(str(e.fst), str(e.snd), float(e.trd)) for e in ctx.cluster_edges]

    do_not_link_table = tables_get_optional(ctx.tables, DoNotLinkTableName)
    capture_blocked = ctx.capture_blocked_edges and do_not_link_table.is_present()
    clusters_df, blocked_df = _constrained_greedy_clusters(
        nodes=_to_nodes(),
        edges=_to_edges(),
        unique_id_column_name=str(ctx.unique_id_col),
        capture_blocked=capture_blocked,
        blocked_id_cols=(str(ctx.do_not_link_left_col), str(ctx.do_not_link_right_col)),
    )
    blocked = nothing() if blocked_df is None else Just(BlockedEdgesRows(blocked_df))
    return context_replace(cluster_result=Just(ClusterResult(ClusteredRows(clusters_df), blocked)))


def _result_pairs_table_from_ctx(ctx: SplinkContext) -> ResultPairsTableName:
    pairs_out = ctx.tables.get_required(PairsTableName)
    if ctx.unique_matching:
        unique_pairs_table = ctx.tables.get_required(UniquePairsTableName)
        return ResultPairsTableName(str(unique_pairs_table))
    return ResultPairsTableName(str(pairs_out))


def _persist_diagnostic_blocked_edges(ctx: SplinkContext) -> Run[Unit]:
    do_not_link_table = ctx.tables.get_required(DoNotLinkTableName)
    match ctx.cluster_result:
        case Just(result):
            match result.blocked:
                case Just(blocked_rows):
                    return (
                        sql_register("_blocked_edges_df", blocked_rows.df)
                        ^ sql_exec(SQL(f"CREATE OR REPLACE TABLE {do_not_link_table} AS SELECT * FROM _blocked_edges_df"))
                    )
                case _:
                    return pure(unit)
        case _:
            return throw(ErrorPayload("Cluster result is not initialized."))


def _persist_final_clusters(ctx: SplinkContext) -> Run[Unit]:
    clusters_out = ctx.tables.get_required(ClustersTableName)
    blocked_pairs_out = tables_get_optional(ctx.tables, BlockedPairsTableName)
    match ctx.cluster_result:
        case Just(result):
            def _persist_blocked() -> Run[Unit]:
                match result.blocked:
                    case Just(blocked_rows) if blocked_pairs_out.is_present():
                        return (
                            sql_register("_blocked_edges_df", blocked_rows.df)
                            ^ sql_exec(SQL(f"CREATE OR REPLACE TABLE {blocked_pairs_out} AS SELECT * FROM _blocked_edges_df"))
                        )
                    case _:
                        return pure(unit)

            return (
                sql_register("_constrained_clusters_df", result.clusters.df)
                ^ sql_exec(SQL(f"CREATE OR REPLACE TABLE {clusters_out} AS SELECT * FROM _constrained_clusters_df"))
                ^ _persist_blocked()
                ^ sql_exec(SQL(f"""
                    CREATE OR REPLACE TABLE {ClustersCountsTableName.from_clusters(clusters_out)} AS
                    SELECT cluster_id, COUNT(*)::BIGINT AS member_count
                    FROM {clusters_out}
                    GROUP BY cluster_id
                """))
                ^ context_replace(
                    tables=ctx.tables.set(_result_pairs_table_from_ctx(ctx)).set(ResultClustersTableName(str(clusters_out))),
                    phase=SplinkPhase.PERSIST,
                )
            )
        case _:
            return throw(ErrorPayload("Cluster result is not initialized."))


def _persist_link_only_results(ctx: SplinkContext) -> Run[Unit]:
    return context_replace(
        tables=ctx.tables.set(_result_pairs_table_from_ctx(ctx)),
        phase=SplinkPhase.PERSIST,
    )


def drop_all_splink_tables(exec_fn, query_fn) -> None:
    """Drop all transient Splink-managed tables/views from the active database."""
    rows = query_fn(
        "SELECT table_name, table_type "
        "FROM information_schema.tables "
        "WHERE table_name LIKE '__splink__%'"
    )
    for row in rows:
        name = row["table_name"]
        table_type = str(row["table_type"]).upper()
        stmt = f"DROP VIEW IF EXISTS {name}" if table_type == "VIEW" else f"DROP TABLE IF EXISTS {name}"
        exec_fn(stmt)


def _drop_all_splink_tables_step() -> Run[Unit]:
    def _drop_one(row: Any) -> Run[Unit]:
        name = row["table_name"]
        table_type = str(row["table_type"]).upper()
        stmt = f"DROP VIEW IF EXISTS {name}" if table_type == "VIEW" else f"DROP TABLE IF EXISTS {name}"
        return sql_exec(SQL(stmt))

    return (
        sql_query(SQL(
            "SELECT table_name, table_type "
            "FROM information_schema.tables "
            "WHERE table_name LIKE '__splink__%'"
        ))
        >> (lambda rows: pure(unit) if rows.length == 0 else array_traverse_run(rows, _drop_one).map(lambda _: unit))
    )


def _drop_all_splink_tables_detach_sqldb_step() -> Run[Unit]:
    """
    Drop Splink temp tables while temporarily detaching sqldb.

    DuckDB metadata queries (information_schema/duckdb_tables) can stall when
    a SQLite attachment is present. We only detach around the metadata scan.
    """
    def _escape_sql_literal(value: str) -> str:
        return value.replace("'", "''")

    def _reattach_sqldb(file_path: str) -> Run[Unit]:
        escaped = _escape_sql_literal(file_path)
        return (
            sql_exec(SQL("LOAD sqlite_scanner"))
            ^ sql_exec(SQL(f"ATTACH '{escaped}' AS sqldb (TYPE SQLITE)"))
        )

    def _with_database_list(rows: Array) -> Run[Unit]:
        sqldb_file = None
        for row in rows:
            if str(row.get("name", "")) == "sqldb":
                sqldb_file = str(row.get("file", ""))
                break
        if not sqldb_file:
            return _drop_all_splink_tables_step()
        return (
            sql_exec(SQL("DETACH sqldb"))
            ^ _drop_all_splink_tables_step()
            ^ _reattach_sqldb(sqldb_file)
        )

    return sql_query(SQL("PRAGMA database_list")) >> _with_database_list


def _invalidate_and_drop_splink_tables(ctx: SplinkContext, linker: Any) -> Run[Unit]:
    _ = ctx
    try:
        linker.table_management.invalidate_cache()
        return _drop_all_splink_tables_detach_sqldb_step()
    except Exception:  # pylint: disable=W0718
        return pure(unit)


def diagnostic_cluster_blocked_edges_run(ctx: SplinkContext) -> Run[Unit]:
    """Run constrained clustering to materialize diagnostic blocked-edge records."""
    _ = ctx
    return (
        with_splink_context(_set_pair_id_cols)
        ^ with_splink_context(_set_cluster_nodes)
        ^ with_splink_context(_set_cluster_edges)
        ^ with_splink_context(_set_cluster_result)
        ^ with_splink_context(_persist_diagnostic_blocked_edges)
        ^ with_splink_context_linker(_invalidate_and_drop_splink_tables)
    )


def _run_unique_matching_from_ctx(ctx: SplinkContext) -> Run[Unit]:
    if not ctx.unique_matching:
        return pure(unit)
    pairs_out = ctx.tables.get_required(PairsTableName)
    unique_pairs_table = ctx.tables.get_required(UniquePairsTableName)

    def _with_pairs(rows: Array) -> Run[Unit]:
        graph: nx.Graph = nx.Graph()
        for row in rows:
            graph.add_edge(
                f"l_{row['unique_id_l']}",
                f"r_{row['unique_id_r']}",
                weight=row["match_probability"],
            )
        matching = nx.max_weight_matching(graph)
        matched_pairs: list[tuple[str, str, float]] = []
        seen: set[str] = set()
        for u, v in matching:
            if u not in seen and v not in seen:
                weight = graph[u][v]["weight"]
                if u.startswith("r_"):
                    u, v = v, u
                matched_pairs.append((u[2:], v[2:], weight))
                seen.add(u)
                seen.add(v)

        if matched_pairs:
            values_str = ", ".join(f"({repr(u)}, {repr(v)}, {w})" for u, v, w in matched_pairs)
            return sql_exec(SQL(
                f"CREATE OR REPLACE TABLE {unique_pairs_table} "
                f"AS SELECT * FROM (VALUES {values_str}) "
                "AS t(unique_id_l, unique_id_r, match_probability)"
            ))
        return sql_exec(SQL(f"""
            CREATE OR REPLACE TABLE {unique_pairs_table}
            (
                unique_id_l VARCHAR,
                unique_id_r VARCHAR,
                match_probability DOUBLE
            )
        """))

    return sql_query(SQL(f"""
        SELECT unique_id_l, unique_id_r, match_probability
        FROM {pairs_out} WHERE match_probability > 0
    """)) >> _with_pairs


def run_unique_matching_and_cluster_from_ctx(ctx: SplinkContext) -> Run[Unit]:
    """Execute unique matching and clustering/persistence for the current context."""
    link_type = SplinkLinkType.from_settings(ctx.settings)
    if link_type == SplinkLinkType.LINK_ONLY:
        return with_splink_context(_run_unique_matching_from_ctx) ^ with_splink_context(_persist_link_only_results)
    return (
        with_splink_context(_run_unique_matching_from_ctx)
        ^ with_splink_context(_set_cluster_pairs_table_from_pairs)
        ^ with_splink_context(_set_pair_id_cols)
        ^ with_splink_context(_set_cluster_nodes)
        ^ with_splink_context(_set_cluster_edges)
        ^ with_splink_context(_set_cluster_result)
        ^ with_splink_context(_persist_final_clusters)
    )


from .splink_types import (
    PredictionInputTableNames,
)  # placed late to avoid reorder noise
