"""
Compare cluster tables and return a DuckDB table of unmatched clusters.

Unmatched clusters are defined by member set equality using member_id_col.
Cluster IDs are not compared; signatures are computed per cluster.
"""

from dataclasses import dataclass

from pymonad import Run, SQL, sql_exec, sql_query, pure, put_line


@dataclass(frozen=True)
class ClusterCompareRequest:
    """Inputs required to compare two cluster tables."""

    left_table: str
    right_table: str
    member_id_col: str
    cluster_id_col: str = "cluster_id"
    output_table: str = "cluster_compare_unmatched"


@dataclass(frozen=True)
class ClusterCompareResult:
    """Result metadata for cluster comparison."""

    output_table: str
    left_table: str
    right_table: str
    member_id_col: str
    left_unmatched_clusters: int
    right_unmatched_clusters: int


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _qtable(name: str) -> str:
    return ".".join(_qident(part) for part in name.split("."))


def _table_columns(table_name: str) -> Run[list[str]]:
    return sql_query(SQL(f"PRAGMA table_info('{table_name}');")) >> (
        lambda rows: pure(
            [r["name"] for r in rows if r["name"] != "_band_helper"]
        )
    )


def _order_by_columns(cols: list[str], member_id_col: str) -> list[str]:
    order = []
    if "canonical_surname" in cols:
        order.append("canonical_surname")
    if "cluster_id" in cols:
        order.append("cluster_id")
    if "match_id" in cols and "cluster_id" not in order:
        order.append("match_id")
    if "victim_surname_norm" in cols:
        order.append("victim_surname_norm")
    if "victim_forename_norm" in cols:
        order.append("victim_forename_norm")
    if "victim_name_raw" in cols:
        order.append("victim_name_raw")
    if "article_id" in cols:
        order.append("article_id")
    if member_id_col in cols and member_id_col not in order:
        order.append(member_id_col)
    if not order and member_id_col in cols:
        order.append(member_id_col)
    if not order:
        order.append("cluster_id")
    return order


def _select_list(
    cols: list[str], available: set[str], table_alias: str
) -> str:
    parts = []
    for col in cols:
        if col in available:
            parts.append(f'{table_alias}.{_qident(col)} AS {_qident(col)}')
        else:
            parts.append(f'NULL AS {_qident(col)}')
    return ", ".join(parts)


def _build_compare_sql(
    req: ClusterCompareRequest,
    merged_cols: list[str],
    order_cols: list[str],
    left_cols: set[str],
    right_cols: set[str],
) -> SQL:
    left_table = _qtable(req.left_table)
    right_table = _qtable(req.right_table)
    out_table = _qtable(req.output_table)
    member_col = _qident(req.member_id_col)
    cluster_col = _qident(req.cluster_id_col)

    order_expr = ", ".join([_qident("source")] + [_qident(c) for c in order_cols])
    left_select = _select_list(merged_cols, left_cols, "l")
    right_select = _select_list(merged_cols, right_cols, "r")

    return SQL(
        f"""--sql
CREATE OR REPLACE TABLE {out_table} AS
WITH
  left_sig AS (
    SELECT
      {cluster_col} AS cluster_id,
      string_agg(CAST({member_col} AS VARCHAR),
      '|' ORDER BY CAST({member_col} AS VARCHAR)) AS signature
    FROM {left_table}
    GROUP BY {cluster_col}
  ),
  right_sig AS (
    SELECT
      {cluster_col} AS cluster_id,
      string_agg(CAST({member_col} AS VARCHAR),
      '|' ORDER BY CAST({member_col} AS VARCHAR)) AS signature
    FROM {right_table}
    GROUP BY {cluster_col}
  ),
  left_unmatched AS (
    SELECT l.cluster_id
    FROM left_sig l
    LEFT JOIN right_sig r
      ON l.signature = r.signature
    WHERE r.signature IS NULL
  ),
  right_unmatched AS (
    SELECT r.cluster_id
    FROM right_sig r
    LEFT JOIN left_sig l
      ON r.signature = l.signature
    WHERE l.signature IS NULL
  )
SELECT 1 AS source, {left_select}
FROM {left_table} l
JOIN left_unmatched u
  ON l.{cluster_col} = u.cluster_id
UNION ALL
SELECT 2 AS source, {right_select}
FROM {right_table} r
JOIN right_unmatched u
  ON r.{cluster_col} = u.cluster_id
ORDER BY {order_expr};
"""
    )


def _build_result(req: ClusterCompareRequest) -> Run[ClusterCompareResult]:
    return (
        sql_query(
            SQL(
                f"""--sql
SELECT
  source,
  COUNT(DISTINCT {req.cluster_id_col}) AS n
FROM {req.output_table}
GROUP BY source
ORDER BY source;
"""
            )
        )
        >> (
            lambda rows: pure(
                ClusterCompareResult(
                    output_table=req.output_table,
                    left_table=req.left_table,
                    right_table=req.right_table,
                    member_id_col=req.member_id_col,
                    left_unmatched_clusters=next(
                        (r["n"] for r in rows if r["source"] == 1), 0
                    ),
                    right_unmatched_clusters=next(
                        (r["n"] for r in rows if r["source"] == 2), 0
                    ),
                )
            )
        )
    )


def compare_cluster_tables(req: ClusterCompareRequest) -> Run[ClusterCompareResult]:
    """
    Compare two cluster tables and create a new DuckDB table of unmatched clusters.

    The output table includes all original columns plus a leading `source` column
    with values 1 (left) or 2 (right), sorted to match typical final_clusters
    exports when possible.
    """

    return _table_columns(req.left_table) >> (
        lambda left_cols: _table_columns(req.right_table)
        >> (
            lambda right_cols: (
                _validate_columns(req, left_cols, right_cols)
                >> (lambda _: _compare_with_columns(req, left_cols, right_cols))
            )
        )
    )


def _compare_with_columns(
    req: ClusterCompareRequest,
    left_cols: list[str],
    right_cols: list[str],
) -> Run[ClusterCompareResult]:
    left_set = set(left_cols)
    right_set = set(right_cols)
    if left_set != right_set:
        only_left = sorted(left_set - right_set)
        only_right = sorted(right_set - left_set)
        warn = (
            "Column mismatch; filling missing columns with NULLs. "
            f"Only in left: {only_left}. Only in right: {only_right}."
        )
        warn_run = put_line(f"[C] {warn}")
    else:
        warn_run = pure(None)

    merged_cols = [c for c in left_cols if c in right_set]
    merged_cols.extend([c for c in right_cols if c not in left_set])
    order_cols = _order_by_columns(merged_cols, req.member_id_col)

    return (
        warn_run
        ^ sql_exec(
            _build_compare_sql(
                req,
                merged_cols,
                order_cols,
                left_set,
                right_set,
            )
        )
        ^ put_line(
            f"[C] Wrote {req.output_table} with unmatched clusters from "
            f"{req.left_table} and {req.right_table}."
        )
        ^ _build_result(req)
    )


def _validate_columns(
    req: ClusterCompareRequest, left_cols: list[str], right_cols: list[str]
) -> Run[None]:
    missing = []
    for col in (req.cluster_id_col, req.member_id_col):
        if col not in left_cols:
            missing.append(f"{req.left_table}.{col}")
        if col not in right_cols:
            missing.append(f"{req.right_table}.{col}")
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return pure(None)
