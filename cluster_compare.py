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

    sort_cols = ["canonical_surname", "victim_surname_norm", "victim_forename_norm"]
    order_expr = ", ".join(
        [
            _qident("change_order"),
            _qident("family_canonical_surname") + " NULLS LAST",
            _qident("family_victim_surname_norm") + " NULLS LAST",
            _qident("family_victim_forename_norm") + " NULLS LAST",
            _qident("family_id"),
            _qident("source"),
        ]
        + [_qident(c) for c in order_cols]
    )
    left_select = _select_list(merged_cols, left_cols, "l")
    right_select = _select_list(merged_cols, right_cols, "r")
    left_sort_select = ", ".join(
        [
            (
                f"min(l.{_qident(c)}) AS {_qident(c)}"
                if c in left_cols
                else f"NULL AS {_qident(c)}"
            )
            for c in sort_cols
        ]
    )
    right_sort_select = ", ".join(
        [
            (
                f"min(r.{_qident(c)}) AS {_qident(c)}"
                if c in right_cols
                else f"NULL AS {_qident(c)}"
            )
            for c in sort_cols
        ]
    )

    return SQL(
        f"""--sql
CREATE OR REPLACE TABLE {out_table} AS
WITH
  left_members AS (
    SELECT
      {cluster_col} AS cluster_id,
      {member_col} AS member_id
    FROM {left_table}
  ),
  right_members AS (
    SELECT
      {cluster_col} AS cluster_id,
      {member_col} AS member_id
    FROM {right_table}
  ),
  left_sizes AS (
    SELECT
      cluster_id,
      COUNT(DISTINCT member_id) AS size
    FROM left_members
    GROUP BY cluster_id
  ),
  right_sizes AS (
    SELECT
      cluster_id,
      COUNT(DISTINCT member_id) AS size
    FROM right_members
    GROUP BY cluster_id
  ),
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
  ),
  left_unmatched_members AS (
    SELECT lm.cluster_id, lm.member_id
    FROM left_members lm
    JOIN left_unmatched lu
      ON lm.cluster_id = lu.cluster_id
  ),
  right_unmatched_members AS (
    SELECT rm.cluster_id, rm.member_id
    FROM right_members rm
    JOIN right_unmatched ru
      ON rm.cluster_id = ru.cluster_id
  ),
  left_right_intersection AS (
    SELECT
      l.cluster_id AS left_cluster_id,
      r.cluster_id AS right_cluster_id,
      COUNT(DISTINCT l.member_id) AS inter_cnt
    FROM left_unmatched_members l
    JOIN right_unmatched_members r
      ON l.member_id = r.member_id
    GROUP BY 1, 2
  ),
  left_change AS (
    SELECT
      lu.cluster_id,
      CASE
        WHEN EXISTS (
          SELECT 1
          FROM left_right_intersection i
          JOIN right_sizes rs
            ON rs.cluster_id = i.right_cluster_id
          JOIN left_sizes ls
            ON ls.cluster_id = lu.cluster_id
          WHERE i.left_cluster_id = lu.cluster_id
            AND i.inter_cnt = ls.size
        ) THEN 'merged'
        WHEN EXISTS (
          SELECT 1
          FROM left_right_intersection i
          JOIN right_sizes rs
            ON rs.cluster_id = i.right_cluster_id
          WHERE i.left_cluster_id = lu.cluster_id
            AND i.inter_cnt = rs.size
        ) THEN 'split'
        ELSE 'other'
      END AS change
    FROM left_unmatched lu
  ),
  right_change AS (
    SELECT
      ru.cluster_id,
      CASE
        WHEN EXISTS (
          SELECT 1
          FROM left_right_intersection i
          JOIN left_sizes ls
            ON ls.cluster_id = i.left_cluster_id
          WHERE i.right_cluster_id = ru.cluster_id
            AND i.inter_cnt = ls.size
        ) THEN 'merged'
        WHEN EXISTS (
          SELECT 1
          FROM left_right_intersection i
          JOIN right_sizes rs
            ON rs.cluster_id = ru.cluster_id
          WHERE i.right_cluster_id = ru.cluster_id
            AND i.inter_cnt = rs.size
        ) THEN 'split'
        ELSE 'other'
      END AS change
    FROM right_unmatched ru
  ),
  left_family AS (
    SELECT
      lu.cluster_id,
      lc.change,
      CASE
        WHEN lc.change = 'split' THEN lu.cluster_id
        WHEN lc.change = 'merged' THEN (
          SELECT i.right_cluster_id
          FROM left_right_intersection i
          JOIN left_sizes ls
            ON ls.cluster_id = lu.cluster_id
          JOIN right_sizes rs
            ON rs.cluster_id = i.right_cluster_id
          WHERE i.left_cluster_id = lu.cluster_id
            AND i.inter_cnt = ls.size
          ORDER BY rs.size DESC, i.right_cluster_id
          LIMIT 1
        )
        ELSE lu.cluster_id
      END AS family_id
    FROM left_unmatched lu
    JOIN left_change lc
      ON lc.cluster_id = lu.cluster_id
  ),
  right_family AS (
    SELECT
      ru.cluster_id,
      rc.change,
      CASE
        WHEN rc.change = 'split' THEN (
          SELECT i.left_cluster_id
          FROM left_right_intersection i
          JOIN right_sizes rs
            ON rs.cluster_id = ru.cluster_id
          JOIN left_sizes ls
            ON ls.cluster_id = i.left_cluster_id
          WHERE i.right_cluster_id = ru.cluster_id
            AND i.inter_cnt = rs.size
          ORDER BY ls.size DESC, i.left_cluster_id
          LIMIT 1
        )
        WHEN rc.change = 'merged' THEN ru.cluster_id
        ELSE ru.cluster_id
      END AS family_id
    FROM right_unmatched ru
    JOIN right_change rc
      ON rc.cluster_id = ru.cluster_id
  ),
  left_sort AS (
    SELECT
      l.{cluster_col} AS cluster_id,
      {left_sort_select}
    FROM {left_table} l
    GROUP BY l.{cluster_col}
  ),
  right_sort AS (
    SELECT
      r.{cluster_col} AS cluster_id,
      {right_sort_select}
    FROM {right_table} r
    GROUP BY r.{cluster_col}
  ),
  left_family_sort AS (
    SELECT
      lf.cluster_id,
      lf.change,
      lf.family_id,
      CASE
        WHEN lf.change = 'merged' THEN rs.canonical_surname
        ELSE ls.canonical_surname
      END AS family_canonical_surname,
      CASE
        WHEN lf.change = 'merged' THEN rs.victim_surname_norm
        ELSE ls.victim_surname_norm
      END AS family_victim_surname_norm,
      CASE
        WHEN lf.change = 'merged' THEN rs.victim_forename_norm
        ELSE ls.victim_forename_norm
      END AS family_victim_forename_norm
    FROM left_family lf
    LEFT JOIN left_sort ls
      ON ls.cluster_id = lf.family_id
    LEFT JOIN right_sort rs
      ON rs.cluster_id = lf.family_id
  ),
  right_family_sort AS (
    SELECT
      rf.cluster_id,
      rf.change,
      rf.family_id,
      CASE
        WHEN rf.change = 'split' THEN ls.canonical_surname
        ELSE rs.canonical_surname
      END AS family_canonical_surname,
      CASE
        WHEN rf.change = 'split' THEN ls.victim_surname_norm
        ELSE rs.victim_surname_norm
      END AS family_victim_surname_norm,
      CASE
        WHEN rf.change = 'split' THEN ls.victim_forename_norm
        ELSE rs.victim_forename_norm
      END AS family_victim_forename_norm
    FROM right_family rf
    LEFT JOIN left_sort ls
      ON ls.cluster_id = rf.family_id
    LEFT JOIN right_sort rs
      ON rs.cluster_id = rf.family_id
  )
SELECT
  1 AS source,
  lfs.change AS change,
  CASE
    WHEN lfs.change = 'split' THEN 1
    WHEN lfs.change = 'merged' THEN 2
    ELSE 3
  END AS change_order,
  lfs.family_id AS family_id,
  lfs.family_canonical_surname AS family_canonical_surname,
  lfs.family_victim_surname_norm AS family_victim_surname_norm,
  lfs.family_victim_forename_norm AS family_victim_forename_norm,
  {left_select}
FROM {left_table} l
JOIN left_unmatched u
  ON l.{cluster_col} = u.cluster_id
JOIN left_family_sort lfs
  ON lfs.cluster_id = l.{cluster_col}
UNION ALL
SELECT
  2 AS source,
  rfs.change AS change,
  CASE
    WHEN rfs.change = 'split' THEN 1
    WHEN rfs.change = 'merged' THEN 2
    ELSE 3
  END AS change_order,
  rfs.family_id AS family_id,
  rfs.family_canonical_surname AS family_canonical_surname,
  rfs.family_victim_surname_norm AS family_victim_surname_norm,
  rfs.family_victim_forename_norm AS family_victim_forename_norm,
  {right_select}
FROM {right_table} r
JOIN right_unmatched u
  ON r.{cluster_col} = u.cluster_id
JOIN right_family_sort rfs
  ON rfs.cluster_id = r.{cluster_col}
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
