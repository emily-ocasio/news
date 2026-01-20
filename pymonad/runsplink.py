"""
Intent, eliminator, and smart constructors for Splink.
"""

from typing import Any, Sequence, TypeVar, cast

import altair as alt
import duckdb
import networkx as nx
from pandas import DataFrame
from splink import Linker, DuckDBAPI, blocking_analysis
from splink.internals.blocking_rule_creator import BlockingRuleCreator

# pylint:disable=W0212
from .dispatch import SplinkDedupeJob, SplinkVisualizeJob
from .run import Run

A = TypeVar("A")

_SPLINK_STATE: dict[str, Any] = {
    "conn": None,
    "linker": None,
}


def get_latest_splink_linker() -> Any | None:
    """
    Return the latest Linker produced by Splink jobs in this process.
    """
    return _SPLINK_STATE.get("linker")


def close_splink_resources() -> None:
    """
    Close any cached Splink resources for this process.
    """
    conn = _SPLINK_STATE.get("conn")
    if conn is not None:
        try:
            conn.close()
        finally:
            _SPLINK_STATE["conn"] = None
            _SPLINK_STATE["linker"] = None


def _ensure_duckdb_connection(duckdb_path: str) -> duckdb.DuckDBPyConnection:
    conn = _SPLINK_STATE.get("conn")
    if conn is None:
        conn = duckdb.connect(duckdb_path)
        _SPLINK_STATE["conn"] = conn
    return conn


def _extract_em_params(
        settings_dict: dict[str, Any]
        ) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    """
    Flatten comparison level m/u probabilities to a stable dict
    keyed by (comparison_name, level_name).
    """
    params: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for comparison in settings_dict.get("comparisons", []):
        comp_name = comparison.get("comparison_description", "")
        for level in comparison.get("comparison_levels", []):
            level_name = level.get("comparison_vector_value", level.get("label", ""))
            m_prob = level.get("m_probability")
            u_prob = level.get("u_probability")
            params[(str(comp_name), str(level_name))] = (m_prob, u_prob)
    return params


def _cluster_counts_table_name(clusters_table: str) -> str:
    # Most of the pipeline expects e.g. victim_clusters -> victim_cluster_counts
    if "clusters" in clusters_table:
        return clusters_table.replace("clusters", "cluster_counts")
    return f"{clusters_table}_counts"


def _constrained_greedy_clusters(
    *,
    con: duckdb.DuckDBPyConnection,
    input_table: str,
    unique_id_column_name: str,
    exclusion_id_column_name: str,
    pairs_table: str,
    match_probability_threshold: float,
) -> DataFrame:
    """
    Build clusters using a greedy union-find over pairwise edges, with a hard constraint:
    no resulting cluster may contain two rows with the same exclusion id (e.g. article id).
    """
    # Load all nodes and their exclusion ids so we can ensure singleton clusters exist.
    try:
        nodes = con.execute(
            f"""
            SELECT
              CAST({unique_id_column_name} AS VARCHAR) AS unique_id,
              CAST({exclusion_id_column_name} AS VARCHAR) AS exclusion_id
            FROM {input_table}
            """
        ).fetchall()
    except duckdb.Error as e:
        raise RuntimeError(
            "Constrained clustering requires an exclusion id column on the input table; "
            f"failed selecting ({unique_id_column_name}, {exclusion_id_column_name}) from {input_table}."
        ) from e

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

    pair_cols = {
        r[0]
        for r in con.execute(
            f"SELECT name FROM pragma_table_info('{pairs_table}')"
        ).fetchall()
    }
    left_id_col = f"{unique_id_column_name}_l"
    right_id_col = f"{unique_id_column_name}_r"
    if left_id_col in pair_cols and right_id_col in pair_cols:
        pass
    elif "unique_id_l" in pair_cols and "unique_id_r" in pair_cols:
        left_id_col, right_id_col = "unique_id_l", "unique_id_r"
    else:
        raise RuntimeError(
            f"Constrained clustering cannot find id columns in {pairs_table}. "
            f"Expected ({left_id_col}, {right_id_col}) or (unique_id_l, unique_id_r)."
        )

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

    # Greedily merge edges by descending match probability.
    edges = con.execute(
        f"""
        SELECT
          CAST({left_id_col} AS VARCHAR) AS uid_l,
          CAST({right_id_col} AS VARCHAR) AS uid_r,
          match_probability
        FROM {pairs_table}
        WHERE match_probability >= {match_probability_threshold}
        ORDER BY
          match_probability DESC,
          CAST({left_id_col} AS VARCHAR),
          CAST({right_id_col} AS VARCHAR)
        """
    ).fetchall()

    # Ensure we don't error if an edge references a missing node (shouldn't happen).
    known = set(parent.keys())
    for uid_l, uid_r, _prob in edges:
        if uid_l in known and uid_r in known:
            union(uid_l, uid_r)

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

    return DataFrame(rows, columns=["cluster_id", unique_id_column_name])


def splink_dedupe_job(
    duckdb_path: str,
    input_table: str | list[str],
    settings: dict,
    predict_threshold: float = 0.05,
    cluster_threshold: float = 0,
    pairs_out: str = "incidents_pairs",
    clusters_out: str = "incidents_clusters",
    train_first: bool = False,
    training_blocking_rules: Sequence[str] | None = None,
    deterministic_rules: Sequence[str | BlockingRuleCreator] | None = None,
    deterministic_recall: float = 0.5,
    do_cluster: bool = True,
    visualize: bool = False,
    unique_matching: bool = False,
    unique_pairs_table: str = "unique_pairs",
    em_max_runs: int = 3,
    em_min_runs: int = 1,
    em_stop_delta: float = 0.002


) -> Run[tuple[Any, str, str]]:
    """
    Smart constructor for SplinkDedupeJob intent.
    Returns Run[(linker, pairs_table_name, clusters_table_name)]
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
                do_cluster,
                visualize,
                unique_matching,
                unique_pairs_table,
                em_max_runs,
                em_min_runs,
                em_stop_delta,
            ),
            self,
        ),
        lambda i, c: c._perform(i, c),
    )


def splink_visualize_job(
    linker: Any,
    left_midpoints: Sequence[int] | None = None,
    right_midpoints: Sequence[int] | None = None,
) -> Run[None]:
    """
    Smart constructor for SplinkVisualizeJob intent.
    """
    return Run(
        lambda self: self._perform(
            SplinkVisualizeJob(
                linker=linker,
                left_midpoints=list(left_midpoints) if left_midpoints else None,
                right_midpoints=list(right_midpoints) if right_midpoints else None,
            ),
            self,
        ),
        lambda i, c: c._perform(i, c),
    )


def _run_splink_visualize(job: SplinkVisualizeJob) -> None:
    linker = job.linker
    alt.renderers.enable("browser")
    settings = linker.misc.save_model_to_json(out_path=None)
    unique_id_col = settings.get("unique_id_column_name", "unique_id")
    left_id_col = f"{unique_id_col}_l"
    right_id_col = f"{unique_id_col}_r"

    df_pairs = linker.inference.predict(threshold_match_probability=0)
    pd_pairs = df_pairs.as_pandas_dataframe()
    inspect_df = pd_pairs

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
    print(f"Number of records in waterfall chart: {len(inspect_df)}\n")
    print("Waterfall chart members:")

    if left_id_col not in inspect_df.columns or right_id_col not in inspect_df.columns:
        if "unique_id_l" in inspect_df.columns and "unique_id_r" in inspect_df.columns:
            left_id_col, right_id_col = "unique_id_l", "unique_id_r"
        else:
            left_id_col, right_id_col = None, None

    display_cols = [
        col for col in [
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

    if len(inspect_df) > 0:
        inspect_dict = cast(list[dict[str, Any]], inspect_df.to_dict(orient="records"))
        waterfall = linker.visualisations.waterfall_chart(inspect_dict)
        waterfall.show()  # type: ignore
    else:
        print("No records match the requested midpoint filters; skipping waterfall chart.")

    chart = linker.visualisations.match_weights_chart()
    chart.show()  # type: ignore


def _run_splink_dedupe_with_conn(
    job: SplinkDedupeJob,
    con: duckdb.DuckDBPyConnection,
) -> tuple[Linker, str, str]:
    db_api = DuckDBAPI(connection=con)
    linker = Linker(
        job.input_table,
        job.settings | {"retain_intermediate_calculation_columns": True, "max_iterations": 100},
        db_api=db_api,
    )
    prediction_rules = job.settings.get("blocking_rules_to_generate_predictions", [])
    if prediction_rules:
        tables = job.input_table if isinstance(job.input_table, list) else [job.input_table]
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

    linker.training.estimate_probability_two_random_records_match(
        list(job.deterministic_rules), recall=job.deterministic_recall
    )
    linker.training.estimate_u_using_random_sampling(1e6)

    if job.train_first:
        # Prefer explicit training rule from intent, otherwise fall back
        # to the blocking rules used for prediction.
        prev_params: dict[tuple[str, str], tuple[float | None, float | None]] | None = None
        for run_idx in range(job.em_max_runs):
            prev_block_params: dict[tuple[str, str], tuple[float | None, float | None]] | None = None
            current_params: dict[tuple[str, str], tuple[float | None, float | None]] | None = None
            for training_rule in job.training_blocking_rules or job.settings.get(
                "blocking_rules_to_generate_predictions", []
            ):
                linker.training.estimate_parameters_using_expectation_maximisation(
                    blocking_rule=training_rule
                )
                current_settings = linker.misc.save_model_to_json(out_path=None)
                current_params = _extract_em_params(current_settings)
                deltas: list[float] = []
                if prev_block_params is not None:
                    for key, (m_now, u_now) in current_params.items():
                        if key not in prev_block_params:
                            continue
                        m_prev, u_prev = prev_block_params[key]
                        if m_now is not None and m_prev is not None:
                            deltas.append(abs(m_now - m_prev))
                        if u_now is not None and u_prev is not None:
                            deltas.append(abs(u_now - u_prev))
                    max_delta = max(deltas) if deltas else 1.0
                    print(
                        f"EM run {run_idx + 1}/{job.em_max_runs} block drift: "
                        f"max_delta={max_delta:.6f} (block to block)"
                    )
                prev_block_params = current_params
            if current_params is None:
                break
            if prev_params is not None:
                for key, (m_now, u_now) in current_params.items():
                    if key not in prev_params:
                        continue
                    m_prev, u_prev = prev_params[key]
                    if m_now is not None and m_prev is not None:
                        deltas.append(abs(m_now - m_prev))
                    if u_now is not None and u_prev is not None:
                        deltas.append(abs(u_now - u_prev))
                max_delta = max(deltas) if deltas else 1.0
                print(
                    f"EM run {run_idx + 1}/{job.em_max_runs}: "
                    f"max_delta={max_delta:.6f} "
                    f"(stop if < {job.em_stop_delta:.6f} after {job.em_min_runs} runs)"
                )
                if run_idx + 1 >= job.em_min_runs and max_delta < job.em_stop_delta:
                    break
            prev_params = current_params
    df_pairs = linker.inference.predict(
        threshold_match_probability=job.predict_threshold
    )

    # Persist outputs into stable tables in the same DB
    con.execute(
        f"CREATE OR REPLACE TABLE {job.pairs_out} AS "
        f"SELECT * FROM {df_pairs.physical_name}"
    )
    if job.settings.get("link_type", "dedupe") == "link_only":
        con.execute(f"""
            CREATE OR REPLACE TABLE {job.pairs_out}_top1 AS
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
            FROM {job.pairs_out}
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1;
            """)
    if job.unique_matching and job.unique_pairs_table:
        # Query all pairs with match_probability > 0
        pairs = con.execute(
            f"""
            SELECT unique_id_l, unique_id_r, match_probability 
            FROM {job.pairs_out} WHERE match_probability > 0
            """
            ).fetchall()
        # Create bipartite graph
        G: nx.Graph = nx.Graph() #pylint: disable=C0103
        # Add edges with weights
        for row in pairs:
            G.add_edge(
                f"l_{row[0]}",
                f"r_{row[1]}",
                weight=row[2]
            )
        # Compute maximum weight matching
        matching = nx.max_weight_matching(G)
        # Extract matched pairs
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
        # Persist to unique_pairs_table
        if matched_pairs:
            values_str = ', '.join(f"({repr(u)}, {repr(v)}, {w})" for u, v, w in matched_pairs)
            con.execute(f"CREATE OR REPLACE TABLE {job.unique_pairs_table} AS SELECT * FROM (VALUES {values_str}) AS t(unique_id_l, unique_id_r, match_probability)")
        else:
            # Create empty table
            con.execute(f"""--sql
                CREATE OR REPLACE TABLE {job.unique_pairs_table}
                (
                    unique_id_l VARCHAR, 
                    unique_id_r VARCHAR, 
                    match_probability DOUBLE
                )
                """)
    if job.do_cluster:
        if not isinstance(job.input_table, str):
            raise RuntimeError(
                "Constrained clustering currently requires a single input table name "
                f"(got {type(job.input_table).__name__})."
            )

        unique_id_col = cast(str, job.settings.get("unique_id_column_name", "unique_id"))
        clusters_df = _constrained_greedy_clusters(
            con=con,
            input_table=job.input_table,
            unique_id_column_name=unique_id_col,
            exclusion_id_column_name="exclusion_id",
            pairs_table=job.pairs_out,
            match_probability_threshold=job.cluster_threshold,
        )
        con.register("_constrained_clusters_df", clusters_df)
        con.execute(
            f"CREATE OR REPLACE TABLE {job.clusters_out} AS "
            "SELECT * FROM _constrained_clusters_df"
        )

        counts_table = _cluster_counts_table_name(job.clusters_out)
        # Some downstream code creates `{x}_cluster_counts` as a VIEW; ensure
        # we don't fail if the object exists with a different type.
        try:
            con.execute(f"DROP VIEW IF EXISTS {counts_table}")
        except duckdb.Error:
            pass
        try:
            con.execute(f"DROP TABLE IF EXISTS {counts_table}")
        except duckdb.Error:
            pass
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {counts_table} AS
            SELECT cluster_id, COUNT(*)::BIGINT AS member_count
            FROM {job.clusters_out}
            GROUP BY cluster_id
            """
        )
    return (
        linker,
        job.unique_pairs_table if job.unique_matching and job.unique_pairs_table else job.pairs_out,
        job.clusters_out if job.do_cluster else ""
    )


def run_splink(prog: Run[A]) -> Run[A]:
    """
    Eliminator for Splink intents.
    """
    def step(self_run: "Run[Any]") -> A:
        parent = self_run._perform

        def perform(intent: Any, current: "Run[Any]") -> Any:
            match intent:
                case SplinkDedupeJob():
                    con = _ensure_duckdb_connection(intent.duckdb_path)
                    out = _run_splink_dedupe_with_conn(intent, con)
                    _SPLINK_STATE["linker"] = out[0]
                    return out
                case SplinkVisualizeJob():
                    return _run_splink_visualize(intent)
                case _:
                    return parent(intent, current)

        inner = Run(prog._step, perform)
        return inner._step(inner)

    return Run(step, lambda i, c: c._perform(i, c))
