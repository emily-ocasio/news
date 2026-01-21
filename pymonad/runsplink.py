"""
Intent, eliminator, and smart constructors for Splink.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence, TypeVar, cast

import altair as alt
import duckdb
import networkx as nx
import pandas as pd
from pandas import DataFrame
from splink import Linker, DuckDBAPI, blocking_analysis
from splink.internals.blocking_rule_creator import BlockingRuleCreator

# pylint:disable=W0212
from .environment import DbBackend, Environment
from .run import Run, ask, throw, ErrorPayload, put_splink_linker, get_splink_linker, \
    HasSplinkLinker
from .runsql import SQL, sql_exec, sql_query, sql_register, with_duckdb
from .array import Array

A = TypeVar("A")

@dataclass(frozen=True)
class SplinkDedupeJob:
    """
    Splink deduplication intent
    """
    input_table: str | list[str]
    settings: dict
    predict_threshold: float
    cluster_threshold: float
    pairs_out: str
    clusters_out: str
    deterministic_rules: Sequence[str | BlockingRuleCreator]
    deterministic_recall: float
    train_first: bool = False
    training_blocking_rules: Sequence[str] | None = None
    do_cluster: bool = True
    visualize: bool = False
    unique_matching: bool = False
    unique_pairs_table: str = ""
    em_max_runs: int = 3
    em_min_runs: int = 1
    em_stop_delta: float = 0.002
    splink_key: Any = None


@dataclass(frozen=True)
class SplinkVisualizeJob:
    """
    Splink visualization intent
    """
    splink_key: Any
    left_midpoints: Sequence[int] | None = None
    right_midpoints: Sequence[int] | None = None


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


def _resolve_pair_id_cols(
    pair_cols: set[str],
    unique_id_column_name: str,
    pairs_table: str,
) -> tuple[str, str]:
    left_id_col = f"{unique_id_column_name}_l"
    right_id_col = f"{unique_id_column_name}_r"
    if left_id_col in pair_cols and right_id_col in pair_cols:
        return left_id_col, right_id_col
    if "unique_id_l" in pair_cols and "unique_id_r" in pair_cols:
        return "unique_id_l", "unique_id_r"
    raise RuntimeError(
        f"Constrained clustering cannot find id columns in {pairs_table}. "
        f"Expected ({left_id_col}, {right_id_col}) or (unique_id_l, unique_id_r)."
    )


def _constrained_greedy_clusters(
    *,
    nodes: list[tuple[str, str]],
    edges: list[tuple[str, str, float]],
    unique_id_column_name: str,
) -> DataFrame:
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
    em_stop_delta: float = 0.002,
    splink_key: Any = None


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
                splink_key,
            ),
            self,
        ),
        lambda i, c: c._perform(i, c),
    )


def splink_visualize_job(
    splink_key: Any,
    left_midpoints: Sequence[int] | None = None,
    right_midpoints: Sequence[int] | None = None,
) -> Run[None]:
    """
    Smart constructor for SplinkVisualizeJob intent.
    """
    return Run(
        lambda self: self._perform(
            SplinkVisualizeJob(
                splink_key=splink_key,
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
    link_type = settings.get("link_type", "dedupe_only")
    unique_id_col = settings.get("unique_id_column_name", "unique_id")
    left_id_col: str | None = f"{unique_id_col}_l"
    right_id_col: str | None = f"{unique_id_col}_r"

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
    chart = linker.visualisations.m_u_parameters_chart()
    chart.show()  # type: ignore
    print("\nGenerating comparison viewer dashboard…")
    linker.visualisations.comparison_viewer_dashboard(
        df_pairs,
        "comparison_viewer.html",
        overwrite=True,
        num_example_rows=5,
    )
    print("Comparison viewer dashboard written to comparison_viewer.html")

    if link_type == "dedupe_only":
        print("\nGenerating unconstrained clusters for charting…")
        df_clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
            df_pairs, 0.01
        )
        inspect_ids: set[Any] = set()
        if left_id_col and left_id_col in inspect_df.columns:
            inspect_ids.update(inspect_df[left_id_col].dropna().tolist())
        if right_id_col and right_id_col in inspect_df.columns:
            inspect_ids.update(inspect_df[right_id_col].dropna().tolist())
        print("\nGenerating Cluster Studio dashboard…")
        try:
            clusters_df = df_clustered.as_pandas_dataframe()
            if inspect_ids:
                filtered = clusters_df[clusters_df[unique_id_col].isin(inspect_ids)]
            else:
                filtered = clusters_df.iloc[0:0]
            cluster_ids = sorted(filtered["cluster_id"].dropna().unique().tolist())
        except Exception as exc: #pylint: disable=W0718
            print(f"Unable to compute cluster_ids for dashboard: {exc}")
            cluster_ids = []
        if not cluster_ids:
            print("No clusters found for the waterfall chart rows; skipping dashboard.")
            return
        try:
            pairs_df = df_pairs.as_pandas_dataframe()
            mask = pd.Series(False, index=pairs_df.index)
            if left_id_col and left_id_col in pairs_df.columns:
                mask |= pairs_df[left_id_col].isin(inspect_ids)
            if right_id_col and right_id_col in pairs_df.columns:
                mask |= pairs_df[right_id_col].isin(inspect_ids)
            filtered_pairs = pairs_df.loc[mask]
            filtered_clusters = clusters_df[clusters_df[unique_id_col].isin(inspect_ids)]
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


def _run_splink_dedupe_with_conn(
    job: SplinkDedupeJob,
    con: duckdb.DuckDBPyConnection,
    current: "Run[Any]",
) -> tuple[Linker, str, str]:
    def _duckdb_exec(sql: str) -> None:
        with_duckdb(sql_exec(SQL(sql)))._step(current)

    def _duckdb_query(sql: str) -> Array:
        return with_duckdb(sql_query(SQL(sql)))._step(current)

    def _duckdb_register(name: str, df: DataFrame) -> None:
        with_duckdb(sql_register(name, df))._step(current)

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
    linker.training.estimate_u_using_random_sampling(1e8)

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
    _duckdb_exec(
        f"CREATE OR REPLACE TABLE {job.pairs_out} AS "
        f"SELECT * FROM {df_pairs.physical_name}"
    )
    if job.settings.get("link_type", "dedupe") == "link_only":
        _duckdb_exec(f"""
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
        pairs = _duckdb_query(
            f"""
            SELECT unique_id_l, unique_id_r, match_probability 
            FROM {job.pairs_out} WHERE match_probability > 0
            """
        )
        # Create bipartite graph
        G: nx.Graph = nx.Graph() #pylint: disable=C0103
        # Add edges with weights
        for row in pairs:
            G.add_edge(
                f"l_{row['unique_id_l']}",
                f"r_{row['unique_id_r']}",
                weight=row["match_probability"]
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
            _duckdb_exec(
                f"CREATE OR REPLACE TABLE {job.unique_pairs_table} "
                f"AS SELECT * FROM (VALUES {values_str}) "
                "AS t(unique_id_l, unique_id_r, match_probability)"
            )
        else:
            # Create empty table
            _duckdb_exec(f"""--sql
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
        nodes_rows = _duckdb_query(
            f"""
            SELECT
              CAST({unique_id_col} AS VARCHAR) AS unique_id,
              CAST(exclusion_id AS VARCHAR) AS exclusion_id
            FROM {job.input_table}
            """
        )
        nodes = [
            (row["unique_id"], row["exclusion_id"])
            for row in nodes_rows
        ]
        pair_cols = {
            row["name"]
            for row in _duckdb_query(
                f"SELECT name FROM pragma_table_info('{job.pairs_out}')"
            )
        }
        left_id_col, right_id_col = _resolve_pair_id_cols(
            pair_cols,
            unique_id_col,
            job.pairs_out,
        )
        edges_rows = _duckdb_query(
            f"""
            SELECT
              CAST({left_id_col} AS VARCHAR) AS uid_l,
              CAST({right_id_col} AS VARCHAR) AS uid_r,
              match_probability
            FROM {job.pairs_out}
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
        clusters_df = _constrained_greedy_clusters(
            nodes=nodes,
            edges=edges,
            unique_id_column_name=unique_id_col,
        )
        _duckdb_register("_constrained_clusters_df", clusters_df)
        _duckdb_exec(
            f"CREATE OR REPLACE TABLE {job.clusters_out} AS "
            "SELECT * FROM _constrained_clusters_df"
        )

        counts_table = _cluster_counts_table_name(job.clusters_out)
        # Always materialize cluster counts as a table, but handle legacy views.
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
            FROM {job.clusters_out}
            GROUP BY cluster_id
            """
        )
    return (
        linker,
        job.unique_pairs_table if job.unique_matching and job.unique_pairs_table else job.pairs_out,
        job.clusters_out if job.do_cluster else ""
    )

def _splink_model_paths(splink_key: Any) -> tuple[Path, Path]:
    key_str = str(splink_key).replace("/", "_")
    base_dir = Path("splink_models")
    model_path = base_dir / f"splink_model_{key_str}.json"
    meta_path = base_dir / f"splink_model_{key_str}.meta.json"
    return model_path, meta_path


def _save_splink_model(
    splink_key: Any,
    linker: Linker,
    input_table: str | list[str],
) -> None:
    if splink_key is None:
        return
    model_path, meta_path = _splink_model_paths(splink_key)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    linker.misc.save_model_to_json(str(model_path), overwrite=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"input_table": input_table}, f, indent=2)


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
    def step(self_run: "Run[Any]") -> A:
        parent = self_run._perform

        def perform(intent: Any, current: "Run[Any]") -> Any:
            match intent:
                case SplinkDedupeJob():
                    env = cast(Environment, ask()._step(current))
                    con = env["connections"].get(DbBackend.DUCKDB)
                    if con is None:
                        return throw(
                            ErrorPayload("Splink requires a DuckDB connection.")
                        )._step(current)
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
