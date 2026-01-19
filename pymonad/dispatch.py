"""
Defines functions with side effects and maps them to intents
"""

from dataclasses import dataclass
from typing import Callable, TypedDict, Sequence, cast, Any
import json
import time
import requests

import altair as alt
import duckdb
from pandas import DataFrame
from splink import Linker, DuckDBAPI
from splink import blocking_analysis
from splink.internals.blocking_rule_creator import BlockingRuleCreator
import networkx as nx

from .string import String


class InputPrompt(String):
    """
    Represents a prompt for user input.
    """


class GeocodeResult(TypedDict, total=False):
    """
    Result of geocode query
    """

    ok: bool
    normalized_input: str
    matched_address: str
    x_lon: float  # MAR returns X/Y; X = lon, Y = lat
    y_lat: float
    raw_json: dict


@dataclass(frozen=True)
class PutLine:
    """Base I/O: output a line."""

    s: str
    end: str = "\n"


@dataclass(frozen=True)
class GetLine:
    """Base I/O: input a line with prompt."""

    prompt: InputPrompt


@dataclass(frozen=True)
class SplinkDedupeJob:
    """
    Splink deduplication intent
    """
    duckdb_path: str
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


@dataclass(frozen=True)
class MarGeocode:
    """Effect: Geocode a DC address via MAR 2 API (findAddress2)."""

    address: str
    mar_key: str
    # You can add optional params here if needed (e.g., preferScoreMin, etc.)


@dataclass(frozen=True)
class Sleep:
    """Effect: Sleep for N milliseconds (for rate limiting)."""

    ms: int


REAL_DISPATCH: dict[type, Callable] = {}

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
    for u, v, _p in edges:
        if u in known and v in known:
            union(u, v)

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


def intentdef(intent: type) -> Callable[[Callable], Callable]:
    """
    Decorator for intent functions
    Registers the function in the REAL_DISPATCH dictionary
    """

    def decorator(func: Callable) -> Callable:
        REAL_DISPATCH[intent] = func
        return func

    return decorator


@intentdef(PutLine)
def _putline(x: PutLine) -> None:
    """
    Print a message to the console
    """
    print(x.s, end=x.end)


@intentdef(GetLine)
def _getline(x: GetLine) -> String:
    """
    Get a line of input from the user
    """
    return String(input(x.prompt if x.prompt[-1] == " " else x.prompt + " ").strip())


@intentdef(SplinkDedupeJob)
def _splink_dedupe(job: SplinkDedupeJob) -> tuple[str, str]:
    with duckdb.connect(job.duckdb_path) as con:
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
            threshold_match_probability= job.predict_threshold
        )
        if job.visualize:
            alt.renderers.enable("browser")
            pd_pairs = df_pairs.as_pandas_dataframe()
            print(f"Total predictions within threshold: {len(pd_pairs)}")
            inspect_df = pd_pairs
            try:
                inspect_df = inspect_df[(
                    # ((inspect_df["midpoint_day_l"] == 2864)
                    #     | (inspect_df["midpoint_day_l"] == 2864))
                    #     & (inspect_df["midpoint_day_r"] > 0)
                    (inspect_df["victim_count_l"] == 2)
                    & (inspect_df["victim_count_r"] == 2)
                    & (inspect_df["midpoint_day_l"] == 3267)
                    # & (inspect_df["month_r"] == 9)
                    )
                ]
            except KeyError as ke:
                print(f"Exception: {ke}")
                print("Column not found in inspect_df")
                print("Allowable columns:")
                print(f"{inspect_df.columns}")

            print(f"Number of records in waterfall chart: {len(inspect_df)}\n")
            print("Waterfall chart members:")
            print_df = inspect_df[["match_probability", "unique_id_l", "unique_id_r"]].reset_index()
            print(print_df)
            inspect_dict = cast(list[dict[str, Any]], inspect_df.to_dict(orient="records"))
            waterfall = linker.visualisations.waterfall_chart(inspect_dict)
            waterfall.show()  # type: ignore


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
        if job.visualize:
            alt.renderers.enable("browser")
            chart = linker.visualisations.match_weights_chart()
            chart.show()  # type: ignore
            mw = chart["data"].to_dict()["values"]
            mw_df = DataFrame(mw)
            linker.visualisations.comparison_viewer_dashboard(
                df_pairs,
                out_path="comparison_viewer.html",
                overwrite=True,
                num_example_rows=20)
            con.register("mw_df", mw_df)
            con.execute("CREATE OR REPLACE TABLE match_weights AS SELECT * FROM mw_df")
    return (job.unique_pairs_table if job.unique_matching and job.unique_pairs_table else job.pairs_out, job.clusters_out if job.do_cluster else "")


@intentdef(MarGeocode)
def _mar_geocode(x: MarGeocode) -> GeocodeResult:
    # MAR 2: https://geocoder.doc.dc.gov/api (findAddress2 endpoint)
    # Simple GET with 'address' and 'f=json'
    url = f"https://datagate.dc.gov/mar/open/api/v2.2/locations/{x.address}"
    params = {"apikey": x.mar_key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        j = resp.json()
    except requests.exceptions.Timeout as e:
        return GeocodeResult(
            ok=False,
            normalized_input=x.address,
            matched_address="",
            x_lon=0,
            y_lat=0,
            raw_json={"message": f"timeout error: {str(e)}"},
        )
    except requests.exceptions.SSLError as e:
        return GeocodeResult(
            ok=False,
            normalized_input=x.address,
            matched_address="",
            x_lon=0,
            y_lat=0,
            raw_json={"message": f"ssl error: {str(e)}"},
        )
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        return GeocodeResult(
            ok=False,
            normalized_input=x.address,
            matched_address="",
            x_lon=0,
            y_lat=0,
            raw_json={"message": f"http_error {code}, {str(e)}"},
        )
    except requests.exceptions.RequestException as e:
        return GeocodeResult(
            ok=False,
            normalized_input=x.address,
            matched_address="",
            x_lon=0,
            y_lat=0,
            raw_json={"message": f"network_error: {str(e)}"},
        )
    except (json.JSONDecodeError, ValueError) as e:
        return GeocodeResult(
            ok=False,
            normalized_input=x.address,
            matched_address="",
            x_lon=0,
            y_lat=0,
            raw_json={"message": f"invalid_json: {str(e)}"},
        )
    try:
        success = j.get("Success", False)
        if not success:
            return GeocodeResult(
                ok=False,
                normalized_input=x.address,
                matched_address="",
                x_lon=0,
                y_lat=0,
                raw_json={"message": j.get("message", "Success= False")},
            )
        result = j.get("Result", {})
        # This is what happens when MAR cannot recognize the address
        if not result:
            return GeocodeResult(
                ok=False,
                normalized_input=x.address,
                matched_address="",
                x_lon=0,
                y_lat=0,
                raw_json={"message": j.get("message", "No Result present")},
            )
        addresses = result.get("addresses", [])
        if not addresses:
            intersections = result.get("intersections", [])
            if not intersections:
                blocks = result.get("blocks", [])
                if not blocks:
                    return GeocodeResult(
                        ok=False,
                        normalized_input=x.address,
                        matched_address="",
                        x_lon=0,
                        y_lat=0,
                        raw_json={"message": "No addresses or intersections found"},
                    )
                # Take first block candidate
                b0 = blocks[0].get("block", {}).get("properties", {})
                return GeocodeResult(
                    ok=True,
                    normalized_input=x.address,
                    matched_address=b0.get("FullBlock", ""),
                    x_lon=float(b0.get("Longitude", 0)),
                    y_lat=float(b0.get("Latitude", 0)),
                    raw_json=j,
                )
            # Take first intersection candidate
            c0 = intersections[0].get("intersection", {}).get("properties", {})
            return GeocodeResult(
                ok=True,
                normalized_input=x.address,
                matched_address=c0.get("FullIntersection", ""),
                x_lon=float(c0.get("Longitude", 0)),
                y_lat=float(c0.get("Latitude", 0)),
                raw_json=j,
            )
        # Take top candidate
        c0 = addresses[0].get("address", {}).get("properties", {})

        return GeocodeResult(
            ok=True,
            normalized_input=x.address,
            matched_address=c0.get("FullAddress", ""),
            x_lon=float(c0.get("Longitude", 0)),
            y_lat=float(c0.get("Latitude", 0)),
            raw_json=j,
        )
    except (TypeError, ValueError) as e:
        return GeocodeResult(
            ok=False,
            normalized_input=x.address,
            matched_address="",
            x_lon=0,
            y_lat=0,
            raw_json={"message": f"parse_error: {str(e)}"},
        )


@intentdef(Sleep)
def _sleep(x: Sleep) -> None:
    time.sleep(max(0, x.ms) / 1000.0)
