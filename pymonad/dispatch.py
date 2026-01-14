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
            job.settings | {"retain_intermediate_calculation_columns": True},
            db_api=db_api,
        )

        linker.training.estimate_probability_two_random_records_match(
            list(job.deterministic_rules), recall=job.deterministic_recall
        )
        linker.training.estimate_u_using_random_sampling(1e6)

        if job.train_first:
            # Prefer explicit training rule from intent, otherwise fall back
            # to the blocking rules used for prediction.
            for _ in range(1):  # 1 iteration of EM
                for training_rule in job.training_blocking_rules or job.settings.get(
                    "blocking_rules_to_generate_predictions", []
                ):
                    linker.training.estimate_parameters_using_expectation_maximisation(
                        blocking_rule=training_rule
                    )

        df_pairs = linker.inference.predict(
            threshold_match_probability=job.predict_threshold
        )

        if job.visualize:
            pd_pairs = df_pairs.as_pandas_dataframe()
            print(len(pd_pairs))
            inspect_df = pd_pairs[
                ((pd_pairs["midpoint_day_l"] == 2624)
                    | (pd_pairs["midpoint_day_l"] == 2624))
                    & (pd_pairs["midpoint_day_r"] == 2624)
            ]
            print(len(inspect_df))
            inspect_dict = cast(list[dict[str, Any]], inspect_df.to_dict(orient="records"))
            alt.renderers.enable("browser")
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
            pairs = con.execute(f"SELECT unique_id_l, unique_id_r, match_probability FROM {job.pairs_out} WHERE match_probability > 0").fetchall()
            # Create bipartite graph
            G: nx.Graph = nx.Graph() #pylint: disable=C0103
            # Add edges with weights
            for row in pairs:
                G.add_edge(row[0], row[1], weight=row[2])
            # Compute maximum weight matching
            matching = nx.max_weight_matching(G)
            # Extract matched pairs
            matched_pairs = []
            seen = set()
            for u, v in matching:
                if u not in seen and v not in seen:
                    weight = G[u][v]['weight']
                    matched_pairs.append((u, v, weight))
                    seen.add(u)
                    seen.add(v)
            # Persist to unique_pairs_table
            if matched_pairs:
                values_str = ', '.join(f"({repr(u)}, {repr(v)}, {w})" for u, v, w in matched_pairs)
                con.execute(f"CREATE OR REPLACE TABLE {job.unique_pairs_table} AS SELECT * FROM (VALUES {values_str}) AS t(unique_id_l, unique_id_r, match_probability)")
            else:
                # Create empty table
                con.execute(f"CREATE OR REPLACE TABLE {job.unique_pairs_table} (unique_id_l VARCHAR, unique_id_r VARCHAR, match_probability DOUBLE)")
        if job.do_cluster:
            df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
                df_pairs,
                threshold_match_probability=job.cluster_threshold,
            )
            con.execute(
                f"CREATE OR REPLACE TABLE {job.clusters_out} AS "
                f"SELECT * FROM {df_clusters.physical_name}"
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
