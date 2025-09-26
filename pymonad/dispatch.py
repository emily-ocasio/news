"""
Defines functions with side effects and maps them to intents
"""
from dataclasses import dataclass
from typing import Callable, TypedDict
import time
import requests

import duckdb
from splink import Linker, DuckDBAPI
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
    x_lon: float           # MAR returns X/Y; X = lon, Y = lat
    y_lat: float
    score: float
    ssl: str               # Street Segment ID (very useful!)
    anc: str
    ward: str
    psa: str
    raw_json: dict

@dataclass(frozen=True)
class PutLine:
    """Base I/O: output a line."""
    s: str
    end: str = '\n'

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
    input_table: str
    settings: dict
    predict_threshold: float
    cluster_threshold: float
    pairs_out: str
    clusters_out: str
    train_first: bool = False
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
    return String(input(x.prompt if x.prompt[-1] == ' ' else x.prompt + ' '))

@intentdef(SplinkDedupeJob)
def _splink_dedupe(job: SplinkDedupeJob) -> tuple[str, str]:
    db_api = DuckDBAPI(connection=duckdb.connect(job.duckdb_path))
    linker = Linker(job.input_table, job.settings, db_api=db_api)

    if job.train_first:
        # Use your prediction blocking rules for training too
        training_rules = job.settings.get(
            "blocking_rules_to_generate_predictions", [])
        # Splink v4: British spelling + explicit training rules param
        linker.training.estimate_parameters_using_expectation_maximisation(
            blocking_rule=training_rules
        )

    df_pairs = linker.inference.predict(
        threshold_match_probability=job.predict_threshold
    )
    df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_pairs,
        threshold_match_probability=job.cluster_threshold,
    )
    # 3) Persist outputs into stable tables in the same DB
    con = duckdb.connect(job.duckdb_path)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {job.pairs_out} AS "
                    f"SELECT * FROM {df_pairs.physical_name}")
        con.execute(f"CREATE OR REPLACE TABLE {job.clusters_out} AS "
                    f"SELECT * FROM {df_clusters.physical_name}")
    finally:
        con.close()

    return (job.pairs_out, job.clusters_out)

@intentdef(MarGeocode)
def _mar_geocode(x: MarGeocode) -> GeocodeResult:
    # MAR 2: https://geocoder.doc.dc.gov/api (findAddress2 endpoint)
    # Simple GET with 'address' and 'f=json'
    url = f"https://datagate.dc.gov/mar/open/api/v2.2/locations/{x.address}"
    params = {
        "apikey": x.mar_key
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        cand = (j.get("Result", {}) or {}).get("addresses", [])
        if not cand:
            return GeocodeResult(
                ok=False,
                normalized_input=x.address,
                matched_address="",
                x_lon=0,
                y_lat=0,
                score=0,
                ssl="",
                anc="",
                ward="",
                psa="",
                raw_json=j
        )
        # Take top candidate
        c0 = cand[0]
        addr = c0.get("address")
        score = c0.get("score")
        # Usually returned nested under 'location' or 'x','y' — MAR returns projection also as 'x','y'
        loc = c0.get("location", {})
        x_lon = loc.get("x") if isinstance(loc, dict) else c0.get("x")
        y_lat = loc.get("y") if isinstance(loc, dict) else c0.get("y")

        # Pull useful geography fields when present
        # (some fields are under 'marMatchAddr' / 'nodeId' / 'ssl' etc.)
        mar_attrs = c0.get("marAttributes", {}) or {}
        ssl = mar_attrs.get("SSL") or mar_attrs.get("ssl")
        anc = mar_attrs.get("ANC")
        ward = mar_attrs.get("WARD")
        psa = mar_attrs.get("PSA")

        return GeocodeResult(
            ok=True,
            normalized_input=x.address,
            matched_address=addr,
            x_lon=float(x_lon) if x_lon is not None else 0,
            y_lat=float(y_lat) if y_lat is not None else 0,
            score=float(score) if score is not None else 0,
            ssl=str(ssl) if ssl is not None else "",
            anc=str(anc) if anc is not None else "",
            ward=str(ward) if ward is not None else "",
            psa=str(psa) if psa is not None else "",
            raw_json=j
        )
    except Exception as e:  # keep hard failure as not-ok result
        return GeocodeResult(
            ok=False,
            normalized_input=x.address,
            matched_address="",
            x_lon=0,
            y_lat=0,
            score=0,
            ssl="",
            anc="",
            ward="",
            psa="",
            raw_json={"error": str(e)}
        )

@intentdef(Sleep)
def _sleep(x: Sleep) -> None:
    time.sleep(max(0, x.ms) / 1000.0)
