# geocode_incidents.py
from typing import cast
import json
import re
from pymonad import Run, pure, put_line, sql_query, sql_exec, SQL, \
    with_duckdb, ask, SQLParams, geocode_address, String, process_all, \
    FailureDetail, FailureDetails, Invalid, Validator, FailureType, throw, \
    ErrorPayload, Environment
from pymonad import Array
from menuprompts import NextStep

CREATE_CACHE_SQL = SQL(r"""
CREATE TABLE IF NOT EXISTS mar_cache (
  input_address TEXT PRIMARY KEY,
  matched_address TEXT,
  x_lon DOUBLE,
  y_lat DOUBLE,
  score DOUBLE,
  ssl TEXT,
  anc TEXT,
  ward TEXT,
  psa TEXT,
  raw_json TEXT
);
""")

# Pick your address source; here I’m using stg_article_incidents.location_raw.
# You can switch to a more precise field if your GPT already gives a full address.
SELECT_ADDRESSES_SQL = SQL(r"""
SELECT DISTINCT
  trim(coalesce(location_raw,'')) AS addr_raw
FROM stg_article_incidents
WHERE trim(coalesce(location_raw,'')) <> '';
""")

def normalize_dc(a: str) -> str:
    """
    Normalize string before sending to MAR 2
    """
    a = a.strip()
    if a == "":
        return a
    if "washington" not in a.lower():
        return f"{a}, Washington, DC"
    return a

# Lookup cache
CACHE_GET_SQL = SQL(r"""
SELECT * FROM mar_cache WHERE input_address = ?;
""")

# Insert cache row
INSERT_CACHE_SQL = SQL(r"""
INSERT OR REPLACE INTO mar_cache (
  input_address, matched_address, x_lon, y_lat, score, ssl, anc, ward, psa, raw_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""")

# Materialize into a stable table for joins downstream
CREATE_GEOCODED_VIEW_SQL = SQL(r"""
CREATE OR REPLACE VIEW stg_article_incidents_geo AS
SELECT
  i.*,
  mc.matched_address,
  mc.x_lon AS lon,
  mc.y_lat AS lat,
  mc.score AS geocode_score,
  mc.ssl AS ssl
FROM stg_article_incidents i
LEFT JOIN mar_cache mc
  ON trim(coalesce(i.location_raw,'')) <> ''
 AND mc.input_address = CASE
   WHEN lower(i.location_raw) LIKE '%washington%'
     THEN trim(i.location_raw)
   ELSE trim(i.location_raw) || ', Washington, DC'
 END;
""")

def geocode_all_incident_addresses(env: Environment) -> Run[NextStep]:
    """
    - Ensures cache table exists
    - Pulls distinct addresses
    - For each address, if missing in cache: geocode via MAR -> insert
    - Builds view stg_article_incidents_geo
    """
    def _process_rows(rows: Array[dict], rate_limit_ms: int = 0) -> Run[None]:
        """
        Geocode all addresses using applicative traversal:
        - validators: empty (no pre-checks)
        - happy path:
            * check cache
            * if missing: call MAR; if not ok -> validation failure; if ok -> cache
        - accumulate failures but keep processing others
        """
        # Local normalizer: strip trailing DC city/state; MAR wants street only
        DC_SUFFIX_RE = re.compile(r"(?i)\s*,?\s*(washington(,)?\s*d\.?c\.?|dc)\s*$")
        def normalize_for_mar(a: str) -> str:
            return DC_SUFFIX_RE.sub("", (a or "").strip())

        # Turn the row dicts into the MAR-safe address keys (Array[str])
        addr_items: Array[str] = (lambda r: normalize_for_mar(cast(str, r["addr_raw"]))) & rows

        # --- Validation wiring (empty validators) ---

        # Optional, descriptive failure enum (type annotation only)
        class GeoFailureType(FailureType):
            GEOCODE_FAILED = "GEOCODE_FAILED"
            UNCAUGHT_EXCEPTION = "UNCAUGHT_EXCEPTION"

        validators: Array[Validator] = Array(())  # no validators for now

        # Render a thrown ErrorPayload (from happy path) into FailureDetails
        def render(err) -> FailureDetails[str]:
            return Array((FailureDetail(type=GeoFailureType.UNCAUGHT_EXCEPTION, s=String(f"{err}")),))

        # Happy path for a single address
        def happy(addr_key: str) -> Run[None]:
            return \
                sql_query(CACHE_GET_SQL, SQLParams((String(addr_key),))) >> (lambda rs:
                    # cached -> done
                    put_line("Already cached.\n") ^ pure(None) if len(rs) > 0 else
                    # not cached -> geocode and insert
                    put_line(f"[GEO] MAR: {addr_key}") ^
                    geocode_address(addr_key, env["mar_key"]) >> (lambda g:
                        (
                            # If MAR didn't return an OK candidate, fail this item
                            # by throwing; process_all will re-render it via `render`.
                            throw(ErrorPayload(String(f"MAR no match for '{addr_key}'")))
                            if not g.get("ok")
                            else
                            sql_exec(INSERT_CACHE_SQL, SQLParams((
                                String(addr_key),
                                String(g.get("matched_address", "")),
                                g.get("x_lon", 0),
                                g.get("y_lat", 0),
                                g.get("score", 0),
                                String(g.get("ssl", "")),
                                String(g.get("anc", "")),
                                String(g.get("ward", "")),
                                String(g.get("psa", "")),
                                String(json.dumps(g.get("raw_json", {})))
                            ))) ^ \
                                put_line(f"[GEO] MAR response: {g}") ^ pure(None)
                        )
                    )
                )

        # Process all with applicative accumulation (keeps going after failures)
        return process_all(
            validators=validators,
            render=render,
            happy=happy,
            items=addr_items
        ) >> (lambda v:
            (  # optional summary logging
                put_line(f"[GEO] {v.validity.l.length} address failures accumulated.") ^ \
                    put_line(f"{v.validity.l}") ^ pure(None)
                if isinstance(v.validity, Invalid)
                else pure(None)
            )
        )

    return \
        sql_exec(CREATE_CACHE_SQL) ^ \
        sql_query(SELECT_ADDRESSES_SQL) >> (lambda rows:
            put_line(f"[GEO] Found {len(rows)} distinct raw addresses.") ^
            _process_rows(Array(tuple(rows)))
        ) ^ \
        sql_exec(CREATE_GEOCODED_VIEW_SQL) ^ \
        put_line("[GEO] Built view stg_article_incidents_geo.") ^ \
        pure(NextStep.CONTINUE)




def geocode_incidents() -> Run[NextStep]:
    """
    Entry point for geocode incident controller
    """
    return with_duckdb(
        ask() >> geocode_all_incident_addresses
    )