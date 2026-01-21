"""
Geocode incident addresses using MAR.
"""

from typing import cast
import json
import re
from pymonad import (
    Run,
    pure,
    put_line,
    sql_query,
    sql_exec,
    SQL,
    with_duckdb,
    ask,
    SQLParams,
    geocode_address,
    String,
    process_all,
    FailureDetail,
    FailureDetails,
    Invalid,
    Validator,
    FailureType,
    throw,
    ErrorPayload,
    Environment,
    Unit,
    unit,
    Array,
    sql_script,
    Left,
    Right,
    Either,
    StopProcessing
)
from incidents_setup import CREATE_VICTIMS_CACHED_ENH_SQL
from menuprompts import NextStep

CREATE_CACHE_SQL = SQL(
    r"""
CREATE TABLE IF NOT EXISTS mar_cache (
  input_address TEXT PRIMARY KEY,
  matched_address TEXT,
  x_lon DOUBLE,
  y_lat DOUBLE,
  raw_json TEXT
);
"""
)

# Pick your address source; here I’m using stg_article_incidents.location_raw.
# You can switch to a more precise field if your GPT already gives a full address.
SELECT_ADDRESSES_SQL = SQL(
        r"""
SELECT
    trim(coalesce(location_raw,'')) AS addr_raw,
    -- aggregate distinct article ids per address so we can log them on failures
    string_agg(DISTINCT cast(article_id AS VARCHAR), ',') AS article_ids
FROM stg_article_incidents
WHERE trim(coalesce(location_raw,'')) <> ''
GROUP BY trim(coalesce(location_raw,''));
"""
)


# Lookup cache
CACHE_GET_SQL = SQL(
    r"""
SELECT * FROM mar_cache WHERE input_address = ?;
"""
)

# Insert cache row
INSERT_CACHE_SQL = SQL(
    r"""
INSERT OR REPLACE INTO mar_cache (
  input_address, matched_address, x_lon, y_lat, raw_json
) VALUES (?, ?, ?, ?, ?)
"""
)

# Materialize into a stable table for joins downstream
CREATE_GEOCODED_VIEW_SQL = SQL(
    r"""
CREATE OR REPLACE VIEW stg_article_incidents_geo AS
SELECT
  i.*,
  mc.input_address,
  mc.matched_address,
  mc.x_lon AS lon,
  mc.y_lat AS lat,
  COALESCE(NULLIF(mc.matched_address,''), 
    mc.input_address, trim(coalesce(i.location_raw,''))) AS geo_address_norm
FROM stg_article_incidents i
LEFT JOIN mar_cache mc
  ON trim(coalesce(i.location_raw,'')) <> ''
 AND mc.input_address = regexp_replace(
       trim(coalesce(i.location_raw,'')),
       '(?i)\\s*,?\\s*(washington(,)?\\s*d\\.?c\\.?|dc)\\s*$',
       ''
     );
"""
)


def geocode_all_incident_addresses(env: Environment) -> Run[NextStep]:
    """
    - Ensures cache table exists
    - Pulls distinct addresses
    - For each address, if missing in cache: geocode via MAR -> insert
    - Builds view stg_article_incidents_geo
    """

    def _process_rows(rows: Array[dict]) -> Run[None]:
        """
        Geocode all addresses using applicative traversal:
        - validators: empty (no pre-checks)
        - happy path:
            * check cache
            * if missing: call MAR; if not ok -> validation failure; if ok -> cache
        - accumulate failures but keep processing others
        """
        # Local normalizer: strip trailing DC city/state; MAR wants street only
        dc_suffix_re = re.compile(r"(?i)\s*,?\s*(washington(,)?\s*d\.?c\.?|dc)\s*$")

        def normalize_for_mar(a: str) -> str:
            return dc_suffix_re.sub("", (a or "").strip())

        # Turn the row dicts into MAR-safe (addr_key, article_ids) pairs
        # article_ids is a CSV of distinct article IDs for that address
        addr_items: Array[tuple[str, str]] = (
            lambda r: (
                normalize_for_mar(cast(str, r["addr_raw"])),
                cast(str, r.get("article_ids") or ""),
            )
        ) & rows

        # --- Validation wiring (empty validators) ---

        # Optional, descriptive failure enum (type annotation only)
        class GeoFailureType(FailureType):
            """Enum of failure types for geocoding"""

            GEOCODE_FAILED = "GEOCODE_FAILED"
            UNCAUGHT_EXCEPTION = "UNCAUGHT_EXCEPTION"

        validators: Array[Validator] = Array(())  # no validators for now

        # Render a thrown ErrorPayload (from happy path) into FailureDetails
        def render(err: ErrorPayload) -> FailureDetails[str]:
            return Array(
                (
                    FailureDetail(
                        type=GeoFailureType.UNCAUGHT_EXCEPTION,
                        s=String(f"{err}"),
                    ),
                )
            )

        def unhappy(err: ErrorPayload) -> Run[Unit]:
            """Optional unhappy handler: log to console"""
            return put_line("[GEO] MAR error:") ^ put_line(f"  {err}\n") ^ pure(unit)

        # Happy path for a single address
        def happy(item: tuple[str, str]) -> Run[None]:
            addr_key, article_ids = item
            if addr_key.lower() == "unknown":
                # Special case: set lon/lat to null
                return put_line(f"[GEO] Special case 'unknown': {addr_key} articles={article_ids}") ^ \
                    sql_exec(
                        INSERT_CACHE_SQL,
                        SQLParams(
                            (
                                String(addr_key),
                                String(""),
                                None,  # x_lon null
                                None,  # y_lat null
                                String(json.dumps({})),
                            )
                        ),
                    ) ^ pure(None)
            else:
                return sql_query(CACHE_GET_SQL, SQLParams((String(addr_key),))) >> (
                    lambda rs:
                    # cached -> done (log differently for cached permanent failures)
                    (
                        put_line(f"[GEO] Already cached (permanent failure): {addr_key} articles={article_ids}") ^ pure(None)
                        if len(rs) > 0 and rs[0]["x_lon"] is None
                        else put_line(f"[GEO] Already cached (success): {addr_key}") ^ pure(None)
                        if len(rs) > 0
                        else
                        # not cached -> geocode and insert
                        put_line(f"[GEO] MAR request: {addr_key} articles={article_ids}")
                        ^ geocode_address(addr_key, env["mar_key"])
                        >> (
                            lambda g: (
                                # If MAR returned OK, cache success
                                sql_exec(
                                    INSERT_CACHE_SQL,
                                    SQLParams(
                                        (
                                            String(addr_key),
                                            String(g.get("matched_address", "")),
                                            g.get("x_lon", 0),
                                            g.get("y_lat", 0),
                                            String(json.dumps(g.get("raw_json", {}))),
                                        )
                                    ),
                                )
                                ^ put_line(f"[GEO] MAR response OK, matched address: {g['matched_address']} articles={article_ids}")
                                ^ pure(None)
                                if g.get("ok")
                                else
                                # If not OK but "No Result present", cache failure (permanent)
                                sql_exec(
                                    INSERT_CACHE_SQL,
                                    SQLParams(
                                        (
                                            String(addr_key),
                                            String(""),
                                            None,  # x_lon null
                                            None,  # y_lat null
                                            String(json.dumps(g.get("raw_json", {}))),
                                        )
                                    ),
                                )
                                ^ put_line(f"[GEO] Cached permanent failure: {g} articles={article_ids}")
                                ^ pure(None)
                                if g.get("raw_json", {}).get("message") == "No Result present"
                                else
                                # Other failures: log and throw for retry
                                (put_line(f"[GEO] MAR transient failure for {addr_key} articles={article_ids}: {g}") ^
                                 throw(
                                    ErrorPayload(
                                        String(f"Error during match for '{addr_key}'"),
                                        app=g.get("raw_json", {}).get(
                                            "message", "no details"
                                        ),
                                    )
                                ))
                            )
                        )
                    )
                )

        # Process all with applicative accumulation (keeps going after failures)
        return process_all(
            validators=validators,
            render=render,
            happy=happy,
            items=addr_items,
            unhappy=unhappy,
        ) >> (lambda result:
            (
                put_line(
                    "[GEO] Geocoding stopped by user after "
                    f"{result.l.acc.processed} of {addr_items.length} addresses.\n"
                ) ^ pure(None)
            )
            if isinstance(result, Left)
            else (
                put_line(
                    f"[GEO] {result.r.validity.l.length} address failures accumulated."
                )
                ^ pure(None)
                if isinstance(result.r.validity, Invalid)
                else pure(None)
            )
        )

    return (
        sql_script(SQL(r"""
            LOAD splink_udfs;
        """)) ^
        sql_exec(CREATE_CACHE_SQL)
        ^ sql_query(SELECT_ADDRESSES_SQL)
        >> (
            lambda rows: put_line(f"[GEO] Found {len(rows)} distinct raw addresses.")
            ^ _process_rows(Array(tuple(rows)))
        )
        ^ sql_exec(CREATE_GEOCODED_VIEW_SQL)
        ^ put_line("[GEO] Built view stg_article_incidents_geo.")
        ^ sql_exec(
            SQL(
                r"""
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS geo_address_norm TEXT;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS lon DOUBLE;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS lat DOUBLE;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            UPDATE victims_cached vc
            SET geo_address_norm = g.geo_address_norm,
                lon = g.lon,
                lat = g.lat
            FROM stg_article_incidents_geo g
            WHERE vc.article_id = g.article_id
              AND vc.incident_idx = g.incident_idx;
        """
            )
        )
        ^ put_line(
            "[GEO] Populated geo_address_norm/lon/lat in victims_cached."
        )  # no DROP required — CREATE_VICTIMS_CACHED_ENH_SQL is CREATE OR REPLACE
        ^ sql_exec(CREATE_VICTIMS_CACHED_ENH_SQL)
        ^ put_line("[GEO] Rebuilt victims_cached_enh with geocode columns.")
        ^ pure(NextStep.CONTINUE)
    )


def geocode_incidents() -> Run[NextStep]:
    """
    Entry point for geocode incident controller
    """
    return with_duckdb(ask() >> geocode_all_incident_addresses)
