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
    GeocodeResult,
    String,
    process_all,
    FailureDetail,
    FailureDetails,
    AddressResultType,
    mar_result_type_with_input,
    mar_result_score,
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
)
from incidents_setup import CREATE_VICTIMS_CACHED_ENH_SQL
from menuprompts import NextStep

CREATE_CACHE_SQL = SQL(
    """--sql
CREATE TABLE IF NOT EXISTS mar_cache (
  input_address TEXT PRIMARY KEY,
  matched_address TEXT,
  x_lon DOUBLE,
  y_lat DOUBLE,
  raw_json TEXT,
  address_type TEXT
);
"""
)

# Pick your address source; here I’m using stg_article_incidents.location_raw.
# You can switch to a more precise field if your GPT already gives a full address.
SELECT_ADDRESSES_SQL = SQL(
    """--sql
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
    """--sql
SELECT * FROM mar_cache WHERE input_address = ?;
"""
)

# Ensure address_type column exists on existing cache tables
ALTER_CACHE_SQL = SQL(
    """--sql
ALTER TABLE mar_cache ADD COLUMN IF NOT EXISTS address_type TEXT;
"""
)

# Insert cache row
INSERT_CACHE_SQL = SQL(
    """--sql
INSERT OR REPLACE INTO mar_cache (
  input_address, matched_address, x_lon, y_lat, raw_json, address_type
) VALUES (?, ?, ?, ?, ?, ?)
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
  COALESCE(mc.address_type, 'NO_SUCCESS') AS address_type,
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

        block_of_re = re.compile(r"\bBLOCK\b(?!\s+OF\b)")

        def normalize_for_mar(a: str) -> str:
            normalized = dc_suffix_re.sub("", (a or "").strip()).upper()
            normalized = normalized.replace(".", "")
            normalized = re.sub(
                r"\b(\d+)\s+1/2\s+(\w+)\s+(AVE|AV|AVENUE|ST|STREET|RD|ROAD|PL|PLACE|PLZ|PLAZA|TERR|TER|TERRACE|BLVD|BOULEVARD|PKWAY|PKWY|PARKWAY|HWY|HIGHWAY|DR|DRIVE|CT|COURT|LN|LANE|CIR|CIRCLE|WAY|SQ|SQUARE)\b",
                r"\1 \2 \3",
                normalized,
            )
            street_types = {
                "AVE": "AVENUE",
                "AV": "AVENUE",
                "RD": "ROAD",
                "PL": "PLACE",
                "PLZ": "PLAZA",
                "TERR": "TERRACE",
                "TER": "TERRACE",
                "BLVD": "BOULEVARD",
                "PKWAY": "PARKWAY",
                "PKWY": "PARKWAY",
                "HWY": "HIGHWAY",
                "DR": "DRIVE",
                "CT": "COURT",
                "LN": "LANE",
                "CIR": "CIRCLE",
                "WAY": "WAY",
                "SQ": "SQUARE",
            }
            for short, full in street_types.items():
                normalized = re.sub(rf"\b{short}\b", full, normalized)
            normalized = re.sub(r"(?<=\s)ST\b", "STREET", normalized)
            ordinals = {
                "FIRST": "1ST",
                "SECOND": "2ND",
                "THIRD": "3RD",
                "FOURTH": "4TH",
                "FIFTH": "5TH",
                "SIXTH": "6TH",
                "SEVENTH": "7TH",
                "EIGHTH": "8TH",
                "NINTH": "9TH",
            }
            for word, num in ordinals.items():
                normalized = re.sub(rf"\b{word}\b", num, normalized)
            return block_of_re.sub("BLOCK OF", normalized)

        def cache_result_type(
            raw_json_value: object, addr_key: str
        ) -> AddressResultType:
            if isinstance(raw_json_value, str):
                try:
                    parsed = json.loads(raw_json_value or "{}")
                except json.JSONDecodeError:
                    parsed = {}
            elif isinstance(raw_json_value, dict):
                parsed = raw_json_value
            else:
                parsed = {}
            return mar_result_type_with_input(addr_key, parsed)

        def cache_result_score(raw_json_value: object) -> float:
            if isinstance(raw_json_value, str):
                try:
                    parsed = json.loads(raw_json_value or "{}")
                except json.JSONDecodeError:
                    parsed = {}
            elif isinstance(raw_json_value, dict):
                parsed = raw_json_value
            else:
                parsed = {}
            return mar_result_score(parsed)

        def color_for_score(
            msg: str,
            article_ids: str,
            result_type: AddressResultType,
        ) -> str:
            if result_type == AddressResultType.UNRECOGNIZED_PLACE:
                labeled = msg if "type=" in msg else f"{msg} type={result_type.value}"
                colored = f"\x1b[31m{labeled}\x1b[0m"
                if "articles=" in msg:
                    return colored
                return f"{colored} articles={article_ids}"
            return msg

        def should_bypass_cache(rs: Array, addr_key: str) -> bool:
            if len(rs) == 0:
                return False
            if rs[0]["x_lon"] is None:
                return False
            rtype = cache_result_type(rs[0]["raw_json"], addr_key)
            if rtype != AddressResultType.BLOCK:
                return False
            matched = (rs[0].get("matched_address") or "")
            return "BLOCK OF" not in matched.upper()

        # Turn the row dicts into MAR-safe (addr_key, article_ids, addr_raw) pairs
        # article_ids is a CSV of distinct article IDs for that address
        addr_items: Array[tuple[str, str, str]] = (
            lambda r: (
                normalize_for_mar(cast(str, r["addr_raw"])),
                cast(str, r.get("article_ids") or ""),
                cast(str, r["addr_raw"]),
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
        def happy(item: tuple[str, str, str]) -> Run[None]:
            addr_key, article_ids, addr_raw = item
            if addr_key.lower() == "unknown":
                # Special case: set lon/lat to null
                return (
                    put_line(
                        "[GEO] Special case 'unknown':"
                        f" {addr_key} articles={article_ids}"
                    )
                    ^ sql_exec(
                        INSERT_CACHE_SQL,
                        SQLParams(
                            (
                                String(addr_key),
                                String(""),
                                None,  # x_lon null
                                None,  # y_lat null
                                        String(json.dumps({})),
                                        String(AddressResultType.NO_SUCCESS.value),
                                    )
                                ),
                            )
                    ^ pure(None)
                )
            else:
                def handle_mar(g: GeocodeResult) -> Run[None]:
                    address_type = mar_result_type_with_input(addr_key, g.raw_json).value
                    if g.ok:
                        return (
                            sql_exec(
                                INSERT_CACHE_SQL,
                                SQLParams(
                                    (
                                        String(addr_key),
                                        String(g.matched_address),
                                        g.x_lon,
                                        g.y_lat,
                                        String(json.dumps(g.raw_json)),
                                        String(address_type),
                                    )
                                ),
                            )
                            ^ put_line(
                                "[GEO] MAR response OK, matched address:"
                                f" {g.matched_address} articles={article_ids}"
                            )
                            ^ pure(None)
                        )
                    if g.raw_json.get("message") == "No Result present":
                        return (
                            sql_exec(
                                INSERT_CACHE_SQL,
                                SQLParams(
                                    (
                                        String(addr_key),
                                        String(""),
                                        None,  # x_lon null
                                        None,  # y_lat null
                                        String(json.dumps(g.raw_json)),
                                        String(address_type),
                                    )
                                ),
                            )
                            ^ put_line(
                                "[GEO] Cached permanent failure:"
                                f" {g} articles={article_ids}"
                            )
                            ^ pure(None)
                        )
                    return (
                        put_line(
                            "[GEO] MAR transient failure for"
                            f" {addr_key} articles={article_ids}: {g}"
                        )
                        ^ throw(
                            ErrorPayload(
                                String(f"Error during match for '{addr_key}'"),
                                app=g.raw_json.get("message", "no details"),
                            )
                        )
                    )

                def request_mar() -> Run[None]:
                    return (
                        put_line(
                            f"[GEO] MAR request: {addr_key} articles={article_ids}"
                        )
                        ^ geocode_address(addr_key, env["mar_key"])
                        >> handle_mar
                    )

                def handle_cache(rs: Array) -> Run[None]:
                    if len(rs) == 0:
                        return request_mar()
                    debug_run = (
                        put_line(
                            "[GEO][DEBUG] block normalization: "
                            f"raw={repr(addr_raw)} -> normalized={repr(addr_key)}"
                        )
                        if "BLOCK" in addr_key and "BLOCK OF" not in addr_key
                        else pure(None)
                    )
                    if rs[0]["x_lon"] is None:
                        return (
                            debug_run
                            ^ put_line(
                                f"[GEO] Already cached (permanent failure): {addr_key} "
                                f"articles={article_ids} "
                                f"type={cache_result_type(
                                            rs[0]['raw_json'], addr_key).value}"
                            )
                            ^ pure(None)
                        )
                    matched_address = rs[0].get("matched_address")
                    raw_json_value = rs[0].get("raw_json")
                    cached_type_value = rs[0].get("address_type")
                    score_value = cache_result_score(raw_json_value)
                    matched_is_missing = matched_address is None or (
                        isinstance(matched_address, str)
                        and matched_address.strip().lower() == "none"
                    )
                    def proceed(type_value: AddressResultType) -> Run[None]:
                        if should_bypass_cache(rs, addr_key):
                            if (
                                matched_is_missing
                                and type_value != AddressResultType.STREET_ONLY
                            ):
                                msg = (
                                    f"[GEO] Cache bypass for BLOCK w/o 'BLOCK': {addr_key} "
                                    f"type={type_value.value} score={score_value} "
                                    f"articles={article_ids}"
                                    if type_value == AddressResultType.UNRECOGNIZED_PLACE
                                    else
                                    "[GEO] Cache bypass for BLOCK w/o 'BLOCK':"
                                    f" {addr_key} "
                                    f"raw_json={raw_json_value} "
                                    f"type={type_value.value} "
                                    f"score={score_value} "
                                    f"articles={article_ids}"
                                )
                                return (
                                    debug_run
                                    ^ put_line(
                                        color_for_score(
                                            msg,
                                            article_ids,
                                            type_value,
                                        )
                                    )
                                    ^ request_mar()
                                )
                            return (
                                debug_run
                                ^ put_line(
                                    color_for_score(
                                        "[GEO] Cache bypass for BLOCK w/o 'BLOCK':"
                                        f" {addr_key} "
                                        f"matched={matched_address} "
                                        f"type={type_value.value} "
                                        f"score={score_value} "
                                        f"articles={article_ids}",
                                        article_ids,
                                        type_value,
                                    )
                                )
                                ^ request_mar()
                            )
                        if (
                            matched_is_missing
                            and type_value == AddressResultType.NAMED_PLACE
                        ):
                            return (
                                debug_run
                                ^ put_line(
                                    color_for_score(
                                        "[GEO] Cache refresh for NAMED_PLACE missing"
                                        f" match: {addr_key} "
                                        f"type={type_value.value} score={score_value} "
                                        f"articles={article_ids}",
                                        article_ids,
                                        type_value,
                                    )
                                )
                                ^ request_mar()
                            )
                        if (
                            matched_is_missing
                            and type_value != AddressResultType.STREET_ONLY
                        ):
                            msg = (
                                f"[GEO] Cached (success): {addr_key} "
                                f"type={type_value.value} "
                                f"score={score_value} "
                                f"articles={article_ids}"
                                if type_value == AddressResultType.UNRECOGNIZED_PLACE
                                else
                                f"[GEO] Cached (success): {addr_key} "
                                f"raw_json={raw_json_value} "
                                f"type={type_value.value} "
                                f"score={score_value} "
                                f"articles={article_ids}"
                            )
                            return (
                                debug_run
                                ^ put_line(
                                    color_for_score(
                                        msg,
                                        article_ids,
                                        type_value,
                                    )
                                )
                                ^ pure(None)
                            )
                        return (
                            debug_run
                            ^ put_line(
                                color_for_score(
                                    f"[GEO] Cached (success): {addr_key} "
                                    f"matched={matched_address} "
                                    f"type={type_value.value} "
                                    f"score={score_value}",
                                    article_ids,
                                    type_value,
                                )
                            )
                            ^ pure(None)
                        )

                    if cached_type_value:
                        return proceed(AddressResultType(str(cached_type_value)))

                    computed_type = cache_result_type(raw_json_value, addr_key)
                    return (
                        sql_exec(
                            SQL(
                                "UPDATE mar_cache SET address_type = ? WHERE input_address = ?;"
                            ),
                            SQLParams((String(computed_type.value), String(addr_key))),
                        )
                        ^ proceed(computed_type)
                    )

                return (
                    sql_query(CACHE_GET_SQL, SQLParams((String(addr_key),)))
                    >> handle_cache
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
        ^ sql_exec(ALTER_CACHE_SQL)
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
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS address_type TEXT;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            UPDATE victims_cached vc
            SET geo_address_norm = g.geo_address_norm,
                lon = g.lon,
                lat = g.lat,
                address_type = g.address_type
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
