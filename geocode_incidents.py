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
    addr_key_type,
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
  address_type TEXT,
  geo_score DOUBLE
);
"""
)

CREATE_ADDR_MAP_SQL = SQL(
    """--sql
CREATE TABLE IF NOT EXISTS mar_addr_map (
  addr_raw TEXT PRIMARY KEY,
  addr_key TEXT
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
ALTER TABLE mar_cache ADD COLUMN IF NOT EXISTS geo_score DOUBLE;
"""
)

# Insert cache row
INSERT_CACHE_SQL = SQL(
    """--sql
INSERT OR REPLACE INTO mar_cache (
  input_address, matched_address, x_lon, y_lat, raw_json, address_type, geo_score
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""
)

INSERT_ADDR_MAP_SQL = SQL(
    """--sql
INSERT OR REPLACE INTO mar_addr_map (addr_raw, addr_key) VALUES (?, ?)
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
  mc.geo_score AS geo_score,
  CASE
    WHEN COALESCE(mc.address_type, 'NO_SUCCESS') IN ('INTERSECTION','NO_SUCCESS_INTERSECTION','NO_RESULT_INTERSECTION') THEN
      list_extract(
        regexp_split_to_array(
          COALESCE(NULLIF(mc.matched_address,''), mc.input_address, trim(coalesce(i.location_raw,''))),
          '[[:space:]]*(&|AND|AT)[[:space:]]*'
        ),
        1
      )
    WHEN COALESCE(mc.address_type, 'NO_SUCCESS') IN ('BLOCK','NO_SUCCESS_BLOCK','NO_RESULT_BLOCK') THEN
      regexp_replace(
        regexp_replace(
          COALESCE(NULLIF(mc.matched_address,''), mc.input_address, trim(coalesce(i.location_raw,''))),
          '[[:space:]]+',
          ' '
        ),
        '^[[:space:]]*[0-9]+[[:space:]]+(BLOCK|BLK)[[:space:]]+OF[[:space:]]+',
        ''
      )
    WHEN COALESCE(mc.address_type, 'NO_SUCCESS') IN ('ADDRESS','NO_SUCCESS_ADDRESS','NO_RESULT_ADDRESS')
      AND regexp_matches(
        COALESCE(NULLIF(mc.matched_address,''), mc.input_address, trim(coalesce(i.location_raw,''))),
        '^[0-9]+[[:space:]]+'
      ) THEN
      regexp_replace(
        COALESCE(NULLIF(mc.matched_address,''), mc.input_address, trim(coalesce(i.location_raw,''))),
        '^[0-9]+[[:space:]]+',
        ''
      )
    ELSE COALESCE(NULLIF(mc.matched_address,''), mc.input_address, trim(coalesce(i.location_raw,'')))
  END AS geo_address_short,
  CASE
    WHEN COALESCE(mc.address_type, 'NO_SUCCESS') IN ('INTERSECTION','NO_SUCCESS_INTERSECTION','NO_RESULT_INTERSECTION') THEN
      list_extract(
        regexp_split_to_array(
          COALESCE(NULLIF(mc.matched_address,''), mc.input_address, trim(coalesce(i.location_raw,''))),
          '[[:space:]]*(&|AND|AT)[[:space:]]*'
        ),
        2
      )
    ELSE NULL
  END AS geo_address_short_2,
  COALESCE(NULLIF(mc.matched_address,''), 
    mc.input_address, trim(coalesce(i.location_raw,''))) AS geo_address_norm
FROM stg_article_incidents i
LEFT JOIN mar_addr_map m
  ON trim(coalesce(i.location_raw,'')) = m.addr_raw
LEFT JOIN mar_cache mc
  ON mc.input_address = m.addr_key;
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
        dc_suffix_re = re.compile(
            r"(?i)\s*,?\s*(washington(,)?\s*d\.?c\.?|dc|washington)\s*$"
        )

        block_of_re = re.compile(r"\bBLOCK\b(?!\s+OF\b)")

        def normalize_for_mar(a: str) -> str:
            normalized = dc_suffix_re.sub("", (a or "").strip()).upper()
            normalized = normalized.replace(".", "")
            normalized = re.sub(r"^([0-9]+)(?:-?A)\b", r"\1", normalized)
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
            def add_ordinal_suffix(m: re.Match) -> str:
                n = int(m.group(1))
                if 10 <= (n % 100) <= 20:
                    suffix = "TH"
                else:
                    suffix = {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
                return f"{n}{suffix} STREET"

            normalized = re.sub(r"\b([0-9]+)\s+STREET\b", add_ordinal_suffix, normalized)
            normalized = re.sub(r"\bGOOD\s+HOPE\s+ROAD\b", "MARION BARRY AVENUE", normalized)
            normalized = re.sub(
                r"\bCAPIT[AO]L\s+HILTON\s+HOTEL\b", "CAPITAL HILTON", normalized
            )
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
            addr_key: str,
        ) -> str:
            if result_type == AddressResultType.ADDRESS:
                if addr_key_type(addr_key) != AddressResultType.ADDRESS:
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
            def upsert_addr_map() -> Run[None]:
                return sql_exec(
                    INSERT_ADDR_MAP_SQL,
                    SQLParams((String(addr_raw), String(addr_key))),
                ) ^ pure(None)
            if addr_key.lower() == "unknown":
                # Special case: set lon/lat to null
                return upsert_addr_map() ^ (
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
                                None,
                            )
                        ),
                    )
                    ^ pure(None)
                )
            else:
                def handle_mar(g: GeocodeResult) -> Run[None]:
                    address_type = mar_result_type_with_input(addr_key, g.raw_json).value
                    geo_score = mar_result_score(g.raw_json)
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
                                        geo_score,
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
                                        geo_score,
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
                        upsert_addr_map()
                        ^ put_line(
                            f"[GEO] MAR request: {addr_key} articles={article_ids}"
                        )
                        ^ geocode_address(addr_key, env["mar_key"])
                        >> handle_mar
                    )

                def handle_cache(rs: Array) -> Run[None]:
                    if len(rs) == 0:
                        return upsert_addr_map() ^ request_mar()
                    debug_run = (
                        put_line(
                            "[GEO][DEBUG] block normalization: "
                            f"raw={repr(addr_raw)} -> normalized={repr(addr_key)}"
                        )
                        if "BLOCK" in addr_key and "BLOCK OF" not in addr_key
                        else pure(None)
                    )
                    if rs[0]["x_lon"] is None:
                        raw_json_value = rs[0].get("raw_json")
                        cached_type_value = rs[0].get("address_type")
                        cached_score_value = rs[0].get("geo_score")
                        score_value = cache_result_score(raw_json_value)
                        type_value = (
                            AddressResultType(str(cached_type_value))
                            if cached_type_value
                            else cache_result_type(raw_json_value, addr_key)
                        )
                        return (
                            (
                                sql_exec(
                                    SQL(
                                        "UPDATE mar_cache SET address_type = ?, geo_score = ? WHERE input_address = ?;"
                                    ),
                                    SQLParams((String(type_value.value), score_value, String(addr_key))),
                                )
                                if not cached_type_value or cached_score_value is None
                                else pure(unit)
                            )
                            ^ debug_run
                            ^ put_line(
                                f"[GEO] Already cached (permanent failure): {addr_key} "
                                f"articles={article_ids} "
                                f"type={type_value.value}"
                            )
                            ^ pure(None)
                        )
                    matched_address = rs[0].get("matched_address")
                    raw_json_value = rs[0].get("raw_json")
                    cached_type_value = rs[0].get("address_type")
                    cached_score_value = rs[0].get("geo_score")
                    score_value = cache_result_score(raw_json_value)
                    matched_is_missing = matched_address is None or (
                        isinstance(matched_address, str)
                        and matched_address.strip().lower() == "none"
                    )
                    def proceed(type_value: AddressResultType) -> Run[None]:
                        if (
                            type_value == AddressResultType.ADDRESS
                            and addr_key_type(addr_key)
                            == AddressResultType.UNRECOGNIZED_PLACE
                        ):
                            return (
                                debug_run
                                ^ put_line(
                                    color_for_score(
                                        "[GEO] Cache refresh for APPROXIMATE_PLACE: "
                                        f"{addr_key} type={type_value.value} "
                                        f"score={score_value} articles={article_ids}",
                                        article_ids,
                                        type_value,
                                        addr_key,
                                    )
                                )
                                ^ request_mar()
                            )
                        if type_value in (
                            AddressResultType.STREET_ONLY,
                            AddressResultType.UNRECOGNIZED_PLACE,
                        ) and (rs[0].get("x_lon") in (0, None) or rs[0].get("y_lat") in (0, None)):
                            return (
                                debug_run
                                ^ put_line(
                                    "[GEO] Cache refresh for missing coords: "
                                    f"{addr_key} type={type_value.value} "
                                    f"score={score_value} articles={article_ids}"
                                )
                                ^ request_mar()
                            )
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
                                            addr_key,
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
                                        addr_key,
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
                                        addr_key,
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
                                        addr_key,
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
                                addr_key,
                            )
                            )
                            ^ pure(None)
                        )

                    if cached_type_value:
                        return (
                            sql_exec(
                                SQL(
                                    "UPDATE mar_cache SET geo_score = ? WHERE input_address = ?;"
                                ),
                                SQLParams((score_value, String(addr_key))),
                            )
                            if cached_score_value is None
                            else pure(unit)
                        ) ^ proceed(AddressResultType(str(cached_type_value)))

                    computed_type = cache_result_type(raw_json_value, addr_key)
                    return (
                        sql_exec(
                            SQL(
                                "UPDATE mar_cache SET address_type = ?, geo_score = ? WHERE input_address = ?;"
                            ),
                            SQLParams((String(computed_type.value), score_value, String(addr_key))),
                        )
                        ^ proceed(computed_type)
                    )

                return (
                    upsert_addr_map()
                    ^ sql_query(CACHE_GET_SQL, SQLParams((String(addr_key),)))
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
        ^ sql_exec(CREATE_ADDR_MAP_SQL)
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
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS geo_score DOUBLE;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS geo_address_short TEXT;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS geo_address_short_2 TEXT;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            UPDATE victims_cached vc
            SET geo_address_norm = g.geo_address_norm,
                lon = CASE WHEN g.lon = 0 THEN NULL ELSE g.lon END,
                lat = CASE WHEN g.lat = 0 THEN NULL ELSE g.lat END,
                address_type = g.address_type,
                geo_score = g.geo_score,
                geo_address_short = g.geo_address_short,
                geo_address_short_2 = g.geo_address_short_2
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
