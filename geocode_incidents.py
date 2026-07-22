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
    geocode_arcgis_address,
    geocode_nominatim_address,
    GeocodeResult,
    String,
    process_items,
    FailureDetail,
    FailureDetails,
    AddressResultType,
    addr_key_type,
    addr_key_type_without_comma_suffix,
    mar_result_type_with_input,
    mar_result_score,
    arcgis_result_type_with_input,
    arcgis_result_score,
    is_nominatim_result,
    nominatim_result_type_with_input,
    nominatim_result_score,
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

ADDRESS_NORMALIZATION_VERSION = "v2"

GENERAL_STREET_TYPES: dict[str, str] = {
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

GENERAL_ORDINAL_WORDS: dict[str, str] = {
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

GENERAL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<=\s)ST\b"), "STREET"),
)

SPECIFIC_OVERRIDE_RULES: dict[str, str] = {
    "GOOD HOPE ROAD": "MARION BARRY AVENUE",
    "CAPITAL HILTON HOTEL": "CAPITAL HILTON",
    "CAPITOL HILTON HOTEL": "CAPITAL HILTON",
    "CONDON TERRACE CIRCLE": "CONDON TERRACE SE AND 8TH STREET SE",
    "600 BLOCK OF CONDON TERRACE SE": "601 BLOCK OF CONDON TERRACE SE",
}

DC_SUFFIX_RE = re.compile(
    r"(?i)\s*,?\s*(washington(,)?\s*d\.?c\.?|dc|washington)\s*$"
)
BLOCK_OF_RE = re.compile(r"\bBLOCK\b(?!\s+OF\b)")
HALF_ADDRESS_RE = re.compile(
    r"\b(\d+)\s+1/2\s+(\w+)\s+"
    r"(AVE|AV|AVENUE|ST|STREET|RD|ROAD|PL|PLACE|PLZ|PLAZA|TERR|TER|"
    r"TERRACE|BLVD|BOULEVARD|PKWAY|PKWY|PARKWAY|HWY|HIGHWAY|DR|DRIVE|"
    r"CT|COURT|LN|LANE|CIR|CIRCLE|WAY|SQ|SQUARE)\b"
)
LEADING_A_RE = re.compile(r"^([0-9]+)(?:-?A)\b")
STREET_ORDINAL_NUM_RE = re.compile(r"\b([0-9]+)\s+STREET\b")
NYT_NEW_YORK_STREET_RE = re.compile(
    r"\bNEW YORK (AVENUE|BOULEVARD)\b", re.IGNORECASE
)
NYT_QUEENS_LOCALITY_RE = re.compile(r"\b(QUEENS|JAMAICA)\b", re.IGNORECASE)
NYT_STATION_RE = re.compile(r"\bSTATION\b", re.IGNORECASE)
NYT_STATION_SUBWAY_BMT_RE = re.compile(
    r"\b(?:SUBWAY|BMT|IRT|IND)\s+(?=STATION\b)", re.IGNORECASE
)
NYT_STATION_DIRECTION_RE = re.compile(r"^(?:EAST|WEST)\s+", re.IGNORECASE)
NYT_STATION_TRAILING_DETAIL_RE = re.compile(
    r"(?<=\bSTATION\b)[^,]*(?=,)", re.IGNORECASE
)


def _add_ordinal_suffix(m: re.Match[str]) -> str:
    n = int(m.group(1))
    if 10 <= (n % 100) <= 20:
        suffix = "TH"
    else:
        suffix = {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
    return f"{n}{suffix} STREET"


def normalize_general(a: str) -> str:
    normalized = DC_SUFFIX_RE.sub("", (a or "").strip()).upper()
    normalized = normalized.replace(".", "")
    normalized = LEADING_A_RE.sub(r"\1", normalized)
    normalized = HALF_ADDRESS_RE.sub(r"\1 \2 \3", normalized)
    for short, full in GENERAL_STREET_TYPES.items():
        normalized = re.sub(rf"\b{short}\b", full, normalized)
    for pattern, replacement in GENERAL_RULES:
        normalized = pattern.sub(replacement, normalized)
    for word, num in GENERAL_ORDINAL_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", num, normalized)
    normalized = STREET_ORDINAL_NUM_RE.sub(_add_ordinal_suffix, normalized)
    normalized = BLOCK_OF_RE.sub("BLOCK OF", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def apply_specific_overrides(canonical: str) -> str:
    return SPECIFIC_OVERRIDE_RULES.get(canonical, canonical)


def normalize_for_mar(a: str) -> str:
    canonical = normalize_general(a)
    return apply_specific_overrides(canonical)


def normalize_for_arcgis(a: str) -> str:
    """Apply NYT-specific historical street-name normalization."""
    normalized = (a or "").strip()
    if (
        NYT_NEW_YORK_STREET_RE.search(normalized)
        and NYT_QUEENS_LOCALITY_RE.search(normalized)
    ):
        return NYT_NEW_YORK_STREET_RE.sub(
            "GUY R BREWER BOULEVARD", normalized
        )
    return normalized


def normalize_for_nominatim(a: str) -> str:
    """Apply station-specific normalization for Nominatim requests."""
    normalized = normalize_for_arcgis(a)
    while NYT_STATION_SUBWAY_BMT_RE.search(normalized):
        normalized = NYT_STATION_SUBWAY_BMT_RE.sub("", normalized)
    normalized = NYT_STATION_DIRECTION_RE.sub("", normalized)
    return NYT_STATION_TRAILING_DETAIL_RE.sub("", normalized)

CREATE_CACHE_SQL = SQL(
    """--sql
CREATE TABLE IF NOT EXISTS mar_cache (
  input_address TEXT PRIMARY KEY,
  matched_address TEXT,
  x_lon DOUBLE,
  y_lat DOUBLE,
  raw_json TEXT,
  address_type TEXT,
  geo_score DOUBLE,
  review_code TEXT
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
ALTER TABLE mar_cache ADD COLUMN IF NOT EXISTS review_code TEXT;
"""
)

# Insert cache row
INSERT_CACHE_SQL = SQL(
    """--sql
INSERT OR REPLACE INTO mar_cache (
  input_address, matched_address, x_lon, y_lat, raw_json, address_type,
  geo_score, review_code
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
  CAST(NULL AS TEXT) AS geo_address_review,
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

CREATE_ARCGIS_GEOCODED_VIEW_SQL = SQL(
    r"""
CREATE OR REPLACE VIEW stg_article_incidents_geo AS
WITH source AS (
  SELECT
    i.*,
    mc.input_address,
    mc.matched_address,
    mc.x_lon AS lon,
    mc.y_lat AS lat,
    COALESCE(mc.address_type, 'NO_SUCCESS') AS address_type,
    mc.geo_score AS geo_score,
    NULLIF(mc.review_code, '') AS geo_address_review,
    COALESCE(NULLIF(mc.matched_address, ''), mc.input_address,
      trim(coalesce(i.location_raw, ''))) AS geo_source_address
  FROM stg_article_incidents i
  LEFT JOIN arcgis_addr_map m
    ON trim(coalesce(i.location_raw, '')) = m.addr_raw
  LEFT JOIN arcgis_cache mc
    ON mc.input_address = m.addr_key
), components AS (
  SELECT
    source.*,
    CASE
      WHEN address_type IN ('INTERSECTION', 'NO_SUCCESS_INTERSECTION',
                            'NO_RESULT_INTERSECTION') THEN
        regexp_replace(
          list_extract(
            regexp_split_to_array(
              geo_source_address,
              '[[:space:]]*(&|AND|AT)[[:space:]]*'
            ),
            1
          ),
          ',.*$',
          ''
        )
      WHEN address_type IN ('BLOCK', 'NO_SUCCESS_BLOCK', 'NO_RESULT_BLOCK') THEN
        regexp_replace(
          regexp_replace(
            geo_source_address,
            '^ *[0-9]+(-[0-9]+)? +(BLOCK|BLK) +OF +',
            ''
          ),
          ',.*$',
          ''
        )
      WHEN address_type IN ('ADDRESS', 'NO_SUCCESS_ADDRESS',
                            'NO_RESULT_ADDRESS') THEN
        regexp_replace(
          regexp_replace(
            geo_source_address,
            '^[[:space:]]*[0-9]+(-[0-9]+)?[[:space:]]+',
            ''
          ),
          ',.*$',
          ''
        )
      WHEN address_type IN ('STREET_ONLY', 'NO_SUCCESS_STREET_ONLY',
                            'NO_RESULT_STREET_ONLY') THEN
        regexp_replace(geo_source_address, ',.*$', '')
      ELSE geo_source_address
    END AS geo_address_short,
    CASE
      WHEN address_type IN ('INTERSECTION', 'NO_SUCCESS_INTERSECTION',
                            'NO_RESULT_INTERSECTION') THEN
        regexp_replace(
          list_extract(
            regexp_split_to_array(
              geo_source_address,
              '[[:space:]]*(&|AND|AT)[[:space:]]*'
            ),
            2
          ),
          ',.*$',
          ''
        )
      ELSE NULL
    END AS geo_address_short_2,
    regexp_extract(
      geo_source_address,
      ',[[:space:]]*(.*)$',
      1
    ) AS geo_address_locality
  FROM source
)
SELECT
  components.*,
  geo_source_address AS geo_address_norm
FROM components;
"""
)


def _leading_house_number(address: str) -> str | None:
    """Return a leading house number, excluding ordinal street names."""
    ordinal = re.match(r"^[0-9]+(ST|ND|RD|TH)\b", address.upper())
    if ordinal is not None:
        return None
    house_number = re.match(r"^([0-9]+(?:-[0-9]+)?)\b", address)
    return house_number.group(1) if house_number is not None else None


def _arcgis_review_code(
    input_address: object,
    matched_address: object,
    address_type: object,
) -> str | None:
    """Return a structural ArcGIS review code without changing its type."""
    input_text = str(input_address or "").strip()
    matched_text = str(matched_address or "").strip()
    if not matched_text:
        return None
    input_number = _leading_house_number(input_text)
    matched_number = _leading_house_number(matched_text)
    if (
        input_number
        and matched_number
        and input_number != matched_number
    ):
        return "house_number_changed"
    if input_number and str(address_type) == AddressResultType.ADDRESS.value:
        if matched_number is None:
            return "house_number_missing"
    if input_number and str(address_type) in {
        AddressResultType.STREET_ONLY.value,
        AddressResultType.INTERSECTION.value,
        AddressResultType.NAMED_PLACE.value,
    }:
        return "input_type_changed"
    return None


def _arcgis_review_log(
    review: tuple[object, object, object, object, object, object, object, object],
) -> Run[Unit]:
    """Log an ArcGIS structural discrepancy in a copyable form."""
    (
        input_address,
        matched_address,
        address_type,
        score,
        latitude,
        longitude,
        article_ids,
        code,
    ) = review
    if not code:
        return pure(unit)
    if (
        latitude is None
        or longitude is None
        or float(str(latitude or 0)) == 0
        or float(str(longitude or 0)) == 0
    ):
        coordinates = "unavailable"
    else:
        coordinates = (
            f"{float(str(latitude)):.5f}, {float(str(longitude)):.5f}"
        )
    message = (
        "\033[31m[GEO] ArcGIS review: "
        f"code={code} input={input_address!r} matched={matched_address!r} "
        f"type={address_type} score={score} articles={article_ids} "
        f"coordinates={coordinates}\033[0m"
    )
    return put_line(message) ^ pure(unit)


def geocode_all_incident_addresses(env: Environment) -> Run[NextStep]:
    """
    - Ensures cache table exists
    - Pulls distinct addresses
    - For each address, if missing in cache: geocode via MAR -> insert
    - Builds view stg_article_incidents_geo
    """

    profile = env["publication_profile"]
    provider = str(profile.geocoder_provider)
    cache_table = profile.policies.geocoder_cache_tables.cache
    address_map_table = profile.policies.geocoder_cache_tables.address_map

    def provider_sql(template: SQL) -> SQL:
        return SQL(
            str(template)
            .replace("mar_cache", cache_table)
            .replace("mar_addr_map", address_map_table)
            .replace("arcgis_cache", cache_table)
            .replace("arcgis_addr_map", address_map_table)
        )

    create_cache_sql = provider_sql(CREATE_CACHE_SQL)
    create_addr_map_sql = provider_sql(CREATE_ADDR_MAP_SQL)
    alter_cache_sql = provider_sql(ALTER_CACHE_SQL)
    cache_get_sql = provider_sql(CACHE_GET_SQL)
    insert_cache_sql = provider_sql(INSERT_CACHE_SQL)
    insert_addr_map_sql = provider_sql(INSERT_ADDR_MAP_SQL)
    create_view_sql = provider_sql(
        CREATE_ARCGIS_GEOCODED_VIEW_SQL
        if provider == "stanford_arcgis"
        else CREATE_GEOCODED_VIEW_SQL
    )

    def _process_rows(rows: Array[dict]) -> Run[None]:
        """
        Geocode all addresses using applicative traversal:
        - validators: empty (no pre-checks)
        - happy path:
            * check cache
            * if missing: call MAR; if not ok -> validation failure; if ok -> cache
        - accumulate failures but keep processing others
        """
        def cache_result_type(
            raw_json_value: object, addr_key: str
        ) -> AddressResultType:
            if provider == "stanford_arcgis":
                try:
                    parsed = json.loads(str(raw_json_value or "{}"))
                    if is_nominatim_result(parsed):
                        return nominatim_result_type_with_input(addr_key, parsed)
                    return arcgis_result_type_with_input(addr_key, parsed)
                except (json.JSONDecodeError, TypeError):
                    return AddressResultType.NO_RESULT
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
            if provider == "stanford_arcgis":
                try:
                    parsed = json.loads(str(raw_json_value or "{}"))
                    if is_nominatim_result(parsed):
                        return nominatim_result_score(parsed)
                    return arcgis_result_score(parsed)
                except (json.JSONDecodeError, TypeError):
                    return 0
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
                if (
                    addr_key_type_without_comma_suffix(addr_key)
                    if provider == "stanford_arcgis"
                    else addr_key_type(addr_key)
                ) != AddressResultType.ADDRESS:
                    labeled = msg if "type=" in msg else f"{msg} type={result_type.value}"
                    colored = f"\x1b[31m{labeled}\x1b[0m"
                    if "articles=" in msg:
                        return colored
                    return f"{colored} articles={article_ids}"
            return msg

        def cached_result_is_nominatim(raw_json_value: object) -> bool:
            try:
                return is_nominatim_result(
                    json.loads(str(raw_json_value or "{}"))
                )
            except (json.JSONDecodeError, TypeError):
                return False

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
                (
                    normalize_for_mar(cast(str, r["addr_raw"]))
                    if provider == "mar"
                    else (
                        normalize_for_nominatim(cast(str, r["addr_raw"]))
                        if NYT_STATION_RE.search(cast(str, r["addr_raw"]))
                        else normalize_for_arcgis(cast(str, r["addr_raw"]))
                    )
                ),
                cast(str, r.get("article_ids") or ""),
                cast(str, r["addr_raw"]),
            )
        ) & rows

        # Optional, descriptive failure enum (type annotation only)
        class GeoFailureType(FailureType):
            """Enum of failure types for geocoding"""

            GEOCODE_FAILED = "GEOCODE_FAILED"
            UNCAUGHT_EXCEPTION = "UNCAUGHT_EXCEPTION"

        # Render a thrown ErrorPayload (from happy path) into FailureDetails
        def render(err: ErrorPayload) -> FailureDetails[tuple[str, str, str]]:
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
            use_nominatim = (
                provider == "stanford_arcgis"
                and NYT_STATION_RE.search(addr_raw) is not None
            )
            def upsert_addr_map() -> Run[None]:
                return sql_exec(
                    insert_addr_map_sql,
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
                        insert_cache_sql,
                        SQLParams(
                            (
                                String(addr_key),
                                String(""),
                                None,  # x_lon null
                                None,  # y_lat null
                                String(json.dumps({})),
                                String(AddressResultType.NO_SUCCESS.value),
                                None,
                                None,
                            )
                        ),
                    )
                    ^ pure(None)
                )
            else:
                def handle_mar(g: GeocodeResult) -> Run[None]:
                    address_type = (
                        mar_result_type_with_input(addr_key, g.raw_json).value
                        if provider == "mar"
                        else nominatim_result_type_with_input(
                            addr_key, g.raw_json
                        ).value
                        if use_nominatim
                        else arcgis_result_type_with_input(
                            addr_key, g.raw_json
                        ).value
                    )
                    geo_score = (
                        mar_result_score(g.raw_json)
                        if provider == "mar"
                        else nominatim_result_score(g.raw_json)
                        if use_nominatim
                        else arcgis_result_score(g.raw_json)
                    )
                    review_code = (
                        _arcgis_review_code(addr_key, g.matched_address, address_type)
                        if provider == "stanford_arcgis" and not use_nominatim
                        else None
                    )
                    if g.ok:
                        return (
                            sql_exec(
                                insert_cache_sql,
                                SQLParams(
                                    (
                                        String(addr_key),
                                        String(g.matched_address),
                                        g.x_lon,
                                        g.y_lat,
                                        String(json.dumps(g.raw_json)),
                                        String(address_type),
                                        geo_score,
                                        String(review_code or "")
                                        if provider == "stanford_arcgis"
                                        and not use_nominatim
                                        else None,
                                    )
                                ),
                            )
                            ^ put_line(
                                f"[GEO] {provider} response OK, matched address:"
                                f" {g.matched_address} articles={article_ids}"
                            )
                            ^ (
                                _arcgis_review_log(
                                    (
                                        addr_key,
                                        g.matched_address,
                                        address_type,
                                        geo_score,
                                        g.y_lat,
                                        g.x_lon,
                                        article_ids,
                                        review_code,
                                    )
                                )
                                if provider == "stanford_arcgis" and not use_nominatim
                                else pure(unit)
                            )
                            ^ pure(None)
                        )
                    if g.raw_json.get("message") in {
                        "No Result present",
                        "No railway result present",
                    }:
                        return (
                            sql_exec(
                                insert_cache_sql,
                                SQLParams(
                                    (
                                        String(addr_key),
                                        String(""),
                                        None,  # x_lon null
                                        None,  # y_lat null
                                        String(json.dumps(g.raw_json)),
                                        String(address_type),
                                        geo_score,
                                        String(review_code or "")
                                        if provider == "stanford_arcgis"
                                        and not use_nominatim
                                        else None,
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
                            f"[GEO] {provider} transient failure for"
                            f" {addr_key} articles={article_ids}: {g}"
                        )
                        ^ throw(
                            ErrorPayload(
                                String(f"Error during match for '{addr_key}'"),
                                app=g.raw_json.get("message", "no details"),
                            )
                        )
                    )

                def request_provider() -> Run[None]:
                    return (
                        upsert_addr_map()
                        ^ put_line(
                            f"[GEO] {provider} request: {addr_key} articles={article_ids}"
                        )
                        ^ (
                            geocode_address(addr_key, env["mar_key"])
                            if provider == "mar"
                            else (
                                geocode_nominatim_address(addr_key)
                                if use_nominatim
                                else geocode_arcgis_address(addr_key)
                            )
                        )
                        >> handle_mar
                    )

                def handle_cache(rs: Array) -> Run[None]:
                    if len(rs) == 0:
                        return upsert_addr_map() ^ request_provider()
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
                        if use_nominatim and not cached_result_is_nominatim(
                            raw_json_value
                        ):
                            return (
                                debug_run
                                ^ put_line(
                                    "[GEO] Cache refresh for station via "
                                    f"Nominatim: {addr_key} articles={article_ids}"
                                )
                                ^ request_provider()
                            )
                        cached_type_value = rs[0].get("address_type")
                        cached_score_value = rs[0].get("geo_score")
                        cached_review_code = rs[0].get("review_code")
                        score_value = cache_result_score(raw_json_value)
                        type_value = (
                            AddressResultType(str(cached_type_value))
                            if cached_type_value
                            else cache_result_type(raw_json_value, addr_key)
                        )
                        review_code = (
                            cached_review_code
                            if cached_review_code is not None
                            else _arcgis_review_code(
                                addr_key,
                                "",
                                type_value.value,
                            )
                        )
                        return (
                            (
                                sql_exec(
                                    SQL(
                                        f"UPDATE {cache_table} SET address_type = ?, "
                                        "geo_score = ?, review_code = ? "
                                        "WHERE input_address = ?;"
                                    ),
                                    SQLParams(
                                        (
                                            String(type_value.value),
                                            score_value,
                                            String(review_code or ""),
                                            String(addr_key),
                                        )
                                    ),
                                )
                                if (
                                    not cached_type_value
                                    or cached_score_value is None
                                    or cached_review_code is None
                                )
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
                    cached_review_code = rs[0].get("review_code")
                    score_value = cache_result_score(raw_json_value)
                    matched_is_missing = matched_address is None or (
                        isinstance(matched_address, str)
                        and matched_address.strip().lower() == "none"
                    )
                    if use_nominatim and not cached_result_is_nominatim(
                        raw_json_value
                    ):
                        return (
                            debug_run
                            ^ put_line(
                                "[GEO] Cache refresh for station via "
                                f"Nominatim: {addr_key} articles={article_ids}"
                            )
                            ^ request_provider()
                        )
                    # The existing WP decision tree intentionally has many exits.
                    # NYT returns before entering those retry branches.
                    # pylint: disable=too-many-return-statements
                    def proceed(type_value: AddressResultType) -> Run[None]:
                        if provider == "stanford_arcgis" and not use_nominatim:
                            review_code = (
                                cached_review_code
                                if cached_review_code is not None
                                else _arcgis_review_code(
                                    addr_key,
                                    matched_address,
                                    type_value.value,
                                )
                            )
                            return (
                                (
                                    sql_exec(
                                        SQL(
                                            f"UPDATE {cache_table} SET "
                                            "review_code = ? "
                                            "WHERE input_address = ?;"
                                        ),
                                        SQLParams(
                                            (
                                                String(review_code or ""),
                                                String(addr_key),
                                            )
                                        ),
                                    )
                                    if cached_review_code is None
                                    else pure(unit)
                                )
                                ^ debug_run
                                ^ _arcgis_review_log(
                                    (
                                        addr_key,
                                        matched_address,
                                        type_value.value,
                                        score_value,
                                        rs[0].get("y_lat"),
                                        rs[0].get("x_lon"),
                                        article_ids,
                                        review_code,
                                    )
                                )
                                ^ put_line(
                                    f"[GEO] Cached (success): {addr_key} "
                                    f"matched={matched_address} "
                                    f"type={type_value.value} "
                                    f"score={score_value}"
                                )
                                ^ pure(None)
                            )
                        if (
                            type_value == AddressResultType.ADDRESS
                            and (
                                addr_key_type_without_comma_suffix(addr_key)
                                if provider == "stanford_arcgis"
                                else addr_key_type(addr_key)
                            )
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
                                ^ request_provider()
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
                                ^ request_provider()
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
                                    ^ request_provider()
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
                                ^ request_provider()
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
                                ^ request_provider()
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
                                    f"UPDATE {cache_table} SET geo_score = ? WHERE input_address = ?;"
                                ),
                                SQLParams((score_value, String(addr_key))),
                            )
                            if cached_score_value is None
                            else pure(unit)
                        ) ^ proceed(
                            cache_result_type(raw_json_value, addr_key)
                            if provider == "stanford_arcgis"
                            else AddressResultType(str(cached_type_value))
                        )

                    computed_type = cache_result_type(raw_json_value, addr_key)
                    return (
                        sql_exec(
                            SQL(
                                f"UPDATE {cache_table} SET address_type = ?, geo_score = ? WHERE input_address = ?;"
                            ),
                            SQLParams((String(computed_type.value), score_value, String(addr_key))),
                        )
                        ^ proceed(computed_type)
                    )

                return (
                    upsert_addr_map()
                    ^ sql_query(cache_get_sql, SQLParams((String(addr_key),)))
                    >> handle_cache
                )

        # Process addresses monadically, accumulating runtime failures.
        return process_items(
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
                    f"[GEO] {result.r.failures.length} address failures accumulated."
                )
                ^ pure(None)
                if result.r.failures.length > 0
                else pure(None)
            )
        )

    return (
        sql_script(SQL(r"""
            LOAD splink_udfs;
        """)) ^
        sql_exec(create_cache_sql)
        ^ sql_exec(create_addr_map_sql)
        ^ sql_exec(alter_cache_sql)
        ^ put_line(
            f"[GEO] Address normalization version: {ADDRESS_NORMALIZATION_VERSION}"
        )
        ^ sql_query(SELECT_ADDRESSES_SQL)
        >> (
            lambda rows: put_line(f"[GEO] Found {len(rows)} distinct raw addresses.")
            ^ _process_rows(Array(tuple(rows)))
        )
        ^ sql_exec(create_view_sql)
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
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS geo_address_locality TEXT;
        """
            )
        )
        ^ sql_exec(
            SQL(
                r"""
            ALTER TABLE victims_cached ADD COLUMN IF NOT EXISTS geo_address_review TEXT;
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
                geo_address_short_2 = g.geo_address_short_2,
                geo_address_locality = g.geo_address_locality,
                geo_address_review = g.geo_address_review
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
