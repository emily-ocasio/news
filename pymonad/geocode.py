"""
MAR geocoding intent types and handler.
"""

from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote
import json
import re
import requests


@dataclass(frozen=True)
class GeocodeResult:
    """Result of geocode query"""

    ok: bool
    normalized_input: str
    matched_address: str
    x_lon: float  # MAR returns X/Y; X = lon, Y = lat
    y_lat: float
    raw_json: dict


class AddressResultType(Enum):
    """Result types for MAR geocoding"""

    NO_SUCCESS = "NO_SUCCESS"
    NO_RESULT = "NO_RESULT"
    NO_SUCCESS_ADDRESS = "NO_SUCCESS_ADDRESS"
    NO_SUCCESS_STREET_ONLY = "NO_SUCCESS_STREET_ONLY"
    NO_SUCCESS_BLOCK = "NO_SUCCESS_BLOCK"
    NO_SUCCESS_INTERSECTION = "NO_SUCCESS_INTERSECTION"
    NO_RESULT_ADDRESS = "NO_RESULT_ADDRESS"
    NO_RESULT_STREET_ONLY = "NO_RESULT_STREET_ONLY"
    NO_RESULT_BLOCK = "NO_RESULT_BLOCK"
    NO_RESULT_INTERSECTION = "NO_RESULT_INTERSECTION"
    ADDRESS = "ADDRESS"
    APPROXIMATE_PLACE = "APPROXIMATE_PLACE"
    QUADRANT = "QUADRANT"
    NAMED_PLACE = "NAMED_PLACE"
    UNRECOGNIZED_PLACE = "UNRECOGNIZED_PLACE"
    STREET_ONLY = "STREET_ONLY"
    INTERSECTION = "INTERSECTION"
    BLOCK = "BLOCK"


def addr_key_type(addr_key: str) -> AddressResultType:
    upper = (addr_key or "").upper()
    street_type = r"(STREET|AVENUE|ROAD|PLACE|PLAZA|TERRACE|BOULEVARD|PARKWAY|HIGHWAY|DRIVE|COURT|LANE|CIRCLE|WAY|SQUARE)"
    quadrant = r"(NW|NE|SW|SE)"
    street_addr = rf"^[0-9]+\s+.+\s+{street_type}(?:\s+{quadrant})?$"
    street_only = rf"^.+\s+{street_type}(?:\s+{quadrant})?$"
    block = rf"^[0-9]+\s+(BLOCK|BLK)\s+OF\s+.+\s+{street_type}(?:\s+{quadrant})?$"
    intersection = rf".+\s+{street_type}(?:\s+{quadrant})?\s+(AND|&|AT)\s+.+\s+{street_type}(?:\s+{quadrant})?$"
    if re.match(block, upper):
        return AddressResultType.BLOCK
    if re.match(intersection, upper):
        return AddressResultType.INTERSECTION
    if re.match(street_addr, upper):
        return AddressResultType.ADDRESS
    if re.match(street_only, upper):
        return AddressResultType.STREET_ONLY
    return AddressResultType.UNRECOGNIZED_PLACE


@dataclass(frozen=True)
class MarGeocode:
    """Effect: Geocode a DC address via MAR 2 API (findAddress2)."""

    address: str
    mar_key: str
    # You can add optional params here if needed (e.g., preferScoreMin, etc.)


def mar_result_type(j: dict) -> AddressResultType:
    """Determine the result type from MAR geocode JSON response."""
    success = j.get("Success", False)
    if not success:
        return AddressResultType.NO_SUCCESS
    result = j.get("Result", {})
    if not result:
        return AddressResultType.NO_RESULT
    addresses = result.get("addresses", [])
    if addresses:
        return AddressResultType.ADDRESS
    intersections = result.get("intersections", [])
    if intersections:
        return AddressResultType.INTERSECTION
    blocks = result.get("blocks", [])
    if blocks:
        return AddressResultType.BLOCK
    return AddressResultType.NO_RESULT


def mar_result_type_with_input(addr_key: str, j: dict) -> AddressResultType:
    """Determine result type using input address plus MAR response."""
    base_type = mar_result_type(j)
    if base_type == AddressResultType.NO_SUCCESS or base_type == AddressResultType.NO_RESULT:
        base_guess = addr_key_type(addr_key)
        if base_guess == AddressResultType.BLOCK:
            return AddressResultType.NO_SUCCESS_BLOCK if base_type == AddressResultType.NO_SUCCESS else AddressResultType.NO_RESULT_BLOCK
        if base_guess == AddressResultType.INTERSECTION:
            return AddressResultType.NO_SUCCESS_INTERSECTION if base_type == AddressResultType.NO_SUCCESS else AddressResultType.NO_RESULT_INTERSECTION
        if base_guess == AddressResultType.ADDRESS:
            return AddressResultType.NO_SUCCESS_ADDRESS if base_type == AddressResultType.NO_SUCCESS else AddressResultType.NO_RESULT_ADDRESS
        if base_guess == AddressResultType.STREET_ONLY:
            return AddressResultType.NO_SUCCESS_STREET_ONLY if base_type == AddressResultType.NO_SUCCESS else AddressResultType.NO_RESULT_STREET_ONLY
        if base_guess == AddressResultType.UNRECOGNIZED_PLACE:
            return base_type
        return base_type
    if base_type != AddressResultType.ADDRESS:
        return base_type
    result = j.get("Result", {})
    addresses = result.get("addresses", [])
    if not addresses:
        return base_type
    props = addresses[0].get("address", {}).get("properties", {})
    try:
        score = float(props.get("_Score", 0))
    except (TypeError, ValueError):
        score = 0
    derived = address_result_type_for_score(
        addr_key, score, base_type, props.get("FullAddress")
    )
    if (
        derived == AddressResultType.ADDRESS
        and (addr_key or "").strip().upper()
        in {"NORTHWEST", "NORTHEAST", "SOUTHWEST", "SOUTHEAST"}
    ):
        return AddressResultType.QUADRANT
    if (
        derived == AddressResultType.ADDRESS
        and addr_key_type(addr_key) == AddressResultType.UNRECOGNIZED_PLACE
    ):
        return AddressResultType.APPROXIMATE_PLACE
    if (
        derived == AddressResultType.ADDRESS
        and not (props.get("FullAddress") or "").strip()
    ):
        return AddressResultType.UNRECOGNIZED_PLACE
    return derived


def mar_result_score(j: dict) -> float:
    """Determine the score from MAR geocode JSON response."""
    result_type = mar_result_type(j)
    if result_type in (AddressResultType.NO_SUCCESS, AddressResultType.NO_RESULT):
        return 0
    result = j.get("Result", {})
    match result_type:
        case (
            AddressResultType.ADDRESS
            | AddressResultType.APPROXIMATE_PLACE
            | AddressResultType.QUADRANT
            | AddressResultType.NAMED_PLACE
            | AddressResultType.STREET_ONLY
        ):
            addresses = result.get("addresses", [])
            c0 = addresses[0].get("address", {}).get("properties", {})
        case AddressResultType.INTERSECTION:
            intersections = result.get("intersections", [])
            c0 = intersections[0].get("intersection", {}).get("properties", {})
        case AddressResultType.BLOCK:
            blocks = result.get("blocks", [])
            c0 = blocks[0].get("block", {}).get("properties", {})
        case _:
            return 0
    try:
        return float(c0.get("_Score", 0))
    except (TypeError, ValueError):
        return 0


def address_result_type_for_score(
    addr_key: str,
    score: float,
    base_type: AddressResultType,
    full_address: str | None = None,
) -> AddressResultType:
    """Refine address result type based on score and input address."""
    if base_type != AddressResultType.ADDRESS:
        return base_type
    if not 0 < score:
        return base_type
    if score >= 100 and not (full_address or "").strip():
        return AddressResultType.NAMED_PLACE
    street_types = (
        "STREET",
        "AVENUE",
        "ROAD",
        "PLACE",
        "TERRACE",
        "BOULEVARD",
        "PARKWAY",
        "HIGHWAY",
        "DRIVE",
        "COURT",
        "LANE",
        "CIRCLE",
        "WAY",
        "SQUARE",
        "PLAZA",
    )
    padded = f" {addr_key} "
    has_type = any(f" {stype} " in padded for stype in street_types)
    first_word = addr_key.split()[0] if addr_key.split() else ""
    is_ordinal = re.match(r"^\d+(ST|ND|RD|TH)$", first_word) is not None
    starts_with_digit = first_word[:1].isdigit()
    if not has_type or (starts_with_digit and not is_ordinal):
        return base_type
    if score >= 100:
        return AddressResultType.NAMED_PLACE
    return AddressResultType.STREET_ONLY


def mar_geocode_handler(x: MarGeocode) -> GeocodeResult:
    """
    # MAR 2: https://geocoder.doc.dc.gov/api (findAddress2 endpoint)
    # Simple GET with 'address' and 'f=json'
    # Encode full address as a path segment to avoid "/" being treated as a separator.
    """
    encoded_address = quote(x.address, safe="")
    url = f"https://datagate.dc.gov/mar/open/api/v2.2/locations/{encoded_address}"
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
        result_type = mar_result_type_with_input(x.address, j)
        result = j.get("Result", {})
        addresses = result.get("addresses", [])
        intersections = result.get("intersections", [])
        blocks = result.get("blocks", [])

        match result_type:
            case AddressResultType.NO_SUCCESS | AddressResultType.NO_SUCCESS_ADDRESS | AddressResultType.NO_SUCCESS_STREET_ONLY | AddressResultType.NO_SUCCESS_BLOCK | AddressResultType.NO_SUCCESS_INTERSECTION:
                return GeocodeResult(
                    ok=False,
                    normalized_input=x.address,
                    matched_address="",
                    x_lon=0,
                    y_lat=0,
                    raw_json={"message": j.get("message", "Success= False")},
                )
            case AddressResultType.NO_RESULT | AddressResultType.NO_RESULT_ADDRESS | AddressResultType.NO_RESULT_STREET_ONLY | AddressResultType.NO_RESULT_BLOCK | AddressResultType.NO_RESULT_INTERSECTION:
                if result and not addresses and not intersections and not blocks:
                    message = "No addresses or intersections found"
                else:
                    message = j.get("message", "No Result present")
                return GeocodeResult(
                    ok=False,
                    normalized_input=x.address,
                    matched_address="",
                    x_lon=0,
                    y_lat=0,
                    raw_json={"message": message},
                )
            case AddressResultType.NAMED_PLACE:
                c0 = addresses[0].get("address", {}).get("properties", {})
                return GeocodeResult(
                    ok=True,
                    normalized_input=x.address,
                    matched_address=c0.get("Alias", ""),
                    x_lon=float(c0.get("Longitude", 0)),
                    y_lat=float(c0.get("Latitude", 0)),
                    raw_json=j,
                )
            case AddressResultType.APPROXIMATE_PLACE:
                c0 = addresses[0].get("address", {}).get("properties", {})
                return GeocodeResult(
                    ok=True,
                    normalized_input=x.address,
                    matched_address=x.address,
                    x_lon=float(c0.get("Longitude", 0)),
                    y_lat=float(c0.get("Latitude", 0)),
                    raw_json=j,
                )
            case AddressResultType.QUADRANT:
                c0 = addresses[0].get("address", {}).get("properties", {})
                return GeocodeResult(
                    ok=True,
                    normalized_input=x.address,
                    matched_address=x.address,
                    x_lon=float(c0.get("Longitude", 0)),
                    y_lat=float(c0.get("Latitude", 0)),
                    raw_json=j,
                )
            case AddressResultType.STREET_ONLY:
                c0 = addresses[0].get("address", {}).get("properties", {})
                return GeocodeResult(
                    ok=True,
                    normalized_input=x.address,
                    matched_address="",
                    x_lon=float(c0.get("Longitude", 0)),
                    y_lat=float(c0.get("Latitude", 0)),
                    raw_json=j,
                )
            case AddressResultType.UNRECOGNIZED_PLACE:
                c0 = addresses[0].get("address", {}).get("properties", {})
                return GeocodeResult(
                    ok=True,
                    normalized_input=x.address,
                    matched_address="",
                    x_lon=float(c0.get("Longitude", 0)),
                    y_lat=float(c0.get("Latitude", 0)),
                    raw_json=j,
                )
            case AddressResultType.ADDRESS:
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
            case AddressResultType.INTERSECTION:
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
            case AddressResultType.BLOCK:
                # Take first block candidate
                b0 = blocks[0].get("block", {}).get("properties", {})
                number = b0.get("LowerRange", b0.get("HigherRange", ""))
                street = b0.get("OnStreetDisplay", "")
                full = (
                    f"{number} BLOCK OF {street}"
                    if number and street
                    else b0.get("BlockName", b0.get("FullBlock", ""))
                )
                return GeocodeResult(
                    ok=True,
                    normalized_input=x.address,
                    matched_address=full,
                    x_lon=float(b0.get("Longitude", 0)),
                    y_lat=float(b0.get("Latitude", 0)),
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
