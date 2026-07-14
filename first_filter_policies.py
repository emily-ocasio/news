"""Publication-specific policies for the automatic first filter."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from calculations.calc_core import filter_row, is_good_type, is_in_dc
from nyc_keywords import NYC_REGEX


@dataclass(frozen=True)
class FirstFilterResult:
    """Automatic first-filter result plus positive evidence for review."""

    code: str
    evidence: str


@dataclass(frozen=True)
class FirstFilterPolicy:
    """Pure first-filter behavior registered for one publication profile."""

    location_match: Callable[[Mapping], bool]
    location_name: str


def _wp_location_match(row: Mapping) -> bool:
    """Return whether the existing WP/DC location rule matches."""
    return is_in_dc(row["FullText"])


def _nyt_location_match(row: Mapping) -> bool:
    """Return whether inclusive NYT/NYC location evidence matches."""
    return NYC_REGEX.search(str(row["FullText"] or "")) is not None


def _classify(row: Mapping, policy: FirstFilterPolicy) -> FirstFilterResult:
    """Apply shared first-filter rules with policy-specific location evidence."""
    if not is_good_type(row) or not filter_row(row):
        return FirstFilterResult("N", "not an eligible homicide candidate")
    if not policy.location_match(row):
        return FirstFilterResult(
            "O", f"no {policy.location_name} location evidence"
        )
    return FirstFilterResult("M", f"{policy.location_name} location evidence")


WP_FIRST_FILTER_POLICY = FirstFilterPolicy(_wp_location_match, "Washington, DC")
NYT_FIRST_FILTER_POLICY = FirstFilterPolicy(_nyt_location_match, "New York")


def classify_with_policy(
    row: Mapping, policy: FirstFilterPolicy
) -> FirstFilterResult:
    """Classify one article using the policy stored in its profile."""
    return _classify(row, policy)
