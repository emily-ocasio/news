"""Typed, immutable publication profiles for one application session."""

from dataclasses import dataclass
from datetime import date
from enum import Enum

from pymonad import HashMap, Just, Maybe, Nothing, String
from first_filter_policies import (
    FirstFilterPolicy,
    NYT_FIRST_FILTER_POLICY,
    WP_FIRST_FILTER_POLICY,
)


class PublicationKey(String):
    """Stable command-line and registry key for a publication."""


class PublicationDisplayName(String):
    """Human-readable publication name."""


class TargetLocationKey(String):
    """Stable registry key for a publication's target location."""


class TargetLocationDisplayName(String):
    """Human-readable target-location name."""


class TargetGeographyDescription(String):
    """Human-readable boundary represented by a target location."""


class ProfileDate(String):
    """Validated inclusive date endpoint in ISO format."""

    def __new__(cls, value: str) -> "ProfileDate":
        date.fromisoformat(value)
        return super().__new__(cls, value)


class UnclassifiedDataset(String):
    """Workflow dataset for articles awaiting classification."""


@dataclass(frozen=True)
class RecordIdBase:
    """Publication-specific base used to simplify article ID entry."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("Record ID base must be positive")


class ClassifiedDataset(String):
    """Workflow dataset for classified articles."""


class RawArticleDatabasePath(String):
    """Path to the shared raw-article SQLite database."""


class ActiveDerivedDatabasePath(String):
    """Path used for derived DuckDB data in the current stage."""


class CanonicalDerivedDatabasePath(String):
    """Publication-isolated DuckDB path targeted by Step 4."""


class OutputNamespacePath(String):
    """Publication-isolated output namespace targeted by Step 4."""


class GPTPromptKey(String):
    """Registered GPT prompt key."""


class HostedPromptId(String):
    """Hosted OpenAI prompt identifier."""


class GPTModelName(String):
    """Configured OpenAI model name."""


class ResponseSchemaName(String):
    """Registered structured-response schema name."""


class GeocoderProviderKey(String):
    """Registered geocoder-provider key."""


class ExternalHomicideScope(String):
    """Human-readable external homicide reference-data scope."""


class SplinkProfileKey(String):
    """Registered publication-specific Splink configuration key."""


@dataclass(frozen=True)
class PublicationDatabaseId:
    """Database publication identity, distinct from location identity."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("Publication database ID must be positive")


@dataclass(frozen=True)
class StoredCityId:
    """Legacy stored city identity, distinct from publication identity."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("Stored city ID must be positive")


@dataclass(frozen=True)
class TargetLocation:
    """One fixed target location associated with a publication."""

    key: TargetLocationKey
    display_name: TargetLocationDisplayName
    stored_city_id: StoredCityId
    geography: TargetGeographyDescription


@dataclass(frozen=True)
class ArticleDateScope:
    """Inclusive publication-date scope for source articles."""

    start: ProfileDate
    end: ProfileDate

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Article date scope start must not exceed end")


@dataclass(frozen=True)
class IncidentDateScope:
    """Inclusive incident-date scope for analytical inclusion."""

    start: ProfileDate
    end: ProfileDate

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Incident date scope start must not exceed end")


@dataclass(frozen=True)
class WorkflowDatasets:
    """Publication-specific article workflow dataset names."""

    unclassified: UnclassifiedDataset
    classified: ClassifiedDataset


@dataclass(frozen=True)
class PublicationResources:
    """Resource paths selected for a publication session."""

    raw_article_database: RawArticleDatabasePath
    active_derived_database: ActiveDerivedDatabasePath
    canonical_derived_database: CanonicalDerivedDatabasePath
    output_namespace: OutputNamespacePath


@dataclass(frozen=True)
class GPTConfiguration:
    """Complete registered GPT configuration for one capability."""

    prompt_key: GPTPromptKey
    hosted_prompt_id: HostedPromptId
    model: GPTModelName
    response_schema: ResponseSchemaName


@dataclass(frozen=True)
class PublicationGPTConfigurations:
    """Publication-specific GPT configurations, explicit when unresolved."""

    classification: Maybe[GPTConfiguration]
    extraction: Maybe[GPTConfiguration]


class Availability(Enum):
    """Explicit operational availability state."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PublicationIdentity:
    """Publication identity values independent of target-location identity."""

    key: PublicationKey
    display_name: PublicationDisplayName
    database_id: PublicationDatabaseId


@dataclass(frozen=True)
class PublicationAnalyticalScope:
    """Target location and independent article/incident date scopes."""

    target_location: TargetLocation
    article_date_scope: ArticleDateScope
    incident_date_scope: IncidentDateScope


@dataclass(frozen=True)
class PublicationPolicies:
    """Registered behavioral policies selected by a publication profile."""

    workflow_datasets: WorkflowDatasets
    record_id_base: RecordIdBase
    first_filter_policy: FirstFilterPolicy
    gpt: PublicationGPTConfigurations
    geocoder: Maybe[GeocoderProviderKey]
    external_homicide_scope: ExternalHomicideScope
    splink_profile: SplinkProfileKey


@dataclass(frozen=True)
class PublicationCapabilities:  # pylint: disable=too-many-instance-attributes
    """Required binary availability for every publication pipeline capability."""

    article_selection: Availability
    first_filter: Availability
    gpt_classification: Availability
    incident_extraction: Availability
    incident_staging: Availability
    geocoding: Availability
    named_victim_deduplication: Availability
    orphan_linkage: Availability
    orphan_adjudication: Availability
    shr_linkage: Availability
    finalized_export: Availability


@dataclass(frozen=True)
class PublicationProfile:
    """Immutable source of publication policy for one application session."""

    identity: PublicationIdentity
    analytical_scope: PublicationAnalyticalScope
    policies: PublicationPolicies
    resources: PublicationResources
    session_availability: Availability
    capabilities: PublicationCapabilities

    @property
    def key(self) -> PublicationKey:
        """Return the publication registry key."""
        return self.identity.key

    @property
    def display_name(self) -> PublicationDisplayName:
        """Return the human-readable publication name."""
        return self.identity.display_name

    @property
    def target_location(self) -> TargetLocation:
        """Return the publication's fixed target location."""
        return self.analytical_scope.target_location

    @property
    def session_label(self) -> String:
        """Render the selected publication and target location."""
        return String(
            f"{self.display_name} — {self.target_location.display_name}"
        )


def _capabilities(availability: Availability) -> PublicationCapabilities:
    """Build the complete capability record for a profile."""
    return PublicationCapabilities(
        article_selection=availability,
        first_filter=availability,
        gpt_classification=availability,
        incident_extraction=availability,
        incident_staging=availability,
        geocoding=availability,
        named_victim_deduplication=availability,
        orphan_linkage=availability,
        orphan_adjudication=availability,
        shr_linkage=availability,
        finalized_export=availability,
    )


def _nyt_capabilities() -> PublicationCapabilities:
    """NYT capabilities available through the first-filter stage."""
    return PublicationCapabilities(
        article_selection=Availability.AVAILABLE,
        first_filter=Availability.AVAILABLE,
        gpt_classification=Availability.AVAILABLE,
        incident_extraction=Availability.UNAVAILABLE,
        incident_staging=Availability.UNAVAILABLE,
        geocoding=Availability.UNAVAILABLE,
        named_victim_deduplication=Availability.UNAVAILABLE,
        orphan_linkage=Availability.UNAVAILABLE,
        orphan_adjudication=Availability.UNAVAILABLE,
        shr_linkage=Availability.UNAVAILABLE,
        finalized_export=Availability.UNAVAILABLE,
    )


WP_PROFILE = PublicationProfile(
    identity=PublicationIdentity(
        key=PublicationKey("wp"),
        display_name=PublicationDisplayName("Washington Post"),
        database_id=PublicationDatabaseId(2),
    ),
    analytical_scope=PublicationAnalyticalScope(
        target_location=TargetLocation(
            key=TargetLocationKey("washington_dc"),
            display_name=TargetLocationDisplayName("Washington, DC"),
            stored_city_id=StoredCityId(2),
            geography=TargetGeographyDescription(
                "District of Columbia municipal boundary"
            ),
        ),
        article_date_scope=ArticleDateScope(
            ProfileDate("1977-01-01"), ProfileDate("2000-12-31")
        ),
        incident_date_scope=IncidentDateScope(
            ProfileDate("1977-01-01"), ProfileDate("2000-12-31")
        ),
    ),
    policies=PublicationPolicies(
        workflow_datasets=WorkflowDatasets(
            UnclassifiedDataset("NOCLASS_WP"), ClassifiedDataset("CLASS_WP")
        ),
        record_id_base=RecordIdBase(100_000_000),
        first_filter_policy=WP_FIRST_FILTER_POLICY,
        gpt=PublicationGPTConfigurations(
            classification=Just(
                GPTConfiguration(
                    GPTPromptKey("classify_only_filter_dc"),
                    HostedPromptId(
                        "pmpt_68c8cb74d6e48193afd2925b0ae7c1d60247458288f5c631"
                    ),
                    GPTModelName("gpt-5-nano"),
                    ResponseSchemaName(
                        "WashingtonPostArticleHomicideClassification"
                    ),
                )
            ),
            extraction=Just(
                GPTConfiguration(
                    GPTPromptKey("extract_incidents_dc"),
                    HostedPromptId(
                        "pmpt_68c8d0edb59c8193920e0e6428d01e3a0902d4a752062094"
                    ),
                    GPTModelName("gpt-5-mini"),
                    ResponseSchemaName(
                        "WashingtonPostArticleIncidentExtraction"
                    ),
                ),
            ),
        ),
        geocoder=Just(GeocoderProviderKey("mar")),
        external_homicide_scope=ExternalHomicideScope(
            "SHR records scoped to DC and the WP incident range"
        ),
        splink_profile=SplinkProfileKey("wp_dc"),
    ),
    resources=PublicationResources(
        raw_article_database=RawArticleDatabasePath("newarticles.db"),
        active_derived_database=ActiveDerivedDatabasePath(
            "derived/wp/news.duckdb"
        ),
        canonical_derived_database=CanonicalDerivedDatabasePath(
            "derived/wp/news.duckdb"
        ),
        output_namespace=OutputNamespacePath("out/wp"),
    ),
    session_availability=Availability.AVAILABLE,
    capabilities=_capabilities(Availability.AVAILABLE),
)


NYT_PROFILE = PublicationProfile(
    identity=PublicationIdentity(
        key=PublicationKey("nyt"),
        display_name=PublicationDisplayName("New York Times"),
        database_id=PublicationDatabaseId(3),
    ),
    analytical_scope=PublicationAnalyticalScope(
        target_location=TargetLocation(
            key=TargetLocationKey("new_york_city"),
            display_name=TargetLocationDisplayName("New York City"),
            stored_city_id=StoredCityId(3),
            geography=TargetGeographyDescription(
                "New York City municipal boundary comprising the five boroughs"
            ),
        ),
        article_date_scope=ArticleDateScope(
            ProfileDate("1981-01-01"), ProfileDate("2000-12-31")
        ),
        incident_date_scope=IncidentDateScope(
            ProfileDate("1981-01-01"), ProfileDate("2000-12-31")
        ),
    ),
    policies=PublicationPolicies(
        workflow_datasets=WorkflowDatasets(
            UnclassifiedDataset("NOCLASS_NYT"), ClassifiedDataset("CLASS_NYT")
        ),
        record_id_base=RecordIdBase(200_000_000),
        first_filter_policy=NYT_FIRST_FILTER_POLICY,
        gpt=PublicationGPTConfigurations(
            classification=Just(
                GPTConfiguration(
                    GPTPromptKey("classify_only_filter_nyc"),
                    HostedPromptId(
                        "pmpt_6a56d62d04a88195bc447d3cefc767a40c49debff14a6943"
                    ),
                    GPTModelName("gpt-5-nano"),
                    ResponseSchemaName(
                        "NewYorkTimesArticleHomicideClassification"
                    ),
                )
            ),
            extraction=Nothing,
        ),
        geocoder=Nothing,
        external_homicide_scope=ExternalHomicideScope(
            "SHR records scoped to NYC and the NYT incident range; exact record "
            "selection remains unresolved until Step 17"
        ),
        splink_profile=SplinkProfileKey("nyt_nyc"),
    ),
    resources=PublicationResources(
        raw_article_database=RawArticleDatabasePath("newarticles.db"),
        active_derived_database=ActiveDerivedDatabasePath(
            "derived/nyt/news.duckdb"
        ),
        canonical_derived_database=CanonicalDerivedDatabasePath(
            "derived/nyt/news.duckdb"
        ),
        output_namespace=OutputNamespacePath("out/nyt"),
    ),
    session_availability=Availability.AVAILABLE,
    capabilities=_nyt_capabilities(),
)


PUBLICATION_PROFILES: HashMap[PublicationKey, PublicationProfile] = (
    HashMap[PublicationKey, PublicationProfile]
    .empty()
    .set(WP_PROFILE.key, WP_PROFILE)
    .set(NYT_PROFILE.key, NYT_PROFILE)
)


def resolve_publication_profile(
    key: PublicationKey,
) -> Maybe[PublicationProfile]:
    """Resolve a profile without providing any default publication."""
    profile = PUBLICATION_PROFILES.get(key)
    return Nothing if profile is None else Just(profile)
