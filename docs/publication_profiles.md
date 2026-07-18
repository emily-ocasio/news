# Publication Profiles

## Purpose and authority

This document records the agreed operational requirements for publications
supported by the news articles analysis application. It is the human-readable
requirements baseline for the typed runtime `PublicationProfile` described in
the [publication generalization refactoring plan](publication_generalization_refactoring_plan.md).

The runtime profile will become the operational source of truth once it is
implemented. Until then, implementation must conform to this document and must
not fill unresolved fields with silent defaults.

## Profile invariants

- Exactly one publication profile is selected at application startup.
- A profile is immutable for the duration of an application session.
- Each publication has exactly one fixed target location.
- Publication identity and target-location identity are separate semantic
  concepts, even when their stored numeric values are equal.
- Publication identity comes from the selected profile and the article's
  `Publication` value, never solely from a workflow dataset label.
- Article publication dates and homicide incident dates have separate scopes.
  Incident-date scope determines whether a homicide is included.
- Date-range endpoints in this document are inclusive.
- Derived data, progress, caches, models, adjudications, and outputs are scoped
  to one publication unless a finalized reporting operation explicitly
  combines canonical outputs.
- A pipeline capability must be explicitly available before it can run. An
  unavailable capability must fail clearly rather than fall back to another
  publication's behavior.

## Agreed configuration matrix

| Characteristic | Washington Post | New York Times |
| --- | --- | --- |
| Profile key | `wp` | `nyt` |
| Display name | Washington Post | New York Times |
| Publication database ID | `2` | `3` |
| Target-location key | `washington_dc` | `new_york_city` |
| Stored `city_id` | `2` | `3` |
| Article-date scope | 1977-01-01 through 2000-12-31 | 1981-01-01 through 2000-12-31 |
| Incident-date scope | 1977-01-01 through 2000-12-31 | 1981-01-01 through 2000-12-31 |
| Unclassified workflow dataset | `NOCLASS_WP` | `NOCLASS_NYT` |
| Classified workflow dataset | `CLASS_WP` | `CLASS_NYT` |
| Simplified record-ID input base | `100000000` | `200000000` |
| First-filter policy | Preserve existing WP behavior | Use the same policy with NYC-specific location evidence |
| Target geography | District of Columbia municipal boundary | New York City's five boroughs |
| Classification GPT capability | Existing WP configuration | Unavailable pending Step 8 |
| Extraction GPT capability | Existing WP configuration | Configured; pending manual validation |
| Geocoder | Existing MAR provider | Stanford ArcGIS Locator Service |
| External homicide reference | SHR records scoped to DC and the WP incident range | SHR records scoped to NYC and the NYT incident range |
| SQLite incident-staging table | `articles_wp_subset` | `articles_nyt_subset` |
| Derived DuckDB namespace | `derived/wp/news.duckdb` | `derived/nyt/news.duckdb` |
| Output namespace | `out/wp/` | `out/nyt/` |

## Identity and legacy `city_id` compatibility

The publication IDs and stored city IDs currently have the same numeric values:

| Publication key | Publication ID | Location key | Stored `city_id` |
| --- | ---: | --- | ---: |
| `wp` | 2 | `washington_dc` | 2 |
| `nyt` | 3 | `new_york_city` | 3 |

This equality is an explicit profile mapping, not a claim that publication and
location are the same concept. Typed application-facing values must preserve
the distinction.

As a transitional compatibility mechanism, SQL may continue to project an
article's `Publication` value into `city_id` while both profiles declare and
validate this one-to-one numeric mapping. Code must not generalize that
coincidence into an architectural requirement.

### Washington Post derived-data preservation

Introducing publication profiles, target-location identities, or
publication-specific paths must not require existing Washington Post DuckDB or
Splink results to be regenerated. This requirement covers existing incident
staging, geocodes, models, trained parameters, predictions, clusters, orphan
results, adjudications, and downstream derived results.

Any transition from the current `news.duckdb` location to
`derived/wp/news.duckdb` must reuse the existing database intact and validate
its compatibility. It must not retrain, rematch, recluster, or otherwise
rebuild existing Washington Post results merely to introduce the new profile
boundary.

## Date policy

- Incident date, not article publication date, determines whether an incident
  is within analytical scope.
- The extraction result must assign at least an incident year. Existing prompt
  logic may resolve uncertainty when the exact month or day is unavailable.
- The assigned incident year determines range inclusion when the exact date is
  uncertain.
- Article-date scope independently controls which source articles belong to a
  publication's processing cohort.

## Homicide and article policy

Both publications use the same article eligibility, homicide evidence, and
excluded-killing policy. Only the target-location evidence differs.

The existing exclusions apply to both publications, including:

- Vehicular killings.
- Killings by law enforcement.
- Fictional or cultural depictions of killings.
- Military or war killings.

The first filter retains its existing workflow meanings:

- `M`: potential in-scope homicide candidate.
- `N`: not an eligible homicide candidate under the shared policy.
- `O`: homicide evidence exists, but the location is outside the target
  geography.

Existing Washington Post first-filter behavior must be preserved. New York
Times filtering uses the same article-type and homicide/exclusion policy with
NYC location evidence replacing DC location evidence.

Washington Post-specific special-case terms, including the existing `Hanafi`
and `Urgo` handling, remain specific to the Washington Post profile and must
not automatically transfer to the New York Times profile.

## Target-location policy

### Washington, DC

An included Washington Post incident must have occurred within the municipal
boundary of the District of Columbia. References to the broader Washington
metropolitan area do not establish inclusion.

### New York City

An included New York Times incident must have occurred within New York City's
municipal boundary, comprising exactly these five boroughs:

- The Bronx.
- Brooklyn (Kings County).
- Manhattan (New York County).
- Queens.
- Staten Island (Richmond County).

New York State outside the city, nearby municipalities, and New Jersey are out
of scope.

A generic reference to "New York" may survive the recall-oriented first filter
when accompanied by homicide evidence. It is not sufficient for final
geographic inclusion. GPT second filtering must specifically determine that
the homicide occurred in New York City before the incident can be accepted as
in scope.

The physical location of the homicide controls. An article dateline or the
location of a court, arrest, police agency, offender residence, or victim
residence does not substitute for the homicide location.

## GPT configuration

### Washington Post classification

- Prompt key: `classify_only_filter_dc`.
- Hosted prompt ID:
  `pmpt_68c8cb74d6e48193afd2925b0ae7c1d60247458288f5c631`.
- Model: `gpt-5-nano`.
- Response schema: `WashingtonPostArticleHomicideClassification`.

### Washington Post extraction

- Prompt key: `extract_incidents_dc`.
- Hosted prompt ID:
  `pmpt_68c8d0edb59c8193920e0e6428d01e3a0902d4a752062094`.
- Model: `gpt-5-mini`.
- Response schema: `WashingtonPostArticleIncidentExtraction`.

These entries describe the existing WP configuration and must remain stable
until a separately agreed change is made.

### New York Times classification

- Prompt key: `classify_only_filter_nyc`.
- Hosted prompt ID:
  `pmpt_6a56d62d04a88195bc447d3cefc767a40c49debff14a6943`.
- Model: `gpt-5-nano`.
- Response schema: `NewYorkTimesArticleHomicideClassification`.
- The schema uses the publication-neutral `ClassificationOutcome` values and
  the shared homicide classification values; database codes remain the
  existing publication-neutral workflow codes.
- Validation is performed manually by the project owner.

### New York Times extraction

- Prompt key: `extract_incidents_nyc`.
- Hosted prompt ID:
  `pmpt_6a594f1449f08190b653de53875d1e040524f9eaba100ec4`.
- Model: `gpt-5-mini`.
- Response schema: `ArticleIncidentExtraction`.
- Validation is performed manually by the project owner.

NYT extraction is available for GPT extraction and the `[F]` controller's `[G]`
action. Because NYT incident staging is still unavailable, `[F]` `[G]` saves
the extraction result but skips the post-extraction derived-data refresh. The
application must not reuse WP refresh or geocoding behavior as an implicit NYT
default.

## Geocoding

- Washington Post/DC geocoding continues to use the existing MAR provider.
- The New York Times/NYC provider is Stanford's ArcGIS Locator Service at
  `https://locator.stanford.edu/arcgis/rest/services/geocode/USA/GeocodeServer/findAddressCandidates`.
- NYT geocoding requests append `, New York` to the extracted location and use
  `forStorage=true`, `outFields=Addr_Type`, and `f=json`.
- Provider-specific cache table names are selected by the active profile:
  WP uses `mar_cache`/`mar_addr_map`; NYT uses
  `arcgis_cache`/`arcgis_addr_map`.
- There is no present requirement to retrofit existing Washington Post results
  to Stanford Locator Service or to replace their MAR-derived geocodes.

## External homicide-data scope

Both profiles will link independently to the Supplementary Homicide Reports
(SHR):

- WP linkage is limited to DC homicides within the WP incident-date scope.
- NYT linkage is limited to NYC homicides within the NYT incident-date scope.

The exact NYC SHR agency, county, and geography-selection rules remain
deliberately unresolved pending the coverage audit in Step 17. They must be
settled and evaluated before NYC external linkage becomes available; a simple
substitution of a state predicate is not an acceptable default.

## Storage and output scope

- Raw source articles remain together in the shared SQLite database
  `newarticles.db` and are distinguished by publication ID.
- SQLite maintains one publication-specific incident-staging table per profile.
  This preserves indexed SQLite selection from the large raw `articles` table
  before DuckDB reads the smaller staging table. The staging tables are
  `articles_wp_subset` and `articles_nyt_subset`; each is rebuilt only by its
  active profile.
- Derived working data uses publication-specific DuckDB namespaces:
  `derived/wp/news.duckdb` and `derived/nyt/news.duckdb`.
- Publication outputs use `out/wp/` and `out/nyt/`.
- Caches, model artifacts, thresholds, predictions, clusters, orphan linkage,
  adjudications, and external-linkage results must not cross those boundaries.
- Publication identity is enforced by the active profile, the profile-specific
  SQLite staging table, and the publication-specific DuckDB namespace. Derived
  tables do not require a separate per-row `publication_id` column.
- A future combined Stata export may read finalized canonical outputs from
  both namespaces, while preserving publication and target-location
  provenance. It must not combine or rerun intermediate processing.

## Capability status policy

The WP profile initially exposes the capabilities already supported by the
application. The NYT profile begins with publication-specific pipeline
capabilities unavailable. Each capability becomes operational only when its
corresponding implementation stage passes its acceptance gate.

Unavailable NYT capabilities currently include:

- Article selection and workflow transitions.
- First filtering.
- GPT classification.
- Incident staging.
- Geocoding.
- Named-victim deduplication.
- Unnamed-victim and orphan linkage.
- Orphan adjudication and post-adjudication processing.
- SHR linkage.
- Finalized canonical export.

Availability must be explicit and auditable. An unavailable NYT operation must
not silently execute WP behavior or access WP resources.

## Deliberately unresolved fields

| Field | Current position | Resolution stage |
| --- | --- | ---: |
| NYT classification prompt, model, and schema | Unavailable; no WP fallback | 8 |
| NYT extraction prompt, model, and schema | Resolved; manual validation pending | 10 |
| NYC geocoding provider | Stanford ArcGIS Locator Service | 12 |
| Exact NYC SHR record-selection rules | Intended scope is all and only NYC homicides | 17 |

These are controlled unresolved requirements. They must remain visible in the
profile and capability checks until explicitly settled.
