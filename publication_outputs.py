"""Publication-specific output routing for application controllers."""

from publication_profiles import PublicationProfile
from pymonad import Environment, Run, SQL, ask, sql_export


def publication_output_filename(
    profile: PublicationProfile, filename: str
) -> str:
    """Prepend the active publication's output namespace to a filename."""
    return f"{profile.resources.output_namespace}/{filename}"


def publication_sql_export(
    sql: SQL,
    filename: str,
    sheet: str | None = None,
    band_by_group_col: str | None = None,
    band_wrap: int = 2,
) -> Run[None]:
    """Export a query under the active publication's output namespace."""
    def export_for_environment(env: Environment) -> Run[None]:
        return sql_export(
            sql,
            publication_output_filename(env["publication_profile"], filename),
            sheet,
            band_by_group_col,
            band_wrap,
        )

    return ask() >> export_for_environment
