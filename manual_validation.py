"""Menu and controller for manual pipeline validation."""
# pylint: disable=too-many-lines

import json
from collections.abc import Mapping
from typing import Any

from calculations import rich_text
from pymonad import (
    Array,
    Environment,
    InputPrompt,
    Run,
    PromptKey,
    String,
    SQL,
    SQLParams,
    ask,
    input_number,
    get_line,
    pure,
    put_line,
    sql_exec,
    sql_query,
)
from menuprompts import MenuChoice, MenuPrompts, NextStep, input_from_menu

type CohortRow = Mapping[str, Any]
VALIDATION_STEP_FIRST_FILTER = String("first_filter")
VALIDATION_STEP_SECOND_FILTER = String("second_filter")
VALIDATION_STEP_EXTRACTION = String("incident_extraction")
COHORT_TYPE_NO_HOMICIDE = String("first_filter_no_homicide")
COHORT_TYPE_OUTSIDE_AREA = String("first_filter_outside_area")
COHORT_TYPE_GPT_NO_HOMICIDE = String("second_filter_no_homicide")
COHORT_TYPE_GPT_OUTSIDE_AREA = String("second_filter_outside_area")
COHORT_TYPE_GPT_EARLY_HOMICIDE = String("second_filter_early_homicide")
COHORT_TYPE_EXTRACTION_NO_INCIDENTS = String("extraction_no_incidents")
COHORT_TYPE_EXTRACTION_COMPLETENESS = String("extraction_completeness")
VALIDATION_MENU = (
    "Create a [N]ew cohort",
    "[V]alidate an incomplete cohort",
    "[R]eport a completed cohort",
    "Return to [M]ain menu",
)
NO_HOMICIDE_SCORE_MENU = (
    "[H]omicide",
    "[N]o homicide",
    "Return to [M]ain menu",
)
OUTSIDE_AREA_SCORE_MENU = (
    "In defined area [I] (saves M)",
    "[O]utside defined area",
    "Return to [M]ain menu",
)


def _cohort_summary_sql() -> str:
    """Return cohort completion counts for the active publication."""
    return """
        SELECT
            c.CohortId,
            c.PipelineStep,
            c.CohortType,
            c.RequestedSize,
            c.CreatedAt,
            COUNT(a.RecordId) AS SampledCount,
            SUM(CASE WHEN a.ManualResult IS NOT NULL THEN 1 ELSE 0 END)
                AS ValidatedCount
        FROM manual_validation_cohorts c
        LEFT JOIN manual_validation_articles a
            ON a.CohortId = c.CohortId
        WHERE c.Publication = ?
        GROUP BY c.CohortId
        ORDER BY c.CreatedAt DESC, c.CohortId DESC
    """


def _has_eligible_articles_sql(cohort_type: String) -> str:
    """Return whether the active publication has eligible cohort articles."""
    if _uses_gpt_class_source(cohort_type):
        extraction_filter = (
            "AND gptVictimJson IS NOT NULL"
            if cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS else ""
        )
        return f"""
        SELECT EXISTS (
            SELECT 1
            FROM articles
            WHERE Publication = ?
              AND Dataset = ?
              AND GPTClass = ?
              {extraction_filter}
              AND length(PubDate) = 8
        ) AS HasEligibleArticles
        """
    return """
        SELECT EXISTS (
            SELECT 1
            FROM articles
            WHERE Publication = ?
              AND AutoClass = ?
              AND length(PubDate) = 8
        ) AS HasEligibleArticles
        """


def _insert_cohort_sql() -> str:
    """Insert validation cohort metadata."""
    return """
        INSERT INTO manual_validation_cohorts
            (Publication, PipelineStep, CohortType, RequestedSize)
        VALUES (?, ?, ?, ?)
    """


def _insert_cohort_articles_sql(cohort_type: String) -> str:
    """Insert an equal-per-year random sample in stable cohort order."""
    if _uses_gpt_class_source(cohort_type):
        article_filter = "AND Dataset = ? AND GPTClass = ?"
        if cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS:
            article_filter += " AND gptVictimJson IS NOT NULL"
    else:
        article_filter = "AND AutoClass = ?"
    output_select = (
        "gptVictimJson"
        if cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS else "NULL"
    )
    original_result_select = (
        "CapturedExtraction"
        if cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS else "?"
    )
    return f"""
        WITH ranked AS (
            SELECT
                RecordId,
                {output_select} AS CapturedExtraction,
                CAST(substr(PubDate, 1, 4) AS INTEGER) AS PublicationYear,
                ROW_NUMBER() OVER (
                    PARTITION BY substr(PubDate, 1, 4)
                    ORDER BY RANDOM()
                ) AS YearRank
            FROM articles
            WHERE Publication = ?
              {article_filter}
              AND length(PubDate) = 8
        ), layered AS (
            SELECT
                RecordId,
                PublicationYear,
                CapturedExtraction,
                ROW_NUMBER() OVER (
                    ORDER BY YearRank, RANDOM()
                ) AS SampleOrder
            FROM ranked
        )
        INSERT INTO manual_validation_articles
            (CohortId, RecordId, SampleOrder, PublicationYear, OriginalAutoClass)
        SELECT ?, RecordId, SampleOrder, PublicationYear,
               {original_result_select}
        FROM layered
        WHERE SampleOrder <= ?
        ORDER BY SampleOrder
    """


def _eligibility_params(
    publication_id: int,
    classified_dataset: String,
    cohort_type: String,
    original_result: String,
) -> tuple[int | String, ...]:
    """Build filter parameters for the selected pipeline step."""
    if _uses_gpt_class_source(cohort_type):
        return (publication_id, classified_dataset, original_result)
    return (publication_id, original_result)


def _sample_params(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    publication_id: int,
    classified_dataset: String,
    cohort_type: String,
    original_result: String,
    cohort_id: int,
    size: int,
) -> tuple[int | String, ...]:
    """Build sampling parameters for the selected pipeline step."""
    filters = _eligibility_params(
        publication_id, classified_dataset, cohort_type, original_result
    )
    if cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS:
        return filters + (cohort_id, size)
    return filters + (cohort_id, original_result, size)


def _step_for_cohort_type(cohort_type: String) -> String:
    """Return the pipeline step represented by a cohort type."""
    if cohort_type in (
        COHORT_TYPE_EXTRACTION_NO_INCIDENTS,
        COHORT_TYPE_EXTRACTION_COMPLETENESS,
    ):
        return VALIDATION_STEP_EXTRACTION
    if cohort_type in (
        COHORT_TYPE_GPT_NO_HOMICIDE,
        COHORT_TYPE_GPT_OUTSIDE_AREA,
        COHORT_TYPE_GPT_EARLY_HOMICIDE,
    ):
        return VALIDATION_STEP_SECOND_FILTER
    return VALIDATION_STEP_FIRST_FILTER


def _uses_gpt_class_source(cohort_type: String) -> bool:
    """Return whether the cohort samples the GPTClass result column."""
    return cohort_type in (
        COHORT_TYPE_GPT_NO_HOMICIDE,
        COHORT_TYPE_GPT_OUTSIDE_AREA,
        COHORT_TYPE_GPT_EARLY_HOMICIDE,
        COHORT_TYPE_EXTRACTION_NO_INCIDENTS,
        COHORT_TYPE_EXTRACTION_COMPLETENESS,
    )


def _last_cohort_id_sql() -> str:
    """Read the ID assigned by the immediately preceding cohort insert."""
    return "SELECT last_insert_rowid() AS CohortId"


def _cohort_size_sql() -> str:
    """Count sampled articles in the newly created cohort."""
    return """
        SELECT COUNT(*) AS SampledCount
        FROM manual_validation_articles
        WHERE CohortId = ?
    """


def _incomplete_cohort_sql() -> str:
    """Check that a requested cohort is incomplete and belongs to the session."""
    return """
        SELECT c.CohortType
        FROM manual_validation_cohorts c
        WHERE c.CohortId = ?
          AND c.Publication = ?
          AND EXISTS (
              SELECT 1 FROM manual_validation_articles a
              WHERE a.CohortId = c.CohortId AND a.ManualResult IS NULL
          )
        LIMIT 1
    """


def _next_unvalidated_article_sql() -> str:
    """Get the next pending review row and any captured extraction output."""
    return """
        SELECT
            a.RecordId,
            a.Title,
            a.PubDate,
            a.FullText,
            a.Publication,
            m.OriginalAutoClass AS ExtractionSnapshot,
            m.SampleOrder AS ProgressNumber,
            (
                SELECT COUNT(*)
                FROM manual_validation_articles total
                WHERE total.CohortId = m.CohortId
            ) AS CohortSize
        FROM manual_validation_articles m
        JOIN manual_validation_cohorts c ON c.CohortId = m.CohortId
        JOIN articles a
            ON a.RecordId = m.RecordId AND a.Publication = c.Publication
        WHERE m.CohortId = ?
          AND m.ManualResult IS NULL
          AND c.PipelineStep = ?
        ORDER BY m.SampleOrder
        LIMIT 1
    """


def _save_manual_result_sql() -> str:
    """Save one human classification without changing article pipeline data."""
    return """
        UPDATE manual_validation_articles
        SET ManualResult = ?
        WHERE CohortId = ?
          AND RecordId = ?
          AND ManualResult IS NULL
    """


def _completed_cohort_sql() -> str:
    """Check that a requested cohort is complete and in the active scope."""
    return """
        SELECT c.CohortType
        FROM manual_validation_cohorts c
        WHERE c.CohortId = ?
          AND c.Publication = ?
          AND EXISTS (
              SELECT 1 FROM manual_validation_articles a
              WHERE a.CohortId = c.CohortId
          )
          AND NOT EXISTS (
              SELECT 1 FROM manual_validation_articles a
              WHERE a.CohortId = c.CohortId
                AND a.ManualResult IS NULL
          )
        LIMIT 1
    """


def _cohort_accuracy_sql() -> str:
    """Compare manual results with original and current automated results."""
    return """
        SELECT
            c.CohortType,
            COUNT(*) AS TotalCount,
            SUM(CASE
                WHEN c.CohortType IN (
                    'second_filter_no_homicide', 'extraction_no_incidents'
                )
                     AND m.ManualResult = 'N' THEN 1
                WHEN c.CohortType = 'second_filter_early_homicide'
                     AND m.ManualResult = 'E' THEN 1
                WHEN c.CohortType = 'extraction_completeness'
                     AND m.ManualResult = 'A' THEN 1
                WHEN c.CohortType <> 'second_filter_no_homicide'
                     AND m.ManualResult = m.OriginalAutoClass THEN 1
                ELSE 0
            END) AS OriginalCorrectCount,
            SUM(CASE
                WHEN c.CohortType IN (
                    'second_filter_no_homicide', 'extraction_no_incidents'
                )
                     AND a.GPTClass IN (
                         'M', 'M_PRELIM', 'O', 'E', 'N_NOHOM', 'N_NOINC',
                         'N_VEH', 'N_FIC', 'N_MIL', 'N_LAW', 'N'
                     ) THEN 1
                WHEN c.CohortType = 'second_filter_outside_area'
                     AND a.GPTClass IN ('M', 'M_PRELIM', 'O') THEN 1
                WHEN c.CohortType = 'second_filter_early_homicide'
                     AND a.GPTClass IN (
                         'E', 'M', 'M_PRELIM', 'N', 'N_NOHOM', 'N_NOINC',
                         'N_VEH', 'N_FIC', 'N_MIL', 'N_LAW'
                     ) THEN 1
                WHEN c.CohortType NOT IN (
                    'second_filter_no_homicide', 'second_filter_outside_area',
                    'second_filter_early_homicide', 'extraction_no_incidents',
                    'extraction_completeness'
                )
                     AND a.AutoClass IN ('M', 'N', 'O') THEN 1
                ELSE 0
            END) AS CurrentAvailableCount,
            SUM(CASE
                WHEN c.CohortType = 'first_filter_no_homicide'
                     AND a.AutoClass = 'N' AND m.ManualResult = 'N' THEN 1
                WHEN c.CohortType = 'first_filter_no_homicide'
                     AND a.AutoClass IN ('M', 'O')
                     AND m.ManualResult = 'H' THEN 1
                WHEN c.CohortType = 'first_filter_outside_area'
                     AND a.AutoClass = m.ManualResult THEN 1
                WHEN c.CohortType IN (
                    'second_filter_no_homicide', 'extraction_no_incidents'
                )
                     AND a.GPTClass IN ('N_NOHOM', 'N_NOINC', 'N_FIC', 'N')
                     AND m.ManualResult = 'N' THEN 1
                WHEN c.CohortType IN (
                    'second_filter_no_homicide', 'extraction_no_incidents'
                )
                     AND a.GPTClass IN (
                         'M', 'M_PRELIM', 'O', 'E', 'N_VEH', 'N_MIL', 'N_LAW'
                     )
                     AND m.ManualResult = 'H' THEN 1
                WHEN c.CohortType = 'second_filter_outside_area'
                     AND a.GPTClass = 'O'
                     AND m.ManualResult = 'O' THEN 1
                WHEN c.CohortType = 'second_filter_outside_area'
                     AND a.GPTClass IN ('M', 'M_PRELIM')
                     AND m.ManualResult = 'M' THEN 1
                WHEN c.CohortType = 'second_filter_early_homicide'
                     AND a.GPTClass = 'E'
                     AND m.ManualResult = 'E' THEN 1
                WHEN c.CohortType = 'second_filter_early_homicide'
                     AND a.GPTClass IN (
                         'M', 'M_PRELIM', 'N', 'N_NOHOM', 'N_NOINC', 'N_VEH',
                         'N_FIC', 'N_MIL', 'N_LAW'
                     )
                     AND m.ManualResult = 'I' THEN 1
                ELSE 0
            END) AS CurrentCorrectCount,
            SUM(CASE
                WHEN c.CohortType = 'extraction_completeness'
                     AND a.gptVictimJson IS NOT m.OriginalAutoClass THEN 1
                ELSE 0
            END) AS CurrentOutputChangedCount
        FROM manual_validation_articles m
        JOIN manual_validation_cohorts c ON c.CohortId = m.CohortId
        JOIN articles a
            ON a.RecordId = m.RecordId AND a.Publication = c.Publication
        WHERE m.CohortId = ?
          AND c.Publication = ?
          AND c.PipelineStep = ?
          AND c.CohortType = ?
          AND m.ManualResult IS NOT NULL
    """


def _create_cohort(
    publication_id: int,
    classified_dataset: String,
    size: int,
    cohort_type: String,
    original_result: String,
) -> Run[NextStep]:
    """Create cohort metadata and insert its stratified article sample."""
    return sql_query(
        SQL(_has_eligible_articles_sql(cohort_type)),
        SQLParams(_eligibility_params(
            publication_id, classified_dataset, cohort_type, original_result
        )),
    ) >> (
        lambda eligible: _create_when_eligible(
            publication_id,
            classified_dataset,
            size,
            cohort_type,
            original_result,
            bool(eligible[0]["HasEligibleArticles"]),
        )
    )


def _create_when_eligible(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    publication_id: int,
    classified_dataset: String,
    size: int,
    cohort_type: String,
    original_result: String,
    eligible: bool,
) -> Run[NextStep]:
    """Avoid creating an empty cohort when no matching articles exist."""
    if not eligible:
        return put_line(
            "No eligible articles are available for this cohort type."
        ) ^ manual_validation()

    return sql_exec(
        SQL(_insert_cohort_sql()),
        SQLParams((
            publication_id,
            _step_for_cohort_type(cohort_type),
            cohort_type,
            size,
        )),
    ) >> (
        lambda _: sql_query(SQL(_last_cohort_id_sql())) >> (
            lambda ids: _save_cohort_sample(
                publication_id,
            classified_dataset,
                size,
                int(ids[0]["CohortId"]),
                cohort_type,
                original_result,
            )
        )
    )


def _save_cohort_sample(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    publication_id: int,
    classified_dataset: String,
    size: int,
    cohort_id: int,
    cohort_type: String,
    original_result: String,
) -> Run[NextStep]:
    """Persist sampled article membership and return to the cohort menu."""
    return sql_exec(
        SQL(_insert_cohort_articles_sql(cohort_type)),
        SQLParams(_sample_params(
            publication_id, classified_dataset, cohort_type,
            original_result, cohort_id, size,
        )),
    ) >> (
        lambda _: sql_query(
            SQL(_cohort_size_sql()), SQLParams((cohort_id,))
        ) >> (
            lambda counts: _report_created_cohort(
                cohort_id, size, int(counts[0]["SampledCount"])
            )
        )
    )


def _report_created_cohort(
    cohort_id: int, requested_size: int, sampled_count: int
) -> Run[NextStep]:
    """Report actual cohort size and redisplay the validation menu."""
    return put_line(
        f"Created cohort {cohort_id}: sampled {sampled_count} of "
        f"{requested_size} requested article(s)."
    ) ^ manual_validation()


def _render_cohorts(rows: Array[CohortRow]) -> str:
    """Render completed and incomplete cohorts in separate sections."""
    completed = tuple(
        row for row in rows
        if int(row["SampledCount"]) > 0
        and int(row["ValidatedCount"]) == int(row["SampledCount"])
    )
    incomplete = tuple(row for row in rows if row not in completed)

    def render_section(
        title: str, cohorts: tuple[CohortRow, ...]
    ) -> tuple[str, ...]:
        if not cohorts:
            return (f"{title}: none",)
        return (title,) + tuple(
            f"  Cohort {row['CohortId']} — "
            f"{_cohort_type_label(String(row['CohortType']))}; "
            f"{row['ValidatedCount']}/{row['SampledCount']} validated; "
            f"requested {row['RequestedSize']}; created {row['CreatedAt']}"
            for row in cohorts
        )

    return "\n".join(
        ("Manual Validation Cohorts", "")
        + render_section("Completed", completed)
        + ("",)
        + render_section("Incomplete", incomplete)
    )


def _cohort_type_label(cohort_type: String) -> str:
    """Return the user-facing label for a validation cohort type."""
    labels = {
        COHORT_TYPE_OUTSIDE_AREA: "First filter outside-area homicide",
        COHORT_TYPE_GPT_NO_HOMICIDE: "Second filter GPT no homicide",
        COHORT_TYPE_GPT_OUTSIDE_AREA: "Second filter outside-area homicide",
        COHORT_TYPE_GPT_EARLY_HOMICIDE: "Second filter early homicide",
        COHORT_TYPE_EXTRACTION_NO_INCIDENTS:
            "Incident extraction no incidents (N_NOINC)",
        COHORT_TYPE_EXTRACTION_COMPLETENESS:
            "Incident extraction completeness (M)",
        COHORT_TYPE_NO_HOMICIDE: "First filter no homicide",
    }
    return labels.get(cohort_type, f"Validation type {cohort_type}")


def manual_validation() -> Run[NextStep]:
    """List cohorts belonging to the active publication and return."""
    def from_environment(env: Environment) -> Run[NextStep]:
        publication_id = env["publication_profile"].identity.database_id.value
        return sql_query(
            SQL(_cohort_summary_sql()),
            SQLParams((publication_id,)),
        ) >> show_cohorts

    def show_cohorts(rows: Array[CohortRow]) -> Run[NextStep]:
        return put_line(_render_cohorts(rows)) ^ (
            input_from_menu(MenuPrompts(VALIDATION_MENU))
            >> handle_menu_choice
        )

    return ask() >> from_environment


def select_cohort_type() -> Run[NextStep]:
    """Ask which supported validation cohort type to create."""
    return ask() >> (
        lambda env: _show_cohort_type_menu(_early_cutoff_year(env))
    )


def _early_cutoff_year(env: Environment) -> int:
    """Read the first in-scope homicide year from the active publication."""
    profile = env["publication_profile"]
    start_date = profile.analytical_scope.incident_date_scope.start
    return int(str(start_date)[:4])


def _show_cohort_type_menu(early_year: int) -> Run[NextStep]:
    """Display cohort types with the publication-specific early cutoff."""
    menu = (
        "[F]irst filter no homicide",
        "First filter [O]utside-area homicide",
        "Second filter [G]PT no homicide",
        "Second filter [T] outside-area homicide",
        f"Second filter [E]arly homicide (before {early_year})",
        "Incident extraction [X] no incidents (N_NOINC)",
        "Incident extraction [C]ompleteness (M; check for missing incidents)",
        "Return to [M]ain menu",
    )
    return (
        put_line("Select the validation cohort type:")
        ^ input_from_menu(MenuPrompts(menu))
        >> handle_cohort_type_choice
    )


def handle_cohort_type_choice(choice: MenuChoice) -> Run[NextStep]:
    """Handle cohort type selection, return, and quit choices."""
    match choice.upper():
        case "F" | "G":
            return prompt_cohort_size(
                COHORT_TYPE_GPT_NO_HOMICIDE
                if choice.upper() == "G" else COHORT_TYPE_NO_HOMICIDE,
                String("N_NOHOM") if choice.upper() == "G" else String("N"),
            )
        case "O" | "T":
            return prompt_cohort_size(
                COHORT_TYPE_GPT_OUTSIDE_AREA
                if choice.upper() == "T" else COHORT_TYPE_OUTSIDE_AREA,
                String("O"),
            )
        case "E" | "X" | "C":
            return prompt_cohort_size(
                COHORT_TYPE_GPT_EARLY_HOMICIDE
                if choice.upper() == "E"
                else COHORT_TYPE_EXTRACTION_COMPLETENESS
                if choice.upper() == "C"
                else COHORT_TYPE_EXTRACTION_NO_INCIDENTS,
                String("E") if choice.upper() == "E"
                else String("M") if choice.upper() == "C"
                else String("N_NOINC"),
            )
        case "M":
            return pure(NextStep.CONTINUE)
        case "Q":
            return pure(NextStep.QUIT)
        case _:
            return select_cohort_type()


def prompt_cohort_size(
    cohort_type: String, original_result: String
) -> Run[NextStep]:
    """Request a positive sample size or return to the cohort menu on zero."""
    return input_number(
        PromptKey("Enter requested cohort size (0 to return) > "),
        min_value=0,
    ) >> (
        lambda size: create_for_requested_size(
            size, cohort_type, original_result
        )
    )


def create_for_requested_size(
    size: int, cohort_type: String, original_result: String
) -> Run[NextStep]:
    """Create a selected validation cohort when its size is positive."""
    if size == 0:
        return manual_validation()
    return ask() >> (
        lambda env: _create_cohort(
            env["publication_profile"].identity.database_id.value,
            String(str(
                env["publication_profile"].policies.workflow_datasets.classified
            )),
            size,
            cohort_type,
            original_result,
        )
    )


def handle_menu_choice(choice: MenuChoice) -> Run[NextStep]:
    """Dispatch creation or return/quit choices from the validation menu."""
    match choice.upper():
        case "N":
            return select_cohort_type()
        case "V":
            return select_incomplete_cohort()
        case "R":
            return select_completed_cohort()
        case "M":
            return pure(NextStep.CONTINUE)
        case "Q":
            return pure(NextStep.QUIT)
        case _:
            return manual_validation()


def select_incomplete_cohort() -> Run[NextStep]:
    """Prompt for an incomplete cohort ID, with main-menu and quit choices."""
    return get_line(InputPrompt(
        "Enter incomplete cohort ID to validate, "
        "[M]ain menu, [Q]uit > "
    )) >> handle_incomplete_cohort_id


def handle_incomplete_cohort_id(raw_id: String) -> Run[NextStep]:
    """Resolve navigation choices or validate the selected cohort ID."""
    normalized = str(raw_id).strip().upper()
    if normalized == "M":
        return pure(NextStep.CONTINUE)
    if normalized == "Q":
        return pure(NextStep.QUIT)
    if not normalized.isdigit() or int(normalized) <= 0:
        return put_line("Enter a listed cohort ID, M, or Q.") ^ manual_validation()
    cohort_id = int(normalized)
    return ask() >> (
        lambda env: _validate_selected_cohort(
            cohort_id,
            env["publication_profile"].identity.database_id.value,
        )
    )


def _validate_selected_cohort(
    cohort_id: int, publication_id: int
) -> Run[NextStep]:
    """Check cohort scope and begin/resume its pending article review."""
    return sql_query(
        SQL(_incomplete_cohort_sql()),
        SQLParams((cohort_id, publication_id)),
    ) >> (
        lambda rows: _begin_validation(cohort_id, String(rows[0]["CohortType"]))
        if rows else _reject_incomplete_cohort(cohort_id)
    )


def _reject_incomplete_cohort(cohort_id: int) -> Run[NextStep]:
    """Report an invalid or completed cohort selection."""
    return put_line(
        f"Cohort {cohort_id} is not an incomplete validation cohort "
        "for the active publication."
    ) ^ manual_validation()


def _begin_validation(cohort_id: int, cohort_type: String) -> Run[NextStep]:
    """Start scoring when pending membership exists, otherwise return to menu."""
    return _validate_next_article(cohort_id, cohort_type)


def _validate_next_article(cohort_id: int, cohort_type: String) -> Run[NextStep]:
    """Display the next pending article or finish and return to the cohort menu."""
    return sql_query(
        SQL(_next_unvalidated_article_sql()),
        SQLParams((cohort_id, _step_for_cohort_type(cohort_type))),
    ) >> (
        lambda rows: _finish_validation(cohort_id)
        if not rows
        else _score_article(cohort_id, cohort_type, rows[0])
    )


def _finish_validation(cohort_id: int) -> Run[NextStep]:
    """Report completion and return to the validation cohort menu."""
    return put_line(f"All articles in cohort {cohort_id} are validated.") ^ (
        manual_validation()
    )


def _article_date(pub_date: str) -> str:
    """Format the stored YYYYMMDD publication date for manual review."""
    if len(pub_date) == 8:
        return f"{pub_date[4:6]}/{pub_date[6:8]}/{pub_date[0:4]}"
    return pub_date


def _score_article(
    cohort_id: int, cohort_type: String, article: CohortRow
) -> Run[NextStep]:
    """Display a cohort-specific review and save the user's decision."""
    if cohort_type == COHORT_TYPE_GPT_EARLY_HOMICIDE:
        return ask() >> (
            lambda env: _display_article_for_score(
                cohort_id,
                cohort_type,
                article,
                _early_cutoff_year(env),
            )
        )
    return _display_article_for_score(cohort_id, cohort_type, article, None)


def _display_article_for_score(
    cohort_id: int,
    cohort_type: String,
    article: CohortRow,
    early_year: int | None,
) -> Run[NextStep]:
    """Render one article and its cohort-specific manual scoring options."""
    publication = int(article["Publication"])
    title = rich_text(str(article["Title"] or ""), publication)
    full_text = rich_text(str(article["FullText"] or ""), publication)
    display = (
        f"Article {article['ProgressNumber']}/{article['CohortSize']}\n"
        f"Title: {title}\n"
        f"Date: {_article_date(str(article['PubDate']))}\n\n"
        f"{full_text}\n"
    )
    if cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS:
        extraction_snapshot = str(article["ExtractionSnapshot"] or "")
        display += (
            "\nExtracted incidents:\n"
            f"{_format_extracted_incidents(extraction_snapshot)}\n"
        )
        score_menu = (
            "[A]gree: all eligible incidents are represented",
            "[D]isagree: at least one eligible incident is missing",
            "Return to [M]ain menu",
        )
    elif cohort_type == COHORT_TYPE_GPT_EARLY_HOMICIDE:
        score_menu = (
            f"[E]arly homicide (before {early_year})",
            f"Not early [I] ({early_year} or later / no homicide)",
            "Return to [M]ain menu",
        )
    elif cohort_type in (
            COHORT_TYPE_OUTSIDE_AREA, COHORT_TYPE_GPT_OUTSIDE_AREA
    ):
        score_menu = OUTSIDE_AREA_SCORE_MENU
    else:
        score_menu = NO_HOMICIDE_SCORE_MENU
    return put_line(display) ^ input_from_menu(
        MenuPrompts(score_menu)
    ) >> (
        lambda choice: _handle_article_score(
            choice, cohort_id, cohort_type, article
        )
    )


def _format_extracted_incidents(raw_json: str) -> str:
    """Summarize captured incident records for a completeness review."""
    try:
        incidents = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json or "No extracted incident list was captured."
    if not isinstance(incidents, list) or not incidents:
        return "No extracted incident list was captured."

    rendered = []
    for position, incident in enumerate(incidents, start=1):
        if not isinstance(incident, dict):
            rendered.append(f"Incident {position}: {incident}")
            continue
        date_parts = [
            str(incident[key])
            for key in ("year", "month", "day")
            if incident.get(key) is not None
        ]
        victims = incident.get("victim", [])
        victim_names = tuple(
            str(victim["victim_name"])
            for victim in victims
            if isinstance(victim, dict) and victim.get("victim_name")
        ) if isinstance(victims, list) else tuple()
        details = [
            f"Date: {'/'.join(date_parts) if date_parts else 'unknown'}",
            f"Location: {incident.get('location') or 'unknown'}",
            f"Victim(s): {', '.join(victim_names) if victim_names else 'not named'}",
        ]
        if incident.get("summary"):
            details.append(f"Summary: {incident['summary']}")
        rendered.append(f"Incident {position}: " + "; ".join(details))
    return "\n".join(rendered)


def _handle_article_score(
    choice: MenuChoice,
    cohort_id: int,
    cohort_type: String,
    article: CohortRow,
) -> Run[NextStep]:
    """Persist a type-specific result or leave the article pending."""
    selected = str(choice.upper())
    if selected == "M":
        return pure(NextStep.CONTINUE)
    if selected == "Q":
        return pure(NextStep.QUIT)
    if cohort_type in (
        COHORT_TYPE_NO_HOMICIDE,
        COHORT_TYPE_GPT_NO_HOMICIDE,
        COHORT_TYPE_EXTRACTION_NO_INCIDENTS,
    ) and selected in ("H", "N"):
        result = String(selected)
    elif cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS and selected in ("A", "D"):
        result = String(selected)
    elif cohort_type in (
        COHORT_TYPE_OUTSIDE_AREA, COHORT_TYPE_GPT_OUTSIDE_AREA
    ) and selected in ("I", "O"):
        result = String("M") if selected == "I" else String("O")
    elif cohort_type == COHORT_TYPE_GPT_EARLY_HOMICIDE and selected in ("E", "I"):
        result = String(selected)
    else:
        return _validate_next_article(cohort_id, cohort_type)
    return _save_article_result(cohort_id, cohort_type, article, result)


def _save_article_result(
    cohort_id: int,
    cohort_type: String,
    article: CohortRow,
    manual_result: String,
) -> Run[NextStep]:
    """Save one manual label and continue with the same cohort type."""
    return sql_exec(
        SQL(_save_manual_result_sql()),
        SQLParams((manual_result, cohort_id, int(article["RecordId"]))),
    ) >> (lambda _: _validate_next_article(cohort_id, cohort_type))


def select_completed_cohort() -> Run[NextStep]:
    """Prompt for a completed cohort ID, with navigation choices."""
    return get_line(InputPrompt(
        "Enter completed cohort ID to report, [M]ain menu, [Q]uit > "
    )) >> handle_completed_cohort_id


def handle_completed_cohort_id(raw_id: String) -> Run[NextStep]:
    """Resolve navigation choices or report the selected completed cohort."""
    normalized = str(raw_id).strip().upper()
    if normalized == "M":
        return pure(NextStep.CONTINUE)
    if normalized == "Q":
        return pure(NextStep.QUIT)
    if not normalized.isdigit() or int(normalized) <= 0:
        return put_line("Enter a listed cohort ID, M, or Q.") ^ manual_validation()
    cohort_id = int(normalized)
    return ask() >> (
        lambda env: _verify_completed_cohort(
            cohort_id,
            env["publication_profile"].identity.database_id.value,
        )
    )


def _verify_completed_cohort(
    cohort_id: int, publication_id: int
) -> Run[NextStep]:
    """Check cohort scope and completion before calculating its report."""
    return sql_query(
        SQL(_completed_cohort_sql()),
        SQLParams((cohort_id, publication_id)),
    ) >> (
        lambda rows: _load_cohort_accuracy(
            cohort_id, publication_id, String(rows[0]["CohortType"])
        ) if rows else _reject_completed_cohort(cohort_id)
    )


def _reject_completed_cohort(cohort_id: int) -> Run[NextStep]:
    """Report an invalid or incomplete cohort selection."""
    return put_line(
        f"Cohort {cohort_id} is not a completed validation cohort "
        "for the active publication."
    ) ^ manual_validation()


def _load_cohort_accuracy(
    cohort_id: int, publication_id: int, cohort_type: String
) -> Run[NextStep]:
    """Calculate a report only for a completed validation cohort."""
    return sql_query(
        SQL(_cohort_accuracy_sql()),
        SQLParams((
            cohort_id,
            publication_id,
            _step_for_cohort_type(cohort_type),
            cohort_type,
        )),
    ) >> (lambda rows: _show_cohort_accuracy(cohort_id, rows[0]))


def _show_cohort_accuracy(  # pylint: disable=too-many-locals
    cohort_id: int, report: CohortRow
) -> Run[NextStep]:
    """Render original/current accuracy and discrepancy rates."""
    total = int(report["TotalCount"])
    original_correct = int(report["OriginalCorrectCount"])
    current_available = int(report["CurrentAvailableCount"])
    current_correct = int(report["CurrentCorrectCount"])
    current_missing = total - current_available
    cohort_type = String(report["CohortType"])
    if cohort_type == COHORT_TYPE_EXTRACTION_COMPLETENESS:
        disagreed = total - original_correct
        agree_rate = 100 * original_correct / total if total else 0.0
        disagree_rate = 100 * disagreed / total if total else 0.0
        changed = int(report["CurrentOutputChangedCount"])
        report_text = (
            f"Completed cohort {cohort_id}: {_cohort_type_label(cohort_type)}\n"
            f"Agree all eligible incidents are represented: "
            f"{original_correct}/{total} ({agree_rate:.1f}%).\n"
            f"Disagree; at least one eligible incident is missing: "
            f"{disagreed}/{total} ({disagree_rate:.1f}%).\n"
            f"Current extraction output differs from the captured list for "
            f"{changed}/{total} article(s). Changed outputs need manual review."
        )
        return put_line(report_text) ^ manual_validation()
    if cohort_type == COHORT_TYPE_GPT_NO_HOMICIDE:
        current_interpretation = (
            "Current second-filter result maps N_NOHOM, N_NOINC, N_FIC, and N "
            "to no homicide; M, M_PRELIM, O, E, N_VEH, N_MIL, and N_LAW to "
            "homicide. Unmapped results are excluded."
        )
        current_label = "Current second-filter"
    elif cohort_type == COHORT_TYPE_GPT_EARLY_HOMICIDE:
        current_interpretation = (
            "Current second-filter result maps E to early and recognized other "
            "results to not early. Unmapped results are excluded."
        )
        current_label = "Current second-filter"
    elif cohort_type == COHORT_TYPE_GPT_OUTSIDE_AREA:
        current_interpretation = (
            "Current second-filter result maps O to outside the area, and M or "
            "M_PRELIM to inside the area. Other results are excluded."
        )
        current_label = "Current second-filter"
    elif cohort_type == COHORT_TYPE_EXTRACTION_NO_INCIDENTS:
        current_interpretation = (
            "Current GPT result maps N_NOHOM, N_NOINC, N_FIC, and N to no "
            "homicide; M, M_PRELIM, O, E, N_VEH, N_MIL, and N_LAW to "
            "homicide. Unmapped results are excluded."
        )
        current_label = "Current extraction result"
    elif cohort_type == COHORT_TYPE_NO_HOMICIDE:
        current_interpretation = (
            "Current M and O results count as homicide; current N counts as "
            "no homicide."
        )
        current_label = "Current first-filter"
    else:
        current_interpretation = (
            "Current M and O results are compared directly with inside/outside "
            "labels."
        )
        current_label = "Current first-filter"
    original_summary = _accuracy_summary(
        "Original-result", original_correct, total
    )
    current_summary = _accuracy_summary(
        current_label, current_correct, current_available
    )
    report_text = (
        f"Completed cohort {cohort_id}: {_cohort_type_label(cohort_type)}\n"
        f"{original_summary}\n"
        f"{current_summary}\n"
        f"Current result unavailable for {current_missing}/{total} article(s).\n"
        f"{current_interpretation}"
    )
    return put_line(report_text) ^ manual_validation()


def _accuracy_summary(label: str, correct: int, denominator: int) -> str:
    """Format accuracy and discrepancy count/rate for one comparison."""
    if denominator == 0:
        return f"{label}: no usable results; accuracy unavailable."
    discrepant = denominator - correct
    accuracy = 100 * correct / denominator
    discrepancy_rate = 100 * discrepant / denominator
    return (
        f"{label} accuracy: {correct}/{denominator} ({accuracy:.1f}%); "
        f"discrepancies: {discrepant}/{denominator} "
        f"({discrepancy_rate:.1f}%)."
    )
