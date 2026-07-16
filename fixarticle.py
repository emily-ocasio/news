"""
Monadic controller for reviewing and making changes to a single article
"""
import json
from enum import Enum
from typing import Any

from calculations import (
    latest_gptresults_for_promptkey_sql,
    latest_gptresults_sql,
    single_article_sql,
)
from pymonad import (
    Run,
    Environment,
    Namespace,
    with_namespace,
    to_prompts,
    put_line,
    pure,
    ask,
    input_number,
    input_with_prompt,
    PromptKey,
    sql_query,
    sql_exec,
    SQL,
    SQLParams,
    throw,
    ErrorPayload,
    set_,
    with_models,
    EnvKey,
    String,
    GPTUsage,
    GPTReasoning,
    run_except,
    Left,
    with_duckdb,
)
from pymonad.array import Array
from appstate import prompt_key
from article import Article, Articles, ArticleAppError
from gpt_filtering import GPT_PROMPTS, GPT_MODELS, \
    PROMPT_KEY_STR, process_all_articles, classification_runtime_configuration
from menuprompts import NextStep, MenuPrompts, MenuChoice, input_from_menu
from incidents import process_all_articles as extract_process_all_articles, \
    GPT_PROMPTS as INCIDENTS_GPT_PROMPTS, PROMPT_KEY_STR as INCIDENTS_PROMPT_KEY_STR, \
    GPT_MODELS as INCIDENTS_GPT_MODELS
from single_article_refresh import refresh_single_article_after_extract
from orphan_adjudication_controller import (
    E2E_CACHE_STAGE,
    PASS1_PROMPT_KEY,
    RANK_PROMPT_KEY,
    force_article_adjudication_cache_refresh,
    force_article_adjudication_cache_refresh_strategy_x,
)
from publication_profiles import RecordIdBase
from publication_profiles import Availability

FIX_PROMPTS: dict[str, str | tuple[str,]] = {
    "record_id": "Please enter the record ID of the article you want to fix: ",
    "delete_cache_entry_index": (
        "Select orphan cache entry number to delete, type 'all', or 0 to go back: "
    ),
}
HUMANIZATION_PROMPT_KEYS = (
    "humanize_extract_incident",
    "humanize_deidentify",
    "humanize_classify",
)

class FixAction(Enum):
    """
    Possible actions that can be applied to an article in the fix article workflow.
    """
    SECOND_FILTER = MenuChoice('S')
    EXTRACT_INCIDENTS = MenuChoice('G')
    FORCE_ORPHAN_ADJ_CACHE_REFRESH = MenuChoice('K')
    FORCE_ORPHAN_ADJ_CACHE_REFRESH_X = MenuChoice('X')
    DELETE_ORPHAN_CACHE = MenuChoice('D')
    CONTINUE = MenuChoice('F')
    MAIN_MENU = MenuChoice('M')
    QUIT = MenuChoice('Q')


def _article_prompt(
    include_delete_cache: bool = False,
    include_force_adjudication: bool = False,
) -> tuple[str, ...]:
    """Build the dynamic article-action menu."""
    options = [
        "Apply [S]econd filter via GPT",
        "Extract incident via [G]PT",
    ]
    if include_force_adjudication:
        options.append("Force orphan adjudication cache refresh via [K]")
        options.append(
            "Force orphan adjudication cache refresh via [X] top-score-union"
            " strategy"
        )
    if include_delete_cache:
        options.append("[D]elete orphan adjudication cache")
    options.extend(
        (
            "Select another article to [F]ix",
            "Go back to [M]ain menu",
        )
    )
    return tuple(options)

def input_record_id() -> Run[int]:
    """
    Prompt the user to input a record ID.
    """
    def normalize_record_id(record_id: int, base: RecordIdBase) -> Run[int]:
        if 0 < record_id < base.value:
            return pure(record_id + base.value)
        return pure(record_id)
    def check_if_zero(record_id: int) -> Run[int]:
        if record_id <= 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(record_id)
    def from_environment(env: Environment) -> Run[int]:
        base = env["publication_profile"].policies.record_id_base

        def normalize_entered_id(record_id: int) -> Run[int]:
            return normalize_record_id(record_id, base)

        return input_number(PromptKey("record_id")) >> normalize_entered_id \
            >> check_if_zero

    return ask() >> from_environment

def retrieve_single_article() -> Run[Article]:
    """
    Retrieve a single article with based on a single record ID.
    """
    def message_intention(record_id: int) -> Run[int]:
        return (
            put_line(f"Retrieving article with record ID: {record_id}...")
            ^ pure(record_id)
        )
    def retrieve_article(record_id: int) -> Run[Articles]:
        return (
            Articles.from_rows
            & sql_query(SQL(single_article_sql()), SQLParams((record_id,)))
        )
    def validate_retrieval(articles: Articles) -> Run[Article]:
        """
        Retrieve and display the article, then continue.
        """
        match len(articles):
            case 0:
                return \
                    throw(ErrorPayload("", ArticleAppError.NO_ARTICLE_FOUND))
            case 1:
                return pure(articles[0])
            case _:
                return \
                    throw(ErrorPayload("", ArticleAppError.MULTIPLE_ARTICLES_FOUND))

    return (
        input_record_id()
        >> message_intention
        >> retrieve_article
        >> validate_retrieval
    )

def _display_article(article: Article) -> Run[Article]:
    """
    Display the retrieved article and continue.
    """
    return \
        put_line(f"Retrieved article:\n {article}") ^ \
        pure(article)

def _display_latest_gpt_response(article: Article) -> Run[Article]:
    """
    Display the latest general GPT result and the latest [K] API 2 result.
    """
    def _format_latest_row(
        label: str, rows: Array[Any], *, empty_message: str
    ) -> Run[Article]:
        if len(rows) == 0:
            return put_line(empty_message) ^ pure(article)
        row = rows[0]
        usage = GPTUsage.from_row(row)
        reasoning = GPTReasoning.from_row(row)
        prompt_key_value = str(row["PromptKey"] or "")
        timestamp = str(row["TimeStamp"] or "")
        return (
            put_line(
                f"{label}\nPromptKey: {prompt_key_value}\nTimestamp: {timestamp}"
            )
            ^ put_line(str(usage))
            ^ put_line(f"GPT reasoning summary:\n{reasoning}")
            ^ pure(article)
        )

    record_id = article.record_id or 0
    excluded_prompt_keys = (
        String(HUMANIZATION_PROMPT_KEYS[0]),
        String(HUMANIZATION_PROMPT_KEYS[1]),
        String(HUMANIZATION_PROMPT_KEYS[2]),
        String(PASS1_PROMPT_KEY),
        String(RANK_PROMPT_KEY),
    )
    return sql_query(
        SQL(latest_gptresults_sql(len(excluded_prompt_keys))),
        SQLParams((record_id, *excluded_prompt_keys)),
    ) >> (
        lambda latest_general_rows: _format_latest_row(
            "Latest non-excluded GPT result:",
            latest_general_rows,
            empty_message="No non-excluded GPT responses captured for this article.\n",
        )
        ^ sql_query(
            SQL(latest_gptresults_for_promptkey_sql()),
            SQLParams((record_id, String(RANK_PROMPT_KEY))),
        )
        >> (
            lambda latest_api2_rows: _format_latest_row(
                "Latest [K] API 2 GPT result:",
                latest_api2_rows,
                empty_message="No [K] API 2 GPT responses captured for this article.\n",
            )
        )
    )

def _cache_rows_for_article(article: Article) -> Run[tuple[dict, ...]]:
    """
    Return current [K]-produced cache rows for this article directly from llm_cache.
    """
    record_id = article.record_id or 0
    if record_id <= 0:
        return pure(tuple())

    return with_duckdb(
        sql_query(
            SQL(
                """
                SELECT
                  lc.model,
                  lc.prompt_version,
                  lc.updated_at AS cache_updated_at,
                  lc.idempotency_key AS orphan_id,
                  lc.idempotency_key AS cache_key,
                  CAST(lc.response_json AS VARCHAR) AS response_json_text
                FROM llm_cache lc
                WHERE lc.stage = ?
                  AND lc.idempotency_key LIKE ?
                ORDER BY lc.idempotency_key;
                """
            ),
            SQLParams((String(E2E_CACHE_STAGE), String(f"{record_id}:%"))),
        )
    ) >> (lambda rows: pure(tuple(dict(r) for r in rows)))


def _adjudication_rows_for_article(article: Article) -> Run[tuple[dict, ...]]:
    """
    Return current [K]-eligible orphan rows for this article.
    """
    record_id = article.record_id or 0
    if record_id <= 0:
        return pure(tuple())

    return with_duckdb(
        sql_query(
            SQL(
                """
                SELECT
                  uid AS orphan_id,
                  article_id,
                  midpoint_day
                FROM orphan_matches_final_current
                WHERE rec_type = 'orphan'
                  AND match_id LIKE 'orphan_%'
                  AND article_id = ?
                ORDER BY midpoint_day NULLS LAST, uid;
                """
            ),
            SQLParams((record_id,)),
        )
    ) >> (lambda rows: pure(tuple(dict(r) for r in rows)))

def _display_cache_rows(rows: tuple[dict, ...]) -> Run[None]:
    """
    Display the orphan adjudication cache content for this article.
    """
    if len(rows) == 0:
        return pure(None)

    def fmt(i: int, row: dict) -> str:
        orphan_id = str(row.get("orphan_id") or "")
        cache_key = str(row.get("cache_key") or "")
        updated_at = str(row.get("cache_updated_at") or "")
        model = str(row.get("model") or "")
        prompt_version = str(row.get("prompt_version") or "")
        response_json_text = str(row.get("response_json_text") or "")
        label = ""
        resolved_entity_id = ""
        try:
            response_payload = json.loads(response_json_text)
            if isinstance(response_payload, dict):
                label = str(response_payload.get("label") or "")
                resolved_entity_id = str(
                    response_payload.get("resolved_entity_id") or ""
                )
        except json.JSONDecodeError:
            pass
        return (
            f"[F] [K] cache entry {i}\n"
            f"  orphan_id: {orphan_id}\n"
            f"  cache_key: {cache_key}\n"
            f"  model: {model}\n"
            f"  prompt_version: {prompt_version}\n"
            f"  cache_updated_at: {updated_at}\n"
            f"  label: {label}\n"
            f"  resolved_entity_id: {resolved_entity_id}"
        )

    message = "[F] [K] orphan adjudication cache entries for this article:\n" + \
        "\n\n".join(fmt(i, row) for i, row in enumerate(rows, start=1))
    return put_line(message) ^ pure(None)

def _choose_cache_entries(rows: tuple[dict, ...]) -> Run[tuple[dict, ...]]:
    """
    Select one or all orphan cache entries to delete.
    """
    if len(rows) == 1:
        return pure((rows[0],))

    listing = "\n".join(
        f"  {i}. orphan_id={str(row.get('orphan_id') or '')} "
        f"cache_key={str(row.get('cache_key') or '')}"
        for i, row in enumerate(rows, start=1)
    )

    def _parse_selection(raw_value: str) -> Run[tuple[dict, ...]]:
        value = raw_value.strip()
        if value == "0":
            return pure(tuple())
        if value.lower() == "all":
            return pure(rows)
        try:
            selected = int(value)
        except ValueError:
            return put_line("[F] Invalid orphan cache entry selection.") >> (
                lambda _: _choose_cache_entries(rows)
            )
        return (
            pure((rows[selected - 1],))
            if 1 <= selected <= len(rows)
            else put_line("[F] Invalid orphan cache entry selection.") >> (
                lambda _: _choose_cache_entries(rows)
            )
        )

    return (
        put_line(
            "[F] Multiple orphan cache entries found. Choose one to delete, "
            "type 'all', or enter 0 to go back:\n"
            "  0. go back\n"
            "  all. delete all entries\n"
            + listing
        )
        ^ input_with_prompt(PromptKey("delete_cache_entry_index"))
    ) >> (lambda selected: _parse_selection(str(selected)))

def _delete_cache_entry(_article: Article, row: dict) -> Run[None]:
    """
    Delete one selected orphan adjudication cache entry.
    """
    orphan_id = str(row.get("orphan_id") or "")
    cache_key = str(row.get("cache_key") or "")

    return with_duckdb(
        sql_exec(
            SQL(
                """
                DELETE FROM llm_cache
                WHERE stage = ?
                  AND idempotency_key = ?;
                """
            ),
            SQLParams((String(E2E_CACHE_STAGE), String(cache_key))),
        )
        ^ put_line(
            "[F] Removed orphan adjudication cache entry "
            f"(orphan_id={orphan_id}, cache_key={cache_key})."
        )
        ^ pure(None)
    )

def _delete_cache_entries(article: Article, rows: tuple[dict, ...]) -> Run[None]:
    """
    Delete one or more selected orphan adjudication cache entries.
    """
    if len(rows) == 0:
        return pure(None)
    return _delete_cache_entry(article, rows[0]) >> (
        lambda _: _delete_cache_entries(article, rows[1:])
    )

def _select_apply_action(
    article: Article, allow_cache_delete: bool = True
) -> Run[NextStep]:
    """
    Select the desired action to apply to the article.
    """
    def _with_article_adjudication_rows(
        cache_rows: tuple[dict, ...], adjudication_rows: tuple[dict, ...]
    ) -> Run[NextStep]:
        return (
            _display_cache_rows(cache_rows)
            ^ input_desired_action(
                include_delete_cache=allow_cache_delete and len(cache_rows) > 0,
                include_force_adjudication=len(adjudication_rows) > 0,
            )
            >> (
                lambda action: _apply_action(
                    article,
                    action,
                    cache_rows,
                    adjudication_rows,
                    allow_cache_delete,
                )
            )
        )

    def from_environment(env: Environment) -> Run[NextStep]:
        if (
            env["publication_profile"].capabilities.orphan_adjudication
            is not Availability.AVAILABLE
        ):
            return _with_article_adjudication_rows(tuple(), tuple())
        return _cache_rows_for_article(article) >> (
            lambda cache_rows: _adjudication_rows_for_article(article)
            >> (
                lambda adjudication_rows: _with_article_adjudication_rows(
                    cache_rows, adjudication_rows
                )
            )
        )

    return ask() >> from_environment

def input_desired_action(
    include_delete_cache: bool = False,
    include_force_adjudication: bool = False,
) -> Run[FixAction]:
    """
    Prompt the user to input the desired action to apply to the article.
    """
    return (
        FixAction & input_from_menu(
            MenuPrompts(
                _article_prompt(
                    include_delete_cache, include_force_adjudication
                )
            )
        )
    )

def second_filter(article: Article) -> Run[Article]:
    """
    Process second filter action for the article.
    """
    return ask() >> (lambda env: (
        process_all_articles(
            Articles((article,)),
            classification_runtime_configuration(env["publication_profile"]),
        ) ^ pure(article)
    ))


def _apply_action(
    article: Article,
    action: FixAction,
    cache_rows: tuple[dict, ...],
    adjudication_rows: tuple[dict, ...],
    allow_cache_delete: bool,
) -> Run[NextStep]:
    """
    Apply the desired action to the article and continue.
    """
    def _post_extract_refresh() -> Run[None]:
        record_id = article.record_id or 0
        return run_except(refresh_single_article_after_extract(record_id)) >> (
            lambda res: (
                put_line(
                    "[F] Warning: post-[G] single-article refresh failed: "
                    f"{res.l}"
                )
                if isinstance(res, Left)
                else pure(None)
            )
        )

    def _run_forced_article_adjudication_refresh() -> Run[NextStep]:
        record_id = article.record_id or 0
        return (
            put_line(
                "[F] Forcing orphan adjudication cache refresh for article "
                f"{record_id}: rows={len(adjudication_rows)}."
            )
            ^ force_article_adjudication_cache_refresh(record_id)
        ) >> (
            lambda summary: put_line(
                "[F] [K] cache refresh completed:"
                f" article_id={summary.article_id},"
                f" orphan_rows={summary.orphan_rows},"
                f" groups={summary.groups},"
                f" cached_terminal={summary.cached_terminal},"
                f" matched={summary.matched},"
                f" no_match={summary.not_same_person},"
                f" insufficient={summary.insufficient_information},"
                f" incomplete={summary.analysis_incomplete}"
            )
            ^ _select_apply_action(article, allow_cache_delete)
        )

    def _run_forced_article_adjudication_refresh_x() -> Run[NextStep]:
        record_id = article.record_id or 0
        return (
            put_line(
                "[F] Forcing orphan adjudication cache refresh for article "
                f"{record_id} via [X] top-score-union strategy:"
                f" rows={len(adjudication_rows)}."
            )
            ^ force_article_adjudication_cache_refresh_strategy_x(record_id)
        ) >> (
            lambda summary: put_line(
                "[F] [X] cache refresh completed:"
                f" article_id={summary.article_id},"
                f" orphan_rows={summary.orphan_rows},"
                f" groups={summary.groups},"
                f" cached_terminal={summary.cached_terminal},"
                f" matched={summary.matched},"
                f" no_match={summary.not_same_person},"
                f" insufficient={summary.insufficient_information},"
                f" incomplete={summary.analysis_incomplete}"
            )
            ^ _select_apply_action(article, allow_cache_delete)
        )

    result: Run[NextStep]
    match action:
        case FixAction.SECOND_FILTER:
            result = \
                put_line("Dispatching to second filter...") ^ \
                set_(prompt_key, String(PROMPT_KEY_STR)) ^ \
                pure(article) >> second_filter >> \
                (lambda updated: _select_apply_action(updated, allow_cache_delete))
        case FixAction.EXTRACT_INCIDENTS:
            result = (
                set_(prompt_key, String(INCIDENTS_PROMPT_KEY_STR)) ^
                extract_process_all_articles(Articles((article,)))
                ^ _post_extract_refresh()
                ^ pure(article)
                >> (lambda updated: _select_apply_action(updated, allow_cache_delete))
            )
        case FixAction.DELETE_ORPHAN_CACHE:
            result = (
                _choose_cache_entries(cache_rows)
                >> (lambda selected: _delete_cache_entries(article, selected))
                >> (lambda _: _select_apply_action(article, allow_cache_delete))
            )
        case FixAction.FORCE_ORPHAN_ADJ_CACHE_REFRESH:
            result = _run_forced_article_adjudication_refresh()
        case FixAction.FORCE_ORPHAN_ADJ_CACHE_REFRESH_X:
            result = _run_forced_article_adjudication_refresh_x()
        case FixAction.QUIT:
            result = pure(NextStep.QUIT)
        case FixAction.CONTINUE:
            result = fix_article()
        case FixAction.MAIN_MENU:
            result = pure(NextStep.CONTINUE)
    return result


def select_fix_article() -> Run[NextStep]:
    """
    Select an article for fixing
    """
    return (
        retrieve_single_article()
        >> _display_article
        >> _display_latest_gpt_response
        >> _select_apply_action
    )

def fix_article() -> Run[NextStep]:
    """
    Fix an article by its record ID.
    """
    def configure(env: Environment) -> Run[NextStep]:
        classification_config = classification_runtime_configuration(
            env["publication_profile"]
        )
        prompts = FIX_PROMPTS | GPT_PROMPTS | INCIDENTS_GPT_PROMPTS | {
            str(classification_config.prompt_key): (
                str(classification_config.prompt_id),
            )
        }
        models = GPT_MODELS | INCIDENTS_GPT_MODELS | {
            EnvKey("filter"): classification_config.model
        }
        return with_models(
            models,
            with_namespace(
                Namespace("fix"),
                to_prompts(prompts),
                select_fix_article()
            )
        )

    return ask() >> configure
