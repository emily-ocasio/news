"""
Monadic controller for reviewing and making changes to a single article
"""
from enum import Enum
from calculations import single_article_sql, latest_gptresults_sql
from pymonad import Run, Namespace, with_namespace, to_prompts, put_line, \
    pure, input_number, PromptKey, sql_query, sql_exec, SQL, SQLParams, throw, \
    ErrorPayload, set_, with_models, String, Tuple, GPTUsage, \
        GPTReasoning, Maybe, Just, _Nothing, gpt_usage_reasoning_from_rows, \
        run_except, Left, with_duckdb
from appstate import prompt_key
from article import Article, Articles, ArticleAppError
from gpt_filtering import GPT_PROMPTS, GPT_MODELS, \
    PROMPT_KEY_STR, process_all_articles
from menuprompts import NextStep, MenuPrompts, MenuChoice, input_from_menu
from incidents import process_all_articles as extract_process_all_articles, \
    GPT_PROMPTS as INCIDENTS_GPT_PROMPTS, PROMPT_KEY_STR as INCIDENTS_PROMPT_KEY_STR, \
    GPT_MODELS as INCIDENTS_GPT_MODELS
from single_article_refresh import refresh_single_article_after_extract
from orphan_adjudication_controller import E2E_CACHE_STAGE

FIX_PROMPTS: dict[str, str | tuple[str,]] = {
    "record_id": "Please enter the record ID of the article you want to fix: ",
    "delete_cache_entry_index": "Select orphan cache entry number to delete: ",
}
ARTICLE_PROMPT = ("Apply [S]econd filter via GPT",
                  "Extract incident via [G]PT",
                  "Select another article to [F]ix",
                  "Go back to [M]ain menu")
ARTICLE_PROMPT_WITH_DELETE = ("Apply [S]econd filter via GPT",
                              "Extract incident via [G]PT",
                              "[D]elete orphan adjudication cache",
                              "Select another article to [F]ix",
                              "Go back to [M]ain menu")
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
    DELETE_ORPHAN_CACHE = MenuChoice('D')
    CONTINUE = MenuChoice('F')
    MAIN_MENU = MenuChoice('M')
    QUIT = MenuChoice('Q')

def input_record_id() -> Run[int]:
    """
    Prompt the user to input a record ID.
    """
    def normalize_record_id(record_id: int) -> Run[int]:
        if 0 < record_id < 100000000:
            return pure(record_id + 100000000)
        return pure(record_id)
    def check_if_zero(record_id: int) -> Run[int]:
        if record_id <= 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(record_id)
    return \
        input_number(PromptKey('record_id')) >> normalize_record_id >> check_if_zero

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
    Display the latest GPT usage + reasoning captured for this article.
    """
    def after_query(maybe_usage: Maybe[Tuple[GPTUsage, GPTReasoning]]) -> Run[Article]:
        match maybe_usage:
            case Just(tup):
                return (
                    put_line(str(tup.fst))
                    ^ put_line(f"GPT reasoning summary:\n{tup.snd}")
                    ^ pure(article)
                )
            case _Nothing():
                return (
                    put_line("No GPT responses captured for this article.\n")
                    ^ pure(article)
                )

    record_id = article.record_id or 0
    return \
        (gpt_usage_reasoning_from_rows & \
        sql_query(SQL(latest_gptresults_sql()), SQLParams((
            record_id,
            String(HUMANIZATION_PROMPT_KEYS[0]),
            String(HUMANIZATION_PROMPT_KEYS[1]),
            String(HUMANIZATION_PROMPT_KEYS[2]),
        )))) \
        >> after_query

def _cache_rows_for_article(article: Article) -> Run[tuple[dict, ...]]:
    """
    Return latest [K]-produced cache rows for this article keyed by orphan.
    """
    record_id = article.record_id or 0
    if record_id <= 0:
        return pure(tuple())

    return with_duckdb(
        sql_query(
            SQL(
                """
                WITH readiness_latest AS (
                  SELECT
                    run_id,
                    orphan_id,
                    article_id,
                    readiness_status,
                    readiness_reason,
                    pass1_idempotency_key,
                    created_at,
                    ROW_NUMBER() OVER (
                      PARTITION BY orphan_id, pass1_idempotency_key
                      ORDER BY created_at DESC, run_id DESC
                    ) AS rn
                  FROM orphan_adj_cache_readiness
                  WHERE article_id = ?
                    AND COALESCE(pass1_idempotency_key, '') <> ''
                )
                SELECT
                  rl.run_id,
                  rl.orphan_id,
                  rl.article_id,
                  rl.readiness_status,
                  rl.readiness_reason,
                  rl.pass1_idempotency_key AS cache_key,
                  rl.created_at AS readiness_created_at,
                  lc.model,
                  lc.prompt_version,
                  lc.updated_at AS cache_updated_at,
                  CAST(lc.input_json AS VARCHAR) AS input_json_text,
                  CAST(lc.response_json AS VARCHAR) AS response_json_text
                FROM readiness_latest rl
                JOIN llm_cache lc
                  ON lc.stage = ?
                 AND lc.idempotency_key = rl.pass1_idempotency_key
                WHERE rl.rn = 1
                ORDER BY rl.orphan_id;
                """
            ),
            SQLParams((record_id, String(E2E_CACHE_STAGE))),
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
        readiness_status = str(row.get("readiness_status") or "")
        readiness_reason = str(row.get("readiness_reason") or "")
        run_id = str(row.get("run_id") or "")
        updated_at = str(row.get("cache_updated_at") or "")
        model = str(row.get("model") or "")
        prompt_version = str(row.get("prompt_version") or "")
        input_json_text = str(row.get("input_json_text") or "")
        response_json_text = str(row.get("response_json_text") or "")
        return (
            f"[F] [K] cache entry {i}\n"
            f"  orphan_id: {orphan_id}\n"
            f"  cache_key: {cache_key}\n"
            f"  run_id: {run_id}\n"
            f"  readiness_status: {readiness_status}\n"
            f"  readiness_reason: {readiness_reason}\n"
            f"  model: {model}\n"
            f"  prompt_version: {prompt_version}\n"
            f"  cache_updated_at: {updated_at}\n"
            f"  input_json: {input_json_text}\n"
            f"  response_json: {response_json_text}"
        )

    message = "[F] [K] orphan adjudication cache entries for this article:\n" + \
        "\n\n".join(fmt(i, row) for i, row in enumerate(rows, start=1))
    return put_line(message) ^ pure(None)

def _choose_cache_entry(rows: tuple[dict, ...]) -> Run[dict]:
    """
    Select a specific orphan cache entry to delete.
    """
    if len(rows) == 1:
        return pure(rows[0])

    listing = "\n".join(
        f"  {i}. orphan_id={str(row.get('orphan_id') or '')} "
        f"cache_key={str(row.get('cache_key') or '')}"
        for i, row in enumerate(rows, start=1)
    )
    return (
        put_line("[F] Multiple orphan cache entries found. Choose one to delete:\n" + listing)
        ^ input_number(PromptKey("delete_cache_entry_index"))
    ) >> (
        lambda selected: (
            pure(rows[selected - 1])
            if 1 <= selected <= len(rows)
            else put_line("[F] Invalid orphan cache entry selection.") >> (lambda _: _choose_cache_entry(rows))
        )
    )

def _delete_cache_entry(article: Article, row: dict) -> Run[None]:
    """
    Delete one selected orphan adjudication cache entry.
    """
    record_id = article.record_id or 0
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
        ^ sql_exec(
            SQL(
                """
                DELETE FROM orphan_adj_cache_readiness
                WHERE article_id = ?
                  AND orphan_id = ?
                  AND pass1_idempotency_key = ?;
                """
            ),
            SQLParams((record_id, String(orphan_id), String(cache_key))),
        )
        ^ put_line(
            "[F] Removed orphan adjudication cache entry "
            f"(article_id={record_id}, orphan_id={orphan_id}, cache_key={cache_key})."
        )
        ^ pure(None)
    )

def _select_apply_action(article: Article, allow_cache_delete: bool = True) -> Run[NextStep]:
    """
    Select the desired action to apply to the article.
    """
    return _cache_rows_for_article(article) >> (
        lambda cache_rows: _display_cache_rows(cache_rows)
        ^ input_desired_action(allow_cache_delete and len(cache_rows) > 0)
        >> (lambda action: _apply_action(article, action, cache_rows, allow_cache_delete))
    )

def input_desired_action(include_delete_cache: bool = False) -> Run[FixAction]:
    """
    Prompt the user to input the desired action to apply to the article.
    """
    return (
        FixAction & input_from_menu(
            MenuPrompts(ARTICLE_PROMPT_WITH_DELETE if include_delete_cache else ARTICLE_PROMPT)
        )
    )

def second_filter(article: Article) -> Run[Article]:
    """
    Process second filter action for the article.
    """
    return (
        process_all_articles(Articles((article,)))
        ^ pure(article)
    )


def _apply_action(
    article: Article,
    action: FixAction,
    cache_rows: tuple[dict, ...],
    allow_cache_delete: bool,
) -> Run[NextStep]:
    """
    Apply the desired action to the article and continue.
    """
    def _post_extract_refresh() -> Run[None]:
        record_id = article.record_id or 0
        return run_except(refresh_single_article_after_extract(record_id)) >> (
            lambda res: (
                put_line(f"[F] Warning: post-[G] single-article refresh failed: {res.l}")
                if isinstance(res, Left)
                else pure(None)
            )
        )

    match action:
        case FixAction.SECOND_FILTER:
            return \
                put_line("Dispatching to second filter...") ^ \
                set_(prompt_key, String(PROMPT_KEY_STR)) ^ \
                pure(article) >> second_filter >> \
                (lambda updated: _select_apply_action(updated, allow_cache_delete))
        case FixAction.EXTRACT_INCIDENTS:
            return (
                set_(prompt_key, String(INCIDENTS_PROMPT_KEY_STR)) ^
                extract_process_all_articles(Articles((article,)))
                ^ _post_extract_refresh()
                ^ pure(article)
                >> (lambda updated: _select_apply_action(updated, allow_cache_delete))
            )
        case FixAction.DELETE_ORPHAN_CACHE:
            return (
                _choose_cache_entry(cache_rows)
                >> (lambda selected: _delete_cache_entry(article, selected))
                >> (lambda _: _select_apply_action(article, False))
            )
        case FixAction.QUIT:
            return pure(NextStep.QUIT)
        case FixAction.CONTINUE:
            return fix_article()
        case FixAction.MAIN_MENU:
            return pure(NextStep.CONTINUE)


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
    # Inject the specific GPT models and desired prompts into the namespace
    #   of a local environment and then proceed
    return with_models(
        GPT_MODELS | INCIDENTS_GPT_MODELS,
        with_namespace(
            Namespace("fix"),
            to_prompts(FIX_PROMPTS | GPT_PROMPTS | INCIDENTS_GPT_PROMPTS),
            select_fix_article()
        )
    )
