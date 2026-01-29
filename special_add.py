"""
Controller for adding special case articles to a victim entity.
"""

from pymonad import (
    Run,
    with_namespace,
    to_prompts,
    Namespace,
    pure,
    put_line,
    sql_query,
    SQL,
    SQLParams,
    throw,
    ErrorPayload,
    PromptKey,
    sql_exec,
    String,
    with_duckdb,
    input_with_prompt,
    bind_first,
    Unit,
    InputPrompt
)
from menuprompts import NextStep
from article import Articles, ArticleAppError, from_rows
from unnamed_match import export_final_victim_entities_excel

SPECIAL_ADD_PROMPTS: dict[str, str | tuple[str,]] = {
    "entity_id": "Enter the victim entity ID: ",
    "special_code": "Enter the special case code (e.g., 'Hanafi'): ",
}


# def input_entity_id() -> Run[String]:
#     """
#     Prompt the user to input the entity ID.
#     """
#     return input_with_prompt(PromptKey("entity_id"))


def input_special_code() -> Run[String]:
    """
    Prompt the user to input the special case code.
    """

    def validate_code(code: String) -> Run[String]:
        if not code:
            return throw(
                ErrorPayload(
                    "Special case code cannot be empty.", ArticleAppError.USER_ABORT
                )
            )
        return pure(String(code.strip()))

    return input_with_prompt(PromptKey("special_code")) >> validate_code


def retrieve_special_articles(code: String) -> Run[Articles]:
    """
    Retrieve articles with the special case class.
    """
    sp_code = f"SP_{code.upper()}"
    return from_rows & sql_query(
        SQL(
            "SELECT * FROM articles WHERE Dataset = 'CLASS_WP' AND gptClass = ? ORDER BY PubDate"
        ),
        SQLParams((String(sp_code),)),
    )


def validate_special_articles(articles: Articles) -> Run[Articles]:
    """
    Check if there are special articles.
    """
    if len(articles) == 0:
        return throw(
            ErrorPayload(
                "No articles found for special case code.",
                ArticleAppError.NO_ARTICLE_FOUND,
            )
        )
    return pure(articles)


def validate_entity_exists(entity_id: String) -> Run[String]:
    """
    Check if the entity exists in victim_entity_reps_new.
    """
    return sql_query(
        SQL(
            "SELECT COUNT(*) AS cnt FROM victim_entity_reps_new WHERE victim_entity_id = ?"
        ),
        SQLParams((String(entity_id),)),
    ) >> (
        lambda rows: (
            throw(
                ErrorPayload(
                    f"Entity {entity_id} not found.", ArticleAppError.NO_ARTICLE_FOUND
                )
            )
            if rows[0]["cnt"] == 0
            else pure(entity_id)
        )
    )

def ensure_special_defaults_table() -> Run[Unit]:
    """
    Ensure the DuckDB defaults table exists.
    """
    return sql_exec(
        SQL("""--sql
            CREATE TABLE IF NOT EXISTS special_case_defaults (
                special_code TEXT PRIMARY KEY,
                victim_entity_id TEXT,
                updated_at TIMESTAMP
            );
        """)
    )

def get_default_entity_id(code: String) -> Run[str | None]:
    """
    Look up a stored entity id for a special code.
    """
    return sql_query(
        SQL("""--sql
            SELECT victim_entity_id
            FROM special_case_defaults
            WHERE special_code = ?
        """),
        SQLParams((String(code.upper()),)),
    ) >> (
        lambda rows: pure(
            str(rows[0]["victim_entity_id"]).strip() if rows else None
        )
    )

def store_default_entity_id(code: String, entity_id: String) -> Run[Unit]:
    """
    Store or update the default entity id for a special code.
    """
    return (
        sql_exec(
            SQL("""--sql
                DELETE FROM special_case_defaults WHERE special_code = ?
            """),
            SQLParams((String(code.upper()),)),
        )
        ^ sql_exec(
            SQL("""--sql
                INSERT INTO special_case_defaults (
                    special_code, victim_entity_id, updated_at
                ) VALUES (?, ?, CURRENT_TIMESTAMP)
            """),
            SQLParams((String(code.upper()), String(entity_id))),
        )
    )

def input_entity_id_with_default(code: String) -> Run[String]:
    """
    Prompt for entity id, using stored default if present.
    """
    def prompt_with_default(default_id: str | None) -> Run[String]:
        prompt = (
            InputPrompt(f"Enter the victim entity ID [{default_id}]: ")
            if default_id
            else InputPrompt("Enter the victim entity ID: ")
        )
        def resolve_entity_id(raw: String) -> Run[String]:
            entered = str(raw).strip()
            if entered:
                return pure(String(entered))
            if default_id:
                return pure(String(default_id))
            return throw(
                ErrorPayload(
                    "Entity ID cannot be empty.", ArticleAppError.USER_ABORT
                )
            )
        return input_with_prompt(prompt) >> resolve_entity_id

    return ensure_special_defaults_table() >> (
        lambda _: get_default_entity_id(code) >> prompt_with_default
    )


def get_current_article_ids(entity_id: String) -> Run[str]:
    """
    Get the current article_ids_csv for the entity.
    """
    return sql_query(
        SQL("""--sql
            SELECT article_ids_csv
            FROM victim_entity_reps_new WHERE victim_entity_id = ?
        """),
        SQLParams((String(entity_id),)),
    ) >> (lambda rows: pure(rows[0]["article_ids_csv"] or ""))


def merge_article_ids(new_articles: Articles, current_csv: str) -> Run[str]:
    """
    Merge existing article IDs with new ones, deduplicating.
    """
    # Note - this use of for is not purely functional, but it uses python sets
    # to achieve the desired result simply.  To make it purely functional we can 
    # something like a HashMap, but that seems overkill here.

    existing_ids = set(current_csv.split(",")) if current_csv else set()
    new_ids = {str(a.record_id) for a in new_articles}
    merged = existing_ids | new_ids
    return pure(",".join(sorted(merged)))


def update_entity_articles(entity_id: String, merged_csv: str) -> Run[None]:
    """
    Update the entity's article_ids_csv.
    """
    return sql_exec(
        SQL("""--sql
            UPDATE victim_entity_reps_new
            SET article_ids_csv = ? WHERE victim_entity_id = ?
            """),
        SQLParams((String(merged_csv), String(entity_id))),
    ) ^ put_line(f"Updated entity {entity_id} with merged article IDs: {merged_csv}")


def get_special_articles() -> Run[Articles]:
    """
    Prompt for special code and retrieve special articles.
    """
    return (
        input_special_code()
        >> retrieve_special_articles
        >> validate_special_articles
    )

def update_entity(articles: Articles, code: String) -> Run[Unit]:
    """
    Prompt for entity ID, validate and update with special articles.
    """
    def update_current_article_ids(entity_id: String) -> Run[None]:
        return (
            get_current_article_ids(entity_id)
            >> bind_first(merge_article_ids, articles)
            >> bind_first(update_entity_articles, entity_id)
        )
    return with_duckdb(
        input_entity_id_with_default(code)
        >> validate_entity_exists
        >> (lambda entity_id: store_default_entity_id(code, entity_id) ^ \
            update_current_article_ids(entity_id))
        ^ export_final_victim_entities_excel()
    )

def add_special_articles_to_entity() -> Run[NextStep]:
    """
    Main controller for adding special articles to an entity.
    """
    return (
        input_special_code()
        >> (lambda code: (
            retrieve_special_articles(code)
            >> validate_special_articles
            >> (lambda arts: update_entity(arts, code))
        ))
        ^ pure(NextStep.CONTINUE)
    )


def add_special_articles() -> Run[NextStep]:
    """
    Entry point for the controller.
    """
    return with_namespace(
        Namespace("special_add"),
        to_prompts(SPECIAL_ADD_PROMPTS),
        add_special_articles_to_entity(),
    )
