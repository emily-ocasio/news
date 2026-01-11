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
    Unit
)
from menuprompts import NextStep
from article import Articles, ArticleAppError, from_rows
from unnamed_match import export_final_victim_entities_excel

SPECIAL_ADD_PROMPTS: dict[str, str | tuple[str,]] = {
    "entity_id": "Enter the victim entity ID: ",
    "special_code": "Enter the special case code (e.g., 'Hanafi'): ",
}


def input_entity_id() -> Run[String]:
    """
    Prompt the user to input the entity ID.
    """
    return input_with_prompt(PromptKey("entity_id"))


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

def update_entity(articles: Articles) -> Run[Unit]:
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
        input_entity_id()
        >> validate_entity_exists
        >> update_current_article_ids
        ^ export_final_victim_entities_excel()
    )

def add_special_articles_to_entity() -> Run[NextStep]:
    """
    Main controller for adding special articles to an entity.
    """
    return (
        get_special_articles()
        >> update_entity
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
