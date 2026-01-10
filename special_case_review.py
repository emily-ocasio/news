"""
Controller for reviewing and updating special case articles.
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
    input_with_prompt,
    PromptKey,
    sql_exec,
    String,
)
from menuprompts import NextStep, MenuPrompts, MenuChoice, input_from_menu
from article import Article, Articles, ArticleAppError, from_rows

SPECIAL_PROMPTS: dict[str, str | tuple[str,]] = {
    "special_code": "Enter the special case code (e.g., 'Hanafi'): ",
}

SPECIAL_MENU = (
    "[N]ext article",
    "[M]ark as valid homicide",
    "[B]ack to main menu",
    "[Q]uit application",
)


def input_special_code() -> Run[str]:
    """
    Prompt the user to input the special case code.
    """

    def validate_code(code: str) -> Run[str]:
        if not code.strip():
            return throw(
                ErrorPayload(
                    "Special case code cannot be empty.", ArticleAppError.USER_ABORT
                )
            )
        return pure(code.strip())

    return input_with_prompt(PromptKey("special_code")) >> (
        lambda s: validate_code(str(s))
    )


def retrieve_special_articles(code: str) -> Run[Articles]:
    """
    Retrieve articles with the special case class.
    """
    sp_code = f"SP_{code.upper()}"
    return from_rows & sql_query(
        SQL("SELECT * FROM articles WHERE gptClass = ? ORDER BY PubDate"),
            SQLParams((String(sp_code),))
    )


def validate_articles(articles: Articles) -> Run[Articles]:
    """
    Validate that there are articles to review.
    """
    if len(articles) == 0:
        return throw(
            ErrorPayload(
                "No articles found for special case code.",
                ArticleAppError.NO_ARTICLE_FOUND,
            )
        )
    return pure(articles)


def display_article(article: Article) -> Run[Article]:
    """
    Display the article details.
    """
    return put_line(f"\n\nSpecial case article:\n {article}") ^ pure(article)


def get_user_choice() -> Run[MenuChoice]:
    """
    Get the user's choice for the article.
    """
    return input_from_menu(MenuPrompts(SPECIAL_MENU))


def mark_as_homicide(article: Article) -> Run[NextStep]:
    """
    Mark the article as a valid homicide and return to main menu.
    """
    return (
        sql_exec(
            SQL("UPDATE articles SET gptClass = 'M_PRELIM' WHERE RecordId = ?"),
            SQLParams((article.record_id,)),
        )
        ^ put_line(f"Article {article.record_id} marked as valid homicide (M_PRELIM).")
        ^ pure(NextStep.CONTINUE)
    )


def handle_choice(choice: MenuChoice, article: Article) -> Run[NextStep]:
    """
    Handle the user's choice.
    """
    match choice.upper():
        case "N":
            return pure(NextStep.CONTINUE)  # Continue to next
        case "M":
            return mark_as_homicide(article)
        case "B":
            return pure(NextStep.CONTINUE)  # Back to menu
        case "Q":
            return pure(NextStep.QUIT)
        case _:
            return put_line("Invalid choice, try again.") ^ get_user_choice() >> (
                lambda c: handle_choice(c, article)
            )


def review_loop(articles: Articles, index: int) -> Run[NextStep]:
    """
    Loop through the articles for review.
    """
    if index >= len(articles):
        return put_line("All special case articles reviewed.") ^ pure(NextStep.CONTINUE)
    article = articles[index]

    def after_choice(choice: MenuChoice, next_step: NextStep) -> Run[NextStep]:
        match next_step:
            case NextStep.CONTINUE if choice.upper() == "N":
                return review_loop(articles, index + 1)
            case NextStep.CONTINUE:
                return pure(NextStep.CONTINUE)
            case NextStep.QUIT:
                return pure(NextStep.QUIT)

    return display_article(article) ^ get_user_choice() >> (
        lambda choice: handle_choice(choice, article)
        >> (lambda next_step: after_choice(choice, next_step))
    )


def review_special_cases() -> Run[NextStep]:
    """
    Main controller for reviewing special case articles.
    """
    return with_namespace(
        Namespace("special"),
        to_prompts(SPECIAL_PROMPTS),
        input_special_code()
        >> retrieve_special_articles
        >> validate_articles
        >> (lambda arts: review_loop(arts, 0)),
    )
