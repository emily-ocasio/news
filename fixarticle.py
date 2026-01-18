"""
Monadic controller for reviewing and making changes to a single article
"""
from enum import Enum
from calculations import single_article_sql, latest_gptresults_sql
from pymonad import Run, Namespace, with_namespace, to_prompts, put_line, \
    pure, input_number, PromptKey, sql_query, SQL, SQLParams, throw, \
    ErrorPayload, set_, with_models, String, bind_first, Tuple, GPTUsage, \
        GPTReasoning, Maybe, Just, _Nothing, gpt_usage_reasoning_from_rows
from appstate import prompt_key
from article import Article, Articles, ArticleAppError
from gpt_filtering import GPT_PROMPTS, GPT_MODELS, \
    PROMPT_KEY_STR, process_all_articles
from menuprompts import NextStep, MenuPrompts, MenuChoice, input_from_menu
from incidents import process_all_articles as extract_process_all_articles, \
    GPT_PROMPTS as INCIDENTS_GPT_PROMPTS, PROMPT_KEY_STR as INCIDENTS_PROMPT_KEY_STR, \
    GPT_MODELS as INCIDENTS_GPT_MODELS

FIX_PROMPTS: dict[str, str | tuple[str,]] = {
    "record_id": "Please enter the record ID of the article you want to fix: ",
}
ARTICLE_PROMPT = ("Apply [S]econd filter via GPT",
                  "Extract incident via [G]PT",
                  "[C]ontinue to select another article",
                  "Go back to [M]ain menu")

class FixAction(Enum):
    """
    Possible actions that can be applied to an article in the fix article workflow.
    """
    SECOND_FILTER = MenuChoice('S')
    EXTRACT_INCIDENTS = MenuChoice('G')
    CONTINUE = MenuChoice('C')
    MAIN_MENU = MenuChoice('M')
    QUIT = MenuChoice('Q')

def input_record_id() -> Run[int]:
    """
    Prompt the user to input a record ID.
    """
    def check_if_zero(record_id: int) -> Run[int]:
        if record_id <= 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(record_id)
    return \
        input_number(PromptKey('record_id')) >> check_if_zero

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
        sql_query(SQL(latest_gptresults_sql()), SQLParams((record_id,)))) \
        >> after_query

def _select_apply_action(article: Article) -> Run[NextStep]:
    """
    Select the desired action to apply to the article.
    """
    return (
        input_desired_action()
        >> bind_first(_apply_action, article)
    )

def input_desired_action() -> Run[FixAction]:
    """
    Prompt the user to input the desired action to apply to the article.
    """
    return (
        FixAction & input_from_menu(MenuPrompts(ARTICLE_PROMPT))
    )

def second_filter(article: Article) -> Run[Article]:
    """
    Process second filter action for the article.
    """
    return (
        process_all_articles(Articles((article,)))
        ^ pure(article)
    )


def _apply_action(article: Article, action: FixAction) -> Run[NextStep]:
    """
    Apply the desired action to the article and continue.
    """
    match action:
        case FixAction.SECOND_FILTER:
            return \
                put_line("Dispatching to second filter...") ^ \
                set_(prompt_key, String(PROMPT_KEY_STR)) ^ \
                pure(article) >> second_filter >> \
                _select_apply_action
        case FixAction.EXTRACT_INCIDENTS:
            return (
                set_(prompt_key, String(INCIDENTS_PROMPT_KEY_STR)) ^
                extract_process_all_articles(Articles((article,)))
                ^ pure(article)
                >> _select_apply_action
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
