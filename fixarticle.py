"""
Monadic controller for reviewing and making changes to a single article
"""
from calculations import single_article_sql
from pymonad import Run, Namespace, with_namespace, to_prompts, put_line, \
    pure, input_number, PromptKey, sql_query, SQL, SQLParams, throw, \
    ErrorPayload, set_, view, with_models, String
from appstate import selected_option, prompt_key
from article import Article, Articles, ArticleAppError, from_rows
from gpt_filtering import GPT_PROMPTS, GPT_MODELS, \
    PROMPT_KEY_STR, process_all_articles
from menuprompts import NextStep, MenuPrompts, MenuChoice, input_from_menu
from incidents import process_all_articles as extract_process_all_articles, \
    GPT_PROMPTS as INCIDENTS_GPT_PROMPTS, PROMPT_KEY_STR as INCIDENTS_PROMPT_KEY_STR

FIX_PROMPTS: dict[str, str | tuple[str,]] = {
    "record_id": "Please enter the record ID of the article you want to fix: ",
}
ARTICLE_PROMPT = ("Apply [S]econd filter via GPT",
                  "Extract incident via [G]PT",
                  "[C]ontinue to select another article",
                  "Go back to [M]ain menu")

def input_record_id() -> Run[int]:
    """
    Prompt the user to input a record ID.
    """
    def check_if_zero(record_id: int) -> Run[int]:
        if record_id <= 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(record_id)
    def after_input(record_id: int) -> Run[int]:
        return \
            put_line(f"Retrieving article with record ID: {record_id}...") ^ \
            pure(record_id)
    return \
        input_number(PromptKey('record_id')) >> check_if_zero >> after_input

def retrieve_article(record_id: int) -> Run[Articles]:
    """
    Retrieve the article with the given record ID.
    """
    return \
        from_rows & \
            sql_query(SQL(single_article_sql()), SQLParams((record_id,)))

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

def display_article(article: Article) -> Run[Article]:
    """
    Display the retrieved article and continue.
    """
    return \
        put_line(f"Retrieved article:\n {article}") ^ \
        pure(article)

def input_desired_action(article) -> Run[Article]:
    """
    Prompt the user to input the desired action to apply to the article.
    """
    def show_menu() -> Run[MenuChoice]:
        return input_from_menu(MenuPrompts(ARTICLE_PROMPT))

    def validate_choice(choice: MenuChoice) -> Run[Article]:
        match x:=choice.upper():
            case 'S' | 'Q' | 'M' | 'G' |'C':
                return \
                    set_(selected_option, x) ^ \
                    pure(article)
            case _:
                return \
                    put_line("Invalid choice, please try again.") ^ \
                    show_menu() >> validate_choice
    return \
        show_menu() >> validate_choice

def second_filter(article: Article) -> Run[Article]:
    """
    Process second filter action for the article.
    """
    return (
        process_all_articles(Articles((article,)))
        ^ pure(article)
    )
    # save_article = save_article_fn(article)
    # save_gpt = save_gpt_fn(article)
    # return \
    #     filter_article(article) >> \
    #     print_gpt_response >> \
    #     save_article >> \
    #     save_gpt >> \
    #     rethrow ^ \
    #     pure(article)

def apply_action(article: Article) -> Run[NextStep]:
    """
    Apply the desired action to the article and continue.
    """
    def dispatch_action(option: str) -> Run[NextStep]:
        match option:
            case 'S':
                return \
                    put_line("Dispatching to second filter...") ^ \
                    set_(prompt_key, String(PROMPT_KEY_STR)) ^ \
                    pure(article) >> second_filter >> \
                    input_desired_action >> apply_action
            case 'G':
                return (
                    set_(prompt_key, String(INCIDENTS_PROMPT_KEY_STR)) ^
                    extract_process_all_articles(Articles((article,)))
                    ^ pure(article)
                    >> input_desired_action >> apply_action
                )
            case 'Q':
                return pure(NextStep.QUIT)
            case 'C':
                return \
                    fix_article()
            case 'M':
                return pure(NextStep.CONTINUE)
            case _:
                return \
                    throw(ErrorPayload(
                        "App error - invalid option in fix article dispatch."))
    return \
        view(selected_option) >> dispatch_action

def select_fix_article() -> Run[NextStep]:
    """
    Select an article for fixing
    """
    return \
        input_record_id() >> \
        retrieve_article >> \
        validate_retrieval >> \
        display_article >> \
        input_desired_action >> \
        apply_action


def fix_article() -> Run[NextStep]:
    """
    Fix an article by its record ID.
    """
    return set_(prompt_key, String(PROMPT_KEY_STR)) ^ \
        with_models(GPT_MODELS,
        with_namespace(Namespace("fix"),
        to_prompts(FIX_PROMPTS | GPT_PROMPTS | INCIDENTS_GPT_PROMPTS),
        select_fix_article()))
