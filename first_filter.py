"""
First filtering of articles using automatic regex classification
"""
from pymonad import Run, with_namespace, to_prompts, Namespace, PromptKey, \
    pure, put_line, sql_query, SQL, SQLParams, sql_exec, input_number, throw, \
    ErrorPayload, process_all, Array, V, Valid, Invalid, \
    Validator, String, FailureDetail
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_classify_sql, classify_sql, cleanup_sql
from calculations.calc_core import classify
from validate import ArticleFailureType, ArticleFailures

FIRST_FILTER_PROMPTS: dict[str, str | tuple[str,]] = {
    "classifydays": "Enter number of days to auto-classify: ",
}

def input_number_of_days_to_classify() -> Run[int]:
    """
    Prompt the user to input the number of days to classify.
    """
    def check_if_zero(num: int) -> Run[int]:
        if num <= 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(num)
    def after_input(num: int) -> Run[int]:
        return \
            put_line(f"Auto-classifying articles from {num} days...\n") ^ \
            pure(num)
    return (
        input_number(PromptKey("classifydays"))
        >> check_if_zero
        >> after_input
    )

def retrieve_articles(num_days: int) -> Run[Articles]:
    """
    Retrieve the articles to be classified from the specified number of days.
    """
    return \
        from_rows & \
            sql_query(SQL(articles_to_classify_sql()),
                      SQLParams((num_days,)))

def classify_single_article(article: Article) -> Run[Article]:
    """
    Classify a single article using regex, returning the article on success.
    """
    # Classify using the pure function from calc_core
    auto_class = String(classify(article.row))
    return \
        put_line(f"\n{article}\n" if auto_class == String("M") else "") ^ \
        put_line(f"Classifying article {article.record_id} as {auto_class}...") ^ \
        sql_exec(SQL(classify_sql()),
            SQLParams((auto_class, article.record_id))) ^ \
        put_line(f"Saved auto-classification: {auto_class}\n") ^ \
        pure(article)

def render_as_failure(err: ErrorPayload) -> Array[FailureDetail]:
    """
    Render an ErrorPayload as an Array[FailureDetail]
    """
    return Array((FailureDetail(
        type=ArticleFailureType.UNCAUGHT_EXCEPTION,
        s=String(f"Exception: {err}")
    ),))

def after_processing(v_process: V[Array[ArticleFailures], Array[Article]]) \
    -> Run[NextStep]:
    """
    Handle the result after processing all articles.
    """
    match v_process.validity:
        case Invalid(articles_failures):
            return \
                put_line("Processing completed with " \
                f"{articles_failures.length} articles " \
                "failing validation.\n") ^ \
                pure(NextStep.CONTINUE)
        case Valid(_):
            return \
                put_line("All articles processed:\n") ^ \
                sql_exec(SQL(cleanup_sql())) ^ \
                put_line("[A] Dates cleanup applied.\n") ^ \
                pure(NextStep.CONTINUE)

def process_all_articles(articles: Articles) -> Run[NextStep]:
    """
    Process all articles for auto-classification using applicative validation.
    """
    validators: Array[Validator[Article]] = Array(())  # No validators needed
    return process_all(
        validators=validators,
        render=render_as_failure,
        happy=classify_single_article,
        items=articles
    ) >> after_processing

def first_filter() -> Run[NextStep]:
    """
    Use regex to automatically classify articles as homicide-related or not.
    """
    def _count_articles(articles: Articles) -> Run[Articles]:
        return (
            put_line(f"Retrieved {len(articles)} articles to classify.\n")
            ^ pure(articles)
        )

    def _first_filter() -> Run[NextStep]:
        return (
            input_number_of_days_to_classify()
            >> retrieve_articles
            >> _count_articles
            >> process_all_articles
        )

    return with_namespace(Namespace("first"), to_prompts(FIRST_FILTER_PROMPTS),
                _first_filter())
