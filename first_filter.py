"""
First filtering of articles using automatic regex classification
"""
from pymonad import Run, with_namespace, to_prompts, Namespace, PromptKey, \
    pure, put_line, sql_query, SQL, SQLParams, sql_exec, input_number, throw, \
    ErrorPayload, process_items, ProcessAcc, Array, \
    String, FailureDetail, Left, Right, Either, StopRun, \
    HashMap
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_classify_sql, classify_sql, cleanup_sql
from calculations.calc_core import classify
from validate import ArticleFailureType

FIRST_FILTER_PROMPTS: dict[str, str | tuple[str,]] = {
    "classifydays": "Enter number of days to auto-classify: ",
}
CLASS_CODE_M = String("M")
CLASS_CODE_N = String("N")
CLASS_CODE_O = String("O")
CLASS_CODE_UNKNOWN = String("UNKNOWN")
CLASS_CODES = Array((CLASS_CODE_M, CLASS_CODE_N, CLASS_CODE_O, CLASS_CODE_UNKNOWN))

def increment_count(counts: HashMap[String, int], code: String) -> HashMap[String, int]:
    """
    Increment count for a class code in an immutable HashMap.
    """
    current = counts.get(code)
    current_value = current if current is not None else 0
    return counts.set(code, current_value + 1)

def get_count(counts: HashMap[String, int], code: String) -> int:
    """
    Read a class count from an immutable HashMap, defaulting to zero.
    """
    current = counts.get(code)
    return current if current is not None else 0

def normalize_class_code(code: str) -> String:
    """
    Normalize class code so unexpected values are grouped under UNKNOWN.
    """
    class_code = String(code)
    return class_code if class_code in CLASS_CODES else CLASS_CODE_UNKNOWN

def count_classes(articles: Array[Article]) -> HashMap[String, int]:
    """
    Count class codes for successfully processed articles using a pure fold.
    """
    def step(counts: HashMap[String, int], article: Article) -> HashMap[String, int]:
        code = normalize_class_code(classify(article.row))
        return increment_count(counts, code)
    empty_counts: HashMap[String, int] = HashMap.empty()
    return articles.foldl(step, empty_counts)

def render_summary(counts: HashMap[String, int],
                   processed: int,
                   failures: int,
                   stopped: bool) -> String:
    """
    Render run summary with class counts, processed total, and failures total.
    """
    header = ("Partial classification summary (this run):"
              if stopped else
              "Classification summary (this run):")
    count_lines = (
        f"  M: {get_count(counts, CLASS_CODE_M)}",
        f"  N: {get_count(counts, CLASS_CODE_N)}",
        f"  O: {get_count(counts, CLASS_CODE_O)}",
    )
    unknown_count = get_count(counts, CLASS_CODE_UNKNOWN)
    unknown_line = (f"  UNKNOWN: {unknown_count}",) if unknown_count > 0 else tuple()
    return String("\n".join(
        (header,) + count_lines + unknown_line + (
            f"  Processed: {processed}",
            f"  Failures: {failures}",
        )
    ))

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

def after_processing(process_acc: ProcessAcc[Article, Article]) \
    -> Run[NextStep]:
    """
    Handle the result after processing all articles.
    """
    failures = process_acc.failures.length
    summary = render_summary(
        count_classes(process_acc.results),
        process_acc.processed,
        failures,
        stopped=False
    )
    if failures > 0:
        return \
            put_line("Processing completed with " \
            f"{failures} articles " \
            "failing validation.\n") ^ \
            put_line(summary) ^ \
            pure(NextStep.CONTINUE)
    return \
        put_line("All articles processed:\n") ^ \
        sql_exec(SQL(cleanup_sql())) ^ \
        put_line("[A] Dates cleanup applied.\n") ^ \
        put_line(summary) ^ \
        pure(NextStep.CONTINUE)

def after_processing_either(articles: Articles,
                            result: Either[
                                StopRun[Article, Article],
                                ProcessAcc[Article, Article]
                            ]) -> Run[NextStep]:
    match result:
        case Left(stop):
            processed = stop.acc.processed
            total = len(articles)
            failures = stop.acc.failures.length
            summary = render_summary(
                count_classes(stop.acc.results),
                processed,
                failures,
                stopped=True
            )
            return \
                put_line(
                    "Processing stopped by user after "
                    f"{processed} of {total} articles.\n"
                ) ^ \
                put_line(
                    f"{failures} article(s) recorded failures before stop.\n"
                ) ^ \
                put_line(summary) ^ \
                pure(NextStep.CONTINUE)
        case Right(v_process):
            return after_processing(v_process)
    raise RuntimeError("Unreachable Either branch")

def process_all_articles(articles: Articles) -> Run[NextStep]:
    """
    Process all articles for auto-classification.
    """
    return process_items(
        render=render_as_failure,
        happy=classify_single_article,
        items=articles
    ) >> (lambda result: after_processing_either(articles, result))

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
