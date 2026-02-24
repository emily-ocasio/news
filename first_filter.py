"""
First filtering of articles using automatic regex classification
"""
from pymonad import Run, with_namespace, to_prompts, Namespace, PromptKey, \
    pure, put_line, sql_query, SQL, SQLParams, sql_exec, input_number, throw, \
    ErrorPayload, process_all, Array, V, Valid, Invalid, \
    Validator, String, FailureDetail, Left, Right, Either, StopProcessing, \
    HashMap
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_classify_sql, classify_sql, cleanup_sql
from calculations.calc_core import classify
from validate import ArticleFailureType, ArticleFailures

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

def increment_count_by(counts: HashMap[String, int], code: String, amount: int) -> HashMap[String, int]:
    """
    Increment count for a class code by a specified amount.
    """
    current_value = get_count(counts, code)
    return counts.set(code, current_value + amount)

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

def subtract_counts(total: HashMap[String, int],
                    to_subtract: HashMap[String, int]) -> HashMap[String, int]:
    """
    Subtract one class-count map from another, clamping at zero.
    """
    def step(counts: HashMap[String, int], code: String) -> HashMap[String, int]:
        remaining = max(0, get_count(total, code) - get_count(to_subtract, code))
        return increment_count_by(counts, code, remaining)
    empty_counts: HashMap[String, int] = HashMap.empty()
    return CLASS_CODES.foldl(step, empty_counts)

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

def after_processing(articles: Articles,
                     v_process: V[Array[ArticleFailures], Array[Article]]) \
    -> Run[NextStep]:
    """
    Handle the result after processing all articles.
    """
    match v_process.validity:
        case Invalid(articles_failures):
            total_counts = count_classes(articles)
            failed_counts = count_classes(articles_failures.map(lambda failure: failure.item))
            success_counts = subtract_counts(total_counts, failed_counts)
            summary = render_summary(
                success_counts,
                len(articles),
                articles_failures.length,
                stopped=False
            )
            return \
                put_line("Processing completed with " \
                f"{articles_failures.length} articles " \
                "failing validation.\n") ^ \
                put_line(summary) ^ \
                pure(NextStep.CONTINUE)
        case Valid(processed_articles):
            summary = render_summary(
                count_classes(processed_articles),
                processed_articles.length,
                0,
                stopped=False
            )
            return \
                put_line("All articles processed:\n") ^ \
                sql_exec(SQL(cleanup_sql())) ^ \
                put_line("[A] Dates cleanup applied.\n") ^ \
                put_line(summary) ^ \
                pure(NextStep.CONTINUE)

def after_processing_either(articles: Articles,
                            result: Either[
                                StopProcessing[Article, Article],
                                V[Array[ArticleFailures], Array[Article]]
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
            return after_processing(articles, v_process)

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
