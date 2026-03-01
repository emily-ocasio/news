"""
First filtering of articles using automatic regex classification
"""
from appstate import run_timer_name, run_timer_start_perf
from pymonad import Run, with_namespace, to_prompts, Namespace, PromptKey, \
    pure, put_line, sql_query, SQL, SQLParams, sql_exec, input_number, throw, \
    ErrorPayload, process_items, ProcessAcc, Array, \
    String, FailureDetail, Left, Right, Either, StopRun, Tuple, \
    HashMap, Unit, unit, set_, view, monotonic_now, Maybe, Just, Nothing
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_classify_sql, classify_sql, cleanup_sql, \
    dates_ready_for_autoclassify_counts_sql
from calculations.calc_core import classify, elapsed_line
from validate import ArticleFailureType

FIRST_FILTER_PROMPTS: dict[str, str | tuple[str,]] = {
    "classifydays": "Enter number of days to auto-classify: ",
}
CLASS_CODE_M = String("M")
CLASS_CODE_N = String("N")
CLASS_CODE_O = String("O")
CLASS_CODE_UNKNOWN = String("UNKNOWN")
CLASS_CODES = Array((CLASS_CODE_M, CLASS_CODE_N, CLASS_CODE_O, CLASS_CODE_UNKNOWN))
RUN_TIMER_NAME = String("first_filter")
type AutoClassResult = Tuple[int, String]

def start_run_timer(run_name: String) -> Run[Unit]:
    """
    Start/replace a named run timer in AppState.
    """
    return (set_(run_timer_name, run_name) ^ monotonic_now()) >> (lambda now:
        set_(run_timer_start_perf, now) ^ pure(unit)
    )

def read_elapsed_display(expected_run_name: String) -> Run[Maybe[String]]:
    """
    Read elapsed display for the expected run timer if available.
    """
    def _just_elapsed(now: float, start: float) -> Run[Maybe[String]]:
        maybe_value: Maybe[String] = Just(String(elapsed_line(now - start)))
        return pure(maybe_value)

    def _from_start(timer_name: str, start: float | None) -> Run[Maybe[String]]:
        if str(timer_name) != str(expected_run_name) or start is None:
            return pure(Nothing)
        return monotonic_now() >> (lambda now: _just_elapsed(now, start))
    return view(run_timer_name) >> (lambda timer_name:
        view(run_timer_start_perf) >> (lambda start:
            _from_start(timer_name, start)
        )
    )


def retrieve_autoclassify_date_counts() -> Run[Array]:
    """
    Retrieve counts of distinct dates ready for auto-classification by year.
    """
    return sql_query(SQL(dates_ready_for_autoclassify_counts_sql()))


def display_autoclassify_date_counts(rows: Array) -> Run[Unit]:
    """
    Display grouped date counts before asking how many days to process.
    """
    if len(rows) == 0:
        return put_line(
            "No dates ready for auto-classification "
            "(Dataset = NOCLASS_WP, Complete = 0).\n"
        ) ^ pure(unit)
    lines = "\n".join(
        f"{row['PubYear']}: {row['ReadyCount']} distinct date(s)"
        for row in rows
    )
    total_days = sum(int(row["ReadyCount"]) for row in rows)
    return put_line(
        "Auto-classify date availability by year "
        "(Dataset = NOCLASS_WP, Complete = 0):\n"
        f"{lines}\n"
        f"Total days: {total_days}\n"
    ) ^ pure(unit)

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

def count_auto_classes(results: Array[AutoClassResult]) -> HashMap[String, int]:
    """
    Count class codes from processed class DTO results.
    """
    def step(counts: HashMap[String, int], result: AutoClassResult) -> HashMap[String, int]:
        code = normalize_class_code(str(result.snd))
        return increment_count(counts, code)
    empty_counts: HashMap[String, int] = HashMap.empty()
    return results.foldl(step, empty_counts)

def render_summary(counts: HashMap[String, int],
                   processed: int,
                   failures: int,
                   stopped: bool,
                   elapsed_display: Maybe[String] = Nothing) -> String:
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
    base_lines = (header,) + count_lines + unknown_line + (
        f"  Processed: {processed}",
        f"  Failures: {failures}",
    )
    match elapsed_display:
        case Just(display):
            elapsed_lines: tuple[str, ...] = (str(display),)
        case _:
            elapsed_lines = tuple()
    return String("\n".join(base_lines + elapsed_lines))

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

def classify_single_article(article: Article) -> Run[AutoClassResult]:
    """
    Classify a single article using regex and return a class DTO on success.
    """
    # Classify using the pure function from calc_core
    auto_class = String(classify(article.row))
    record_id = article.record_id or 0
    return \
        put_line(f"\n{article}\n" if auto_class == String("M") else "") ^ \
        put_line(f"Classifying article {record_id} as {auto_class}...") ^ \
        sql_exec(SQL(classify_sql()),
            SQLParams((auto_class, record_id))) ^ \
        put_line(f"Saved auto-classification: {auto_class}\n") ^ \
        pure(Tuple(record_id, auto_class))

def render_as_failure(err: ErrorPayload) -> Array[FailureDetail]:
    """
    Render an ErrorPayload as an Array[FailureDetail]
    """
    return Array((FailureDetail(
        type=ArticleFailureType.UNCAUGHT_EXCEPTION,
        s=String(f"Exception: {err}")
    ),))

def after_processing(process_acc: ProcessAcc[Article, AutoClassResult]) -> Run[NextStep]:
    """
    Handle the result after processing all articles.
    """
    failures = process_acc.failures.length
    return read_elapsed_display(RUN_TIMER_NAME) >> (lambda elapsed_display:
        (lambda summary:
            put_line("Processing completed with "
                     f"{failures} articles "
                     "failing validation.\n") ^
            put_line(summary) ^
            pure(NextStep.CONTINUE)
            if failures > 0 else
            put_line("All articles processed:\n") ^
            sql_exec(SQL(cleanup_sql())) ^
            put_line("[A] Dates cleanup applied.\n") ^
            put_line(summary) ^
            pure(NextStep.CONTINUE)
        )(
            render_summary(
                count_auto_classes(process_acc.results),
                process_acc.processed,
                failures,
                stopped=False,
                elapsed_display=elapsed_display
            )
        )
    )

def after_processing_either(articles: Articles,
                            result: Either[
                                StopRun[Article, AutoClassResult],
                                ProcessAcc[Article, AutoClassResult]
                            ]) -> Run[NextStep]:
    match result:
        case Left(stop):
            processed = stop.acc.processed
            total = len(articles)
            failures = stop.acc.failures.length
            return read_elapsed_display(RUN_TIMER_NAME) >> (lambda elapsed_display:
                (lambda summary:
                    put_line(
                        "Processing stopped by user after "
                        f"{processed} of {total} articles.\n"
                    ) ^
                    put_line(
                        f"{failures} article(s) recorded failures before stop.\n"
                    ) ^
                    put_line(summary) ^
                    pure(NextStep.CONTINUE)
                )(
                    render_summary(
                        count_auto_classes(stop.acc.results),
                        processed,
                        failures,
                        stopped=True,
                        elapsed_display=elapsed_display
                    )
                )
            )
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
            (start_run_timer(RUN_TIMER_NAME)
            ^ retrieve_autoclassify_date_counts())
            >> display_autoclassify_date_counts
            ^ input_number_of_days_to_classify()
            >> retrieve_articles
            >> _count_articles
            >> process_all_articles
        )

    return with_namespace(Namespace("first"), to_prompts(FIRST_FILTER_PROMPTS),
                _first_filter())
