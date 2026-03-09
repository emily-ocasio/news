"""
Controller for GPT extraction of incident details
"""
from typing import cast

from appstate import run_timer_name, run_timer_start_perf
from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, \
    GPTModel, with_models, response_with_gpt_prompt, to_json,\
    to_gpt_tuple, rethrow, from_either, sql_exec, \
    GPTResponseTuple, GPTFullResponse, String, input_number, throw, \
    ErrorPayload, Tuple, array_sequence, Unit, unit, \
    bind_first, process_items, ProcessAcc, Array, \
    Left, Right, Either, StopRun, set_, view, monotonic_now, Maybe, Just, Nothing
from pymonad.hashmap import HashMap
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_extract_sql, \
    articles_ready_for_extract_counts_sql, gpt_victims_sql
from calculations.calc_core import elapsed_line
from gpt_filtering import render_as_failure, save_gpt_fn, \
    save_article_class, ArticleClassTuple, _to_full, \
    print_gpt_response, GPTFullKreisli
from state import WashingtonPostArticleIncidentExtraction
from validate import ArticleFailureType, ArticleFailures

GPT_PROMPTS: dict[str, str | tuple[str,]] = {
    "extractnumber": \
        "Enter number of articles to extract incident information via GPT: ",
    "extract_incidents_dc": (
        'pmpt_68c8d0edb59c8193920e0e6428d01e3a0902d4a752062094',)
}
GPT_MODELS = {
    EnvKey("extract"): GPTModel.GPT_5_MINI
}
FormatType = WashingtonPostArticleIncidentExtraction
PROMPT_KEY_STR = "extract_incidents_dc"
MODEL_KEY_STR = "extract"
CLASS_CODE_UNKNOWN = String("UNKNOWN")
CLASS_CODE_NULL = String("NULL")
RUN_TIMER_NAME = String("gpt_incidents")
type ExtractClassResult = Tuple[int, String]

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

def input_number_to_extract() -> Run[int]:
    """
    Prompt the user to input the number of articles to extract incidents from.
    """
    def check_if_zero(num: int) -> Run[int]:
        if num < 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(num)
    def after_input(num: int) -> Run[int]:
        return \
            put_line(f"Extracting incidents from {num} articles...\n") ^ \
            put_line(f"JSON schema: \n{to_json(FormatType)}") ^ \
            pure(num)
    return \
        input_number(PromptKey("extractnumber")) >> check_if_zero >> after_input

def retrieve_extract_counts() -> Run[Array]:
    """
    Retrieve counts of articles ready for extraction grouped by year.
    """
    return sql_query(SQL(articles_ready_for_extract_counts_sql()))

def display_extract_counts(rows: Array) -> Run[Unit]:
    """
    Display grouped counts before asking how many records to process.
    """
    if len(rows) == 0:
        return put_line(
            "No records ready for extraction (gptClass = M_PRELIM).\n"
        ) ^ pure(unit)
    total_ready = sum(int(row["ReadyCount"]) for row in rows)
    lines = "\n".join(
        f"{row['PubYear']}: {row['ReadyCount']}"
        for row in rows
    )
    return put_line(
        "Ready for extraction (gptClass = M_PRELIM) by year:\n"
        f"{lines}\n"
        f"Total ready for extraction (all years): {total_ready}\n"
    ) ^ pure(unit)

def retrieve_articles(num: int) -> Run[Articles]:
    """
    Retrieve the articles to be filtered.
    """
    return \
        from_rows & \
            sql_query(SQL(articles_to_extract_sql()),
                      SQLParams((num,)))

def variables_dict(article: Article) -> dict[str, str | None]:
    """
    Create a variables dictionary for the GPT prompt from the article.
    """
    return {
        "article_title": article.title,
        "article_text": article.full_text,
        "article_date": article.full_date
    }

def extract_article(article: Article) -> Run[GPTFullResponse]:
    """
    Extract incident information from a single article using GPT
    """
    variables = variables_dict(article)
    return (
        #view(prompt_key) >> (lambda pk: \
        to_gpt_tuple & response_with_gpt_prompt(
            PromptKey(PROMPT_KEY_STR),
            variables,
            FormatType,
            EnvKey(MODEL_KEY_STR),
            effort="medium",
            stream=False  ## stream not working for now
        )
    )

def save_json(article_class_t: ArticleClassTuple,
              resp_t: GPTResponseTuple) \
    -> Run[GPTResponseTuple]:
    """
    Save the incident JSON to the database if homicide class is 'M'
    """
    record_id = article_class_t.fst.record_id
    gpt_class = article_class_t.snd
    if gpt_class != 'M':
        return pure(resp_t)
    return \
        sql_exec(SQL(gpt_victims_sql()),
            SQLParams((
                String(cast(FormatType, resp_t.parsed.output).incidents_json),
                record_id
            ))) ^ \
        put_line("Saved new GPT Incident JSON.\n") ^ \
        pure(resp_t)

def save_extracted_article(article: Article, resp_t: GPTResponseTuple) \
    -> Run[GPTFullResponse]:
    """
    Save and refresh the article information.
    """
    gpt_class = Article.extracted_gpt_class(resp_t.parsed.output)
    save_new_json = bind_first(save_json, Tuple(article, gpt_class))
    return \
        save_article_class(Tuple(article, gpt_class)) ^ \
        pure(resp_t) >> \
        save_new_json >> \
        _to_full

def save_article_fn(article: Article) -> GPTFullKreisli:
    """
    Return a monadic function that saves the extracted article.
    """
    # bind_extracted is typed as (GPTResponseTuple) -> Run[GPTFullResponse]
    bind_extracted = bind_first(save_extracted_article, article)
    # from_either lifts it to (GPTFullResponse) -> Run[GPTFullResponse]
    return bind_first(from_either, bind_extracted)

def extract_single_article(article: Article) -> Run[ExtractClassResult]:
    """
    Extract incident info from a single article,
    returning (record_id, gptClass) on success.
    """
    save_gpt = save_gpt_fn(article, PromptKey(PROMPT_KEY_STR))
    record_id = article.record_id or 0
    def _on_response(gpt_full: GPTFullResponse) -> Run[ExtractClassResult]:
        match gpt_full:
            case Left():
                return rethrow(gpt_full)
            case Right(resp_t):
                gpt_class = Article.extracted_gpt_class(resp_t.parsed.output)
                class_code = CLASS_CODE_NULL if gpt_class is None else gpt_class
                save_new_json = bind_first(save_json, Tuple(article, gpt_class))
                return \
                    save_article_class(Tuple(article, gpt_class)) ^ \
                    pure(resp_t) >> \
                    save_new_json >> \
                    _to_full >> \
                    save_gpt >> \
                    rethrow ^ \
                    pure(Tuple(record_id, class_code))
    return \
        put_line(f"Extracting incident data from article {article}...\n") ^ \
        extract_article(article) >> \
        print_gpt_response >> \
        _on_response

def count_classes(results: Array[ExtractClassResult]) -> HashMap[String, int]:
    """
    Count class codes from DTO results.
    """
    def step(counts: HashMap[String, int],
             result: ExtractClassResult) -> HashMap[String, int]:
        code = result.snd if len(str(result.snd)) > 0 else CLASS_CODE_UNKNOWN
        current = counts.get(code)
        current_value = current if current is not None else 0
        return counts.set(code, current_value + 1)
    return results.foldl(step, HashMap.empty())

def render_class_counts(counts: HashMap[String, int]) -> String:
    """
    Render class count map as sorted lines.
    """
    if len(counts) == 0:
        return String("  (none)")
    lines = tuple(
        f"  {k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))
    )
    return String("\n".join(lines))

def render_extract_summary(
    *,
    stopped: bool,
    gpt_counts: HashMap[String, int],
    gpt_processed_total: int,
    uncaught_failures_total: int,
    not_processed_due_to_stop: int,
    elapsed_display: Maybe[String] = Nothing,
) -> String:
    """
    Render final summary for extraction run.
    """
    header = "Extract run summary (stopped early)" if stopped else "Extract run summary"
    summary = String(
        f"{header}\n"
        "GPT-processed class counts:\n"
        f"  Total processed: {gpt_processed_total}\n"
        f"{render_class_counts(gpt_counts)}\n"
        "Failures:\n"
        f"  Uncaught exceptions: {uncaught_failures_total}\n"
        "Not processed due to stop (GPT-eligible queue):\n"
        f"  {not_processed_due_to_stop}"
    )
    match elapsed_display:
        case Just(display):
            return String(f"{summary}\n{display}")
        case _:
            return summary

def display_uncaught_exceptions(failures: Array[ArticleFailures]) -> Run[Unit]:
    """
    Display uncaught exceptions from article failures.
    """
    if failures.length == 0:
        return pure(unit)
    def display_article_failure(failures: ArticleFailures) -> Run[Unit]:
        return \
            put_line(f"Article {failures.item.record_id} " \
                     f"failed due to {failures.details[0].s}\n") ^ \
            pure(unit)
    displays = display_article_failure & failures
    return array_sequence(displays) ^ pure(unit)

def after_processing(process_acc: ProcessAcc[Article, ExtractClassResult]) \
    -> Run[NextStep]:
    """
    Handle the result after processing all articles.
    """
    def is_run_error(article_failures: ArticleFailures) -> bool:
        return article_failures.details.length > 0 and \
            article_failures.details[0].type \
                == ArticleFailureType.UNCAUGHT_EXCEPTION
    run_failures = process_acc.failures.filter(is_run_error)
    return read_elapsed_display(RUN_TIMER_NAME) >> (lambda elapsed_display:
        (lambda summary:
            put_line("Processing completed with "
            f"{process_acc.failures.length} articles "
            "failing validation.\n") ^
            put_line(f"{run_failures.length} articles "
            "failed due to uncaught exceptions.\n") ^
            display_uncaught_exceptions(run_failures) ^
            put_line(summary) ^
            pure(NextStep.CONTINUE)
            if run_failures.length > 0 else
            put_line("All articles processed:\n") ^
            put_line(summary) ^
            pure(NextStep.CONTINUE)
        )(
            render_extract_summary(
                stopped=False,
                gpt_counts=count_classes(process_acc.results),
                gpt_processed_total=process_acc.results.length,
                uncaught_failures_total=run_failures.length,
                not_processed_due_to_stop=0,
                elapsed_display=elapsed_display
            )
        )
    )

def after_processing_either(articles: Articles,
                            result: Either[
                                StopRun[Article, ExtractClassResult],
                                ProcessAcc[Article, ExtractClassResult]
                            ]) -> Run[NextStep]:
    match result:
        case Left(stop):
            processed = stop.acc.processed
            total = len(articles)
            def is_run_error(article_failures: ArticleFailures) -> bool:
                return article_failures.details.length > 0 and \
                    article_failures.details[0].type \
                        == ArticleFailureType.UNCAUGHT_EXCEPTION
            failures = stop.acc.failures.filter(is_run_error).length
            return read_elapsed_display(RUN_TIMER_NAME) >> (lambda elapsed_display:
                (lambda summary:
                    put_line(
                        "Extraction stopped by user after "
                        f"{processed} of {total} articles.\n"
                    ) ^ \
                    put_line(
                        f"{failures} article(s) recorded failures before stop.\n"
                    ) ^ \
                    put_line(summary) ^ \
                    pure(NextStep.CONTINUE)
                )(
                    render_extract_summary(
                        stopped=True,
                        gpt_counts=count_classes(stop.acc.results),
                        gpt_processed_total=stop.acc.results.length,
                        uncaught_failures_total=failures,
                        not_processed_due_to_stop=max(0, len(articles) - processed),
                        elapsed_display=elapsed_display
                    )
                )
            )
        case Right(v_process):
            return after_processing(v_process)
    raise RuntimeError("Unreachable Either branch")

def process_all_articles(articles: Articles) -> Run[NextStep]:
    """
    Process all articles for which to extract incident information
    using applicative validation.
    """
    return process_items(
        render=render_as_failure,
        happy=extract_single_article,
        items=articles
    ) >> (lambda result: after_processing_either(articles, result))

def gpt_incidents() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _extract() -> Run[NextStep]:
        return (
            (start_run_timer(RUN_TIMER_NAME)
            ^ retrieve_extract_counts())
            >> display_extract_counts
            ^ input_number_to_extract()
            >> retrieve_articles
            >> process_all_articles
        )

    return (
        with_models(
            GPT_MODELS,
            with_namespace(
                Namespace("gpt"),
                to_prompts(GPT_PROMPTS),
                _extract()
            )
        )
    )
