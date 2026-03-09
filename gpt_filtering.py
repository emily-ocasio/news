"""
Secondary filtering of articles using GPT
"""
from collections.abc import Callable
from typing import cast
from datetime import datetime
import json
import re

from appstate import user_name, run_timer_name, run_timer_start_perf
from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, Right, \
    Left, Either, StopRun, GPTModel, with_models, response_with_gpt_prompt, \
    to_gpt_tuple, response_message, to_json, rethrow, from_either, sql_exec, \
    GPTResponseTuple, GPTFullResponse, String, input_number, throw, \
    ErrorPayload, Tuple, resolve_prompt_template, GPTPromptTemplate, wal, ask, \
    view, bind_first, validate_all_pure, ValidationAcc, process_items, ProcessAcc, \
    Array, Unit, unit, V, FailureDetail, FailureDetails, array_sequence, set_, \
    monotonic_now, Maybe, Just, Nothing
from pymonad.hashmap import HashMap
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_filter_sql, \
    articles_ready_for_filter_counts_sql, gpt_homicide_class_sql, \
    insert_gptresults_sql
from calculations.calc_core import elapsed_line
from state import WashingtonPostArticleHomicideClassification
from validate import ArticleValidator, ArticleFailureDetail, \
    ArticleFailureType, SpecialCaseTerm, ArticleErrInfo, ArticleFailures

GPT_PROMPTS: dict[str, str | tuple[str,]] = {
    "filternumber": "Enter number of articles to further filter via GPT: ",
    "homicide_filter": (
        "pmpt_68b39d0434d88190b0cffa9020bf4d9f0812d04c667840da",),
    "homicide_filter_dc": (
        "pmpt_68b74e67a7348196b249502a0b15992c0d9f2d89c372f7a0",),
    "combined_filter_dc": (
        "pmpt_68b86cea7d9c8190a6c87d22bdeed028041d76441e5cbcee",),
    "analysis_filter_dc": (
        'pmpt_68b8a1beb47081959598ebf1f4a9048804c14be36aa3f9a3',),
    "classify_only_filter_dc": (
        'pmpt_68c8cb74d6e48193afd2925b0ae7c1d60247458288f5c631',)
}
GPT_MODELS = {
    EnvKey("filter"): GPTModel.GPT_5_NANO
}
FormatType = WashingtonPostArticleHomicideClassification
PROMPT_KEY_STR = "classify_only_filter_dc"
MODEL_KEY_STR = "filter"
SPECIAL_CASE_TERMS = ('Hanafi','Urgo')
CLASS_CODE_UNKNOWN = String("UNKNOWN")
CLASS_CODE_NULL = String("NULL")
RUN_TIMER_NAME = String("second_filter")

print_gpt_response = bind_first(response_message, \
        lambda parsed_output: cast(FormatType, parsed_output).result_str)

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

def _to_full(resp_t: GPTResponseTuple) -> Run[GPTFullResponse]:
    out: GPTFullResponse = Right(resp_t)
    return pure(out)

def input_number_to_filter() -> Run[int]:
    """
    Prompt the user to input the number of articles to filter.
    """
    def check_if_zero(num: int) -> Run[int]:
        if num < 0:
            return throw(ErrorPayload("", ArticleAppError.USER_ABORT))
        return pure(num)
    def after_input(num: int) -> Run[int]:
        return \
            put_line(f"Filtering {num} articles...\n") ^ \
            put_line(f"JSON schema: \n{to_json(FormatType)}") ^ \
            pure(num)
    return \
        input_number(PromptKey("filternumber")) >> check_if_zero >> after_input

def retrieve_filter_counts() -> Run[Array]:
    """
    Retrieve counts of articles ready for filtering grouped by year.
    """
    return sql_query(SQL(articles_ready_for_filter_counts_sql()))

def display_filter_counts(rows: Array) -> Run[Unit]:
    """
    Display grouped counts before asking how many records to process.
    """
    if len(rows) == 0:
        return put_line(
            "No records ready for filtering (gptClass IS NULL).\n"
        ) ^ pure(unit)
    total_ready = sum(int(row["ReadyCount"]) for row in rows)
    lines = "\n".join(
        f"{row['PubYear']}: {row['ReadyCount']}"
        for row in rows
    )
    return put_line(
        "Ready for filtering (AutoClass = M, gptClass IS NULL) by year:\n"
        f"{lines}\n"
        f"Total ready for filtering (all years): {total_ready}\n"
    ) ^ pure(unit)

def retrieve_articles(num: int) -> Run[Articles]:
    """
    Retrieve the articles to be filtered.
    """
    return \
        from_rows & \
            sql_query(SQL(articles_to_filter_sql()),
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
def special_case_validators() -> Array[ArticleValidator]:
    """
    Create an array of validators for special case terms.
    """
    terms = SpecialCaseTerm & Array(SPECIAL_CASE_TERMS)
    def validator_from_term(term: SpecialCaseTerm) -> ArticleValidator:
        def validator(article: Article) -> V[Array[FailureDetail], Unit]:
            text: str = (article.title or '') + ' ' + (article.full_text or '')
            pattern = fr"\b{re.escape(term)}"
            if re.search(pattern, text, re.IGNORECASE):
                detail = ArticleFailureDetail(
                    type=ArticleFailureType.CONTAINS_SPECIAL_CASE,
                    s=ArticleErrInfo(term)
                )
                return V.invalid(Array((detail,)))
            return V.pure(unit)
        return validator
    return validator_from_term & terms

def filter_article(article: Article) -> Run[GPTFullResponse]:
    """
    Filter a single article using GPT
    """
    variables = variables_dict(article)
    return (
        #view(prompt_key) >> (lambda pk: \
        to_gpt_tuple & response_with_gpt_prompt(
            PromptKey(PROMPT_KEY_STR),
            variables,
            FormatType,
            EnvKey(MODEL_KEY_STR)
        )
    )

type NewGptClass = String | None
type ArticleClassTuple = Tuple[Article, NewGptClass]
type ClassResult = Tuple[int, String]
def save_article_class(article_class_t: ArticleClassTuple) \
    -> Run[Unit]:
    """
    Save the article classification to the database.
    """
    record_id = article_class_t.fst.record_id
    gpt_class = article_class_t.snd
    return \
        sql_exec(SQL(gpt_homicide_class_sql()),
            SQLParams((gpt_class, record_id))) ^ \
        put_line(f"New GPT class saved: {gpt_class}\n") ^ \
        pure(unit)

def save_filtered_article(article: Article, resp_t: GPTResponseTuple) \
    -> Run[GPTFullResponse]:
    """
    Save and refresh the article information.
    """
    gpt_class = Article.new_gpt_class(resp_t.parsed.output)
    #save_new_json = bind_first(save_json, Tuple(article, gpt_class))
    return \
        save_article_class(Tuple(article, gpt_class)) ^ \
        pure(resp_t) >> \
        _to_full

type GPTFullKreisli = Callable[[GPTFullResponse], Run[GPTFullResponse]]

def save_article_fn(article: Article) -> GPTFullKreisli:
    """
    Return a monadic function that saves the filtered article.
    """
    # bind_filtered is typed as (GPTResponseTuple) -> Run[GPTFullResponse]
    bind_filtered = bind_first(save_filtered_article, article)
    # from_either lifts it to (GPTFullResponse) -> Run[GPTFullResponse]
    return bind_first(from_either, bind_filtered)

def save_gpt_fn(article: Article, prompt_key: PromptKey) -> GPTFullKreisli:
    """
    Adapter to plug save_gpt_result into the monadic chain via from_either
    """
    def save_gpt_result(resp_t: GPTResponseTuple) \
        -> Run[GPTFullResponse]:
        """
        Insert a row into gptResults for this article/filter run.
        """
        timestamp = String(datetime.now().isoformat())
        record_id = article.record_id or 0
        model = String(resp_t.parsed.usage.model_used.value \
            if resp_t.parsed.usage.model_used else '')
        format_type_name = String(str(FormatType))
        variables = variables_dict(article)
        variables_json = String(json.dumps(variables, default=str, indent=2))
        parsed = resp_t.parsed.output
        output_json = String(parsed.model_dump_json(indent=2))
        usage = resp_t.parsed.usage
        total_input_tokens = usage.input_tokens
        cached_input_tokens = usage.cached_tokens
        total_output_tokens = usage.output_tokens
        reasoning_tokens = usage.reasoning_tokens
        reasoning = resp_t.parsed.reasoning
        cost = resp_t.parsed.usage.cost()

        return \
            view(user_name) >> (lambda user: \
            ask() >> (lambda env: \
            resolve_prompt_template(env, prompt_key) >> \
            (lambda _prompt_template: wal( \
                (prompt_template:=cast(GPTPromptTemplate, _prompt_template)), \
                (prompt_id:=prompt_template.id),
                (prompt_version:=prompt_template.version),
            sql_exec(SQL(insert_gptresults_sql()),
                SQLParams((
                    record_id,
                    String(user),
                    timestamp,
                    String(prompt_key),
                    String(prompt_id),
                    String(prompt_version) if prompt_version is not None else None,
                    variables_json,
                    model,
                    format_type_name,
                    output_json,
                    reasoning,
                    total_input_tokens,
                    cached_input_tokens,
                    total_output_tokens,
                    reasoning_tokens,
                    cost
                ))))))) ^ \
                    put_line("Saved GPT result to gptResults.\n") ^ \
                    _to_full(resp_t)

    return bind_first(from_either, save_gpt_result)

def filter_single_article(article: Article) -> Run[ClassResult]:
    """
    Filter a single article and return (record_id, gptClass) on success.
    """
    save_gpt = save_gpt_fn(article, PromptKey(PROMPT_KEY_STR))
    record_id = article.record_id or 0
    def _on_response(gpt_full: GPTFullResponse) -> Run[ClassResult]:
        match gpt_full:
            case Left():
                return rethrow(gpt_full)
            case Right(resp_t):
                gpt_class = Article.new_gpt_class(resp_t.parsed.output)
                class_code = CLASS_CODE_NULL if gpt_class is None else gpt_class
                return \
                    save_article_class(Tuple(article, gpt_class)) ^ \
                    save_gpt(gpt_full) ^ \
                    pure(Tuple(record_id, class_code))
    return \
        put_line(f"Processing article {article}...\n") ^ \
        filter_article(article) >> \
        print_gpt_response >> \
        _on_response

def is_special_case_detail(detail: FailureDetail) -> bool:
    """
    Predicate a failure detail is a special case.
    """
    return detail.type == ArticleFailureType.CONTAINS_SPECIAL_CASE

def count_classes(results: Array[ClassResult]) -> HashMap[String, int]:
    """
    Count class codes from DTO results.
    """
    def step(counts: HashMap[String, int], result: ClassResult) -> HashMap[String, int]:
        code = result.snd if len(str(result.snd)) > 0 else CLASS_CODE_UNKNOWN
        current = counts.get(code)
        current_value = current if current is not None else 0
        return counts.set(code, current_value + 1)
    return results.foldl(step, HashMap.empty())

def get_count(counts: HashMap[String, int], code: String) -> int:
    """
    Read a class count from an immutable HashMap, defaulting to zero.
    """
    current = counts.get(code)
    return current if current is not None else 0

def special_case_class_code(af: ArticleFailures) -> String:
    """
    Derive SP_* class from detected special-case failure details.
    """
    special_case_details = af.details.filter(is_special_case_detail)
    special_case_terms = (lambda d: d.s) & special_case_details
    def to_upper(s: String) -> String:
        return String(s.upper())
    terms = to_upper & special_case_terms
    return String("SP_" + "_".join(terms))

def render_class_counts(counts: HashMap[String, int]) -> String:
    """
    Render class count map as sorted lines.
    """
    if len(counts) == 0:
        return String("  (none)")
    lines = tuple(f"  {k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))
    return String("\n".join(lines))

def render_filter_summary(
    *,
    stopped: bool,
    special_counts: HashMap[String, int],
    special_total: int,
    gpt_counts: HashMap[String, int],
    gpt_processed_total: int,
    uncaught_failures_total: int,
    not_processed_due_to_stop: int,
    elapsed_display: Maybe[String] = Nothing,
) -> String:
    """
    Render final summary for GPT filtering run.
    """
    header = "Filter run summary (stopped early)" if stopped else "Filter run summary"
    summary = String(
        f"{header}\n"
        "Special-case detected (validation):\n"
        f"  Total: {special_total}\n"
        f"{render_class_counts(special_counts)}\n"
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

def display_special_case_failures(special_failures: Array[ArticleFailures]) \
    -> Run[Unit]:
    """
    Display special-case findings during pure-validation phase.
    """
    if special_failures.length == 0:
        return pure(unit)
    def display_failure(af: ArticleFailures) -> Run[Unit]:
        special_case_details = af.details.filter(is_special_case_detail)
        special_case_terms = (lambda d: d.s) & special_case_details
        terms_text = ", ".join(str(s) for s in special_case_terms)
        return \
            put_line(f"Special case article:\n{af.item}") ^ \
            put_line(f"Article is a special case: {terms_text}\n") ^ \
            pure(unit)
    return array_sequence(display_failure & special_failures) ^ pure(unit)

def process_special_cases(special_failures: Array[ArticleFailures]) \
    -> Run[Either[StopRun[ArticleFailures, ClassResult],
                  ProcessAcc[ArticleFailures, ClassResult]]]:
    """
    Process special case article failures.
    """
    def process_all_special_cases() \
        -> Run[Either[StopRun[ArticleFailures, ClassResult],
                      ProcessAcc[ArticleFailures, ClassResult]]]:
        def happy_path(af: ArticleFailures) -> Run[ClassResult]:
            gpt_class = special_case_class_code(af)
            record_id = af.item.record_id or 0
            return \
                put_line(f"Special case article id {record_id} "\
                    f"title: {af.item.title}\n") ^ \
                put_line("Special case article assigned class "\
                    f"{gpt_class}\n") ^ \
                save_article_class(Tuple(af.item, gpt_class)) ^ \
                pure(Tuple(record_id, gpt_class))
        def render(err: ErrorPayload) -> FailureDetails:
            return Array((FailureDetail(
                type=ArticleFailureType.UNCAUGHT_EXCEPTION,
                s=ArticleErrInfo(f"Exception: {err}")
            ),))
        return process_items(
            render=render,
            happy=happy_path,
            items=special_failures
        )
    if special_failures.length == 0:
        return pure(Right.pure(ProcessAcc(Array(()), Array(()), 0)))
    return \
        put_line(f"{special_failures.length} " \
                "special case article(s) skipped GPT filtering.\n") ^ \
        process_all_special_cases()

def after_processing(validation_acc: ValidationAcc[Article],
                     process_acc: ProcessAcc[Article, ClassResult]) \
    -> Run[NextStep]:
    """
    Handle the result after processing all articles.
    """
    def is_run_error(article_failures: ArticleFailures) -> bool:
        return article_failures.details.length > 0 and \
            article_failures.details[0].type \
                == ArticleFailureType.UNCAUGHT_EXCEPTION
    def is_special_case(article_failures: ArticleFailures) -> bool:
        return \
            article_failures.details.filter(is_special_case_detail).length > 0
    failures = validation_acc.failures.append(process_acc.failures)
    run_failures = process_acc.failures.filter(is_run_error)
    special_case_failures = validation_acc.failures.filter(is_special_case)
    gpt_counts = count_classes(process_acc.results)
    special_counts_detected = count_classes(
        special_case_failures.map(
            lambda af: Tuple(af.item.record_id or 0, special_case_class_code(af))
        )
    )
    def _after_special(
        special_result: Either[StopRun[ArticleFailures, ClassResult],
                               ProcessAcc[ArticleFailures, ClassResult]]
    ) -> Run[NextStep]:
        return read_elapsed_display(RUN_TIMER_NAME) >> (lambda elapsed_display:
            _after_special_with_elapsed(special_result, elapsed_display)
        )

    def _after_special_with_elapsed(
        special_result: Either[StopRun[ArticleFailures, ClassResult],
                               ProcessAcc[ArticleFailures, ClassResult]],
        elapsed_display: Maybe[String]
    ) -> Run[NextStep]:
        def is_special_run_error(failure_item) -> bool:
            return failure_item.details.length > 0 and \
                failure_item.details[0].type == ArticleFailureType.UNCAUGHT_EXCEPTION
        special_run_failures = 0
        if isinstance(special_result, Left):
            return \
                put_line("Special case processing stopped by user.\n") ^ \
                put_line(render_filter_summary(
                    stopped=False,
                    special_counts=special_counts_detected,
                    special_total=special_case_failures.length,
                    gpt_counts=gpt_counts,
                    gpt_processed_total=process_acc.results.length,
                    uncaught_failures_total=run_failures.length,
                    not_processed_due_to_stop=0,
                    elapsed_display=elapsed_display
                )) ^ \
                pure(NextStep.CONTINUE)
        special_run_failures = special_result.r.failures.filter(is_special_run_error).length
        total_uncaught = run_failures.length + special_run_failures
        summary2 = render_filter_summary(
            stopped=False,
            special_counts=special_counts_detected,
            special_total=special_case_failures.length,
            gpt_counts=gpt_counts,
            gpt_processed_total=process_acc.results.length,
            uncaught_failures_total=total_uncaught,
            not_processed_due_to_stop=0,
            elapsed_display=elapsed_display
        )
        if failures.length > 0:
            return \
                put_line("Processing completed with " \
                f"{failures.length} articles " \
                "failing validation.\n") ^ \
                display_special_case_failures(special_case_failures) ^ \
                put_line(f"{total_uncaught} articles " \
                "failed due to uncaught exceptions.\n") ^ \
                put_line(summary2) ^ \
                pure(NextStep.CONTINUE)
        return \
            put_line("All articles processed:\n") ^ \
            put_line(summary2) ^ \
            pure(NextStep.CONTINUE)

    return process_special_cases(special_case_failures) >> _after_special

def render_as_failure(err: ErrorPayload) -> FailureDetails:
    """
    Render an ErrorPayload as an Array[FailureDetail]
    """
    return Array((FailureDetail(
        type=ArticleFailureType.UNCAUGHT_EXCEPTION,
        s=ArticleErrInfo(f"Exception: {err}")
    ),))

def after_processing_either(
        articles: Articles,
        validation_acc: ValidationAcc[Article],
        result: Either[
            StopRun[Article, ClassResult],
            ProcessAcc[Article, ClassResult]
        ]) -> Run[NextStep]:
    """
    Handle the result after processing all articles, including stop processing.
    """
    match result:
        case Left(stop):
            processed = stop.acc.processed
            total = len(articles)
            failures = stop.acc.failures.length
            special_case_failures = validation_acc.failures.filter(
                lambda af: af.details.filter(is_special_case_detail).length > 0
            )
            return read_elapsed_display(RUN_TIMER_NAME) >> (lambda elapsed_display:
                (lambda summary:
                    put_line(
                        "Processing stopped by user after "
                        f"{processed} of {total} articles.\n"
                    ) ^ \
                    put_line(
                        f"{failures} article(s) recorded failures before stop.\n"
                    ) ^ \
                    put_line(summary) ^ \
                    pure(NextStep.CONTINUE)
                )(
                    render_filter_summary(
                        stopped=True,
                        special_counts=count_classes(
                            special_case_failures.map(
                                lambda af: Tuple(af.item.record_id or 0,
                                                 special_case_class_code(af))
                            )
                        ),
                        special_total=special_case_failures.length,
                        gpt_counts=count_classes(stop.acc.results),
                        gpt_processed_total=stop.acc.results.length,
                        uncaught_failures_total=failures,
                        not_processed_due_to_stop=max(
                            0,
                            validation_acc.valid_items.length - processed
                        ),
                        elapsed_display=elapsed_display
                    )
                )
            )
        case Right(process_acc):
            return after_processing(validation_acc, process_acc)
    raise RuntimeError("Unreachable Either branch")

def process_all_articles(articles: Articles) -> Run[NextStep]:
    """
    Process all articles to be filtered using pure validation
    followed by monadic processing.
    """
    validation_acc: ValidationAcc[Article] = validate_all_pure(
        special_case_validators(),
        articles
    )
    return process_items(
        render=render_as_failure,
        happy=filter_single_article,
        items=validation_acc.valid_items
    ) >> (lambda result: after_processing_either(
        articles,
        validation_acc,
        result
    ))

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _second_filter() -> Run[NextStep]:
        return (
            (start_run_timer(RUN_TIMER_NAME)
            ^ retrieve_filter_counts())
            >> display_filter_counts
            ^ input_number_to_filter()
            >> retrieve_articles
            >> process_all_articles
        )

    return (
        with_models(
            GPT_MODELS,
            with_namespace(
                Namespace("gpt"),
                to_prompts(GPT_PROMPTS),
                _second_filter()
            )
        )
    )
