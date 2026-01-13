"""
Secondary filtering of articles using GPT
"""
from collections.abc import Callable
from typing import cast
from datetime import datetime
import json
import re

from appstate import user_name, prompt_key
from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, Right, \
    GPTModel, with_models, response_with_gpt_prompt, set_, \
    to_gpt_tuple, response_message, to_json, rethrow, from_either, sql_exec, \
    GPTResponseTuple, GPTFullResponse, String, input_number, throw, \
    ErrorPayload, Tuple, resolve_prompt_template, GPTPromptTemplate, wal, ask, \
    view, bind_first, process_all, Array, Unit, unit, V, FailureDetail, \
    ItemsFailures, FailureDetails, Valid, Invalid, Validator
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_filter_sql, gpt_homicide_class_sql, \
    insert_gptresults_sql
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

print_gpt_response = bind_first(response_message, \
        lambda parsed_output: cast(FormatType, parsed_output).result_str)

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
        def validator(article: Article) \
            -> Run[V[Array[FailureDetail], Unit]]:
            text = (article.title or '') + ' ' + (article.full_text or '')
            pattern = fr"\b{re.escape(term)}"
            if re.search(pattern, text, re.IGNORECASE):
                detail = ArticleFailureDetail(
                    type=ArticleFailureType.CONTAINS_SPECIAL_CASE,
                    s=ArticleErrInfo(term)
                )
                return \
                    put_line(f"Special case article:\n{article}") ^ \
                    put_line(f"Article is a special case: {term}\n") ^ \
                    pure(V.invalid(Array((detail,))))
            return pure(V.pure(unit))
        return validator
    return validator_from_term & terms

def filter_article(article: Article) -> Run[GPTFullResponse]:
    """
    Filter a single article using GPT
    """
    variables = variables_dict(article)
    return \
        view(prompt_key) >> (lambda pk: \
        to_gpt_tuple & response_with_gpt_prompt(
            PromptKey(pk),
            variables,
            FormatType,
            EnvKey(MODEL_KEY_STR)
        ))

type NewGptClass = String | None
type ArticleClassTuple = Tuple[Article, NewGptClass]

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

def save_gpt_result(article: Article, resp_t: GPTResponseTuple) \
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
        view(prompt_key) >> (lambda pk: \
        ask() >> (lambda env: \
        resolve_prompt_template(env, PromptKey(pk)) >> \
        (lambda _prompt_template: wal( \
            (prompt_template:=cast(GPTPromptTemplate, _prompt_template)), \
            (prompt_id:=prompt_template.id),
            (prompt_version:=prompt_template.version),
        sql_exec(SQL(insert_gptresults_sql()),
            SQLParams((
                record_id,
                String(user),
                timestamp,
                String(pk),
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
            )))))))) ^ \
                put_line("Saved GPT result to gptResults.\n") ^ \
                _to_full(resp_t)

def save_gpt_fn(article: Article) -> GPTFullKreisli:
    """
    Adapter to plug save_gpt_result into the monadic chain via from_either
    """
    bind_save = bind_first(save_gpt_result, article)
    return bind_first(from_either, bind_save)

def filter_single_article(article: Article) -> Run[Article]:
    """
    Filter a single article, returning the article on success.
    """
    save_article= save_article_fn(article)
    save_gpt = save_gpt_fn(article)
    return \
        put_line(f"Processing article {article}...\n") ^ \
        filter_article(article) >> \
        print_gpt_response >> \
        save_article >> \
        save_gpt >> \
        rethrow ^ \
        pure(article)

def is_special_case_detail(detail: FailureDetail) -> bool:
    """
    Predicate a failure detail is a special case.
    """
    return detail.type == ArticleFailureType.CONTAINS_SPECIAL_CASE

def process_special_cases(special_failures: Array[ArticleFailures]) \
    -> Run[Unit]:
    """
    Process special case article failures.
    """
    def process_all_special_cases() \
        -> Run[V[ItemsFailures[ArticleFailures], Array[Unit]]]:
        validators: Array[Validator[ArticleFailures]] = Array(())
        def happy_path(af: ArticleFailures) -> Run[Unit]:
            special_case_details = af.details.filter(is_special_case_detail)
            special_case_terms = (lambda d: d.s) & special_case_details
            def to_upper(s: String) -> String:
                return String(s.upper())
            gpt_class = \
                String("SP_" + "_".join(to_upper & special_case_terms))
            return \
                put_line(f"Special case article id {af.item.record_id} "\
                    f"title: {af.item.title}\n") ^ \
                put_line("Special case article assigned class "\
                    f"{gpt_class}\n") ^ \
                save_article_class(Tuple(af.item, gpt_class))
        def render(err: ErrorPayload) -> FailureDetails:
            return Array((FailureDetail(
                type=ArticleFailureType.UNCAUGHT_EXCEPTION,
                s=ArticleErrInfo(f"Exception: {err}")
            ),))
        return process_all(
            validators=validators,  # No validators needed
            render=render,
            happy=happy_path,
            items=special_failures
        )
    if special_failures.length == 0:
        return pure(unit)
    return \
        put_line(f"{special_failures.length} " \
                "special case article(s) skipped GPT filtering.\n") ^ \
        process_all_special_cases() ^ \
        pure(unit)

def after_processing(v_process: V[Array[ArticleFailures], Array[Article]]) \
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
    match v_process.validity:
        case Invalid(articles_failures):
            run_failures = articles_failures.filter(is_run_error)
            special_case_failures = articles_failures.filter(is_special_case)
            return \
                put_line("Processing completed with " \
                f"{articles_failures.length} articles " \
                "failing validation.\n") ^ \
                process_special_cases(special_case_failures) ^ \
                put_line(f"{run_failures.length} articles " \
                "failed due to uncaught exceptions.\n") ^ \
                pure(NextStep.CONTINUE)
        case Valid(_):
            return \
                put_line("All articles processed:\n") ^ \
                pure(NextStep.CONTINUE)

def render_as_failure(err: ErrorPayload) -> FailureDetails:
    """
    Render an ErrorPayload as an Array[FailureDetail]
    """
    return Array((FailureDetail(
        type=ArticleFailureType.UNCAUGHT_EXCEPTION,
        s=ArticleErrInfo(f"Exception: {err}")
    ),))

def process_all_articles(articles: Articles) -> Run[NextStep]:
    """
    Process all articles to be filtered using applicative validation.
    """
    return process_all(
        validators=special_case_validators(),
        render=render_as_failure,
        happy=filter_single_article,
        items=articles
    ) >> after_processing

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _second_filter() -> Run[NextStep]:
        return \
            input_number_to_filter() >> \
            retrieve_articles >> \
            process_all_articles

    return \
        set_(prompt_key, String(PROMPT_KEY_STR)) ^ \
        with_models(GPT_MODELS,
            with_namespace(Namespace("gpt"), to_prompts(GPT_PROMPTS),
                _second_filter()))
