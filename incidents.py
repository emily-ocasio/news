"""
Controller for GPT extraction of incident details
"""
from typing import cast

from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, \
    GPTModel, with_models, response_with_gpt_prompt, to_json,\
    to_gpt_tuple, rethrow, from_either, sql_exec, \
    GPTResponseTuple, GPTFullResponse, String, input_number, throw, \
    ErrorPayload, Tuple, array_sequence, Unit, unit, \
    bind_first, process_all, Array, V, Valid, Invalid, Validator
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_extract_sql, \
    gpt_victims_sql
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

def extract_single_article(article: Article) -> Run[Article]:
    """
    Extract incident info from a single article,
    returning the article on success.
    """
    save_article= save_article_fn(article)
    save_gpt = save_gpt_fn(article, PromptKey(PROMPT_KEY_STR))
    return \
        put_line(f"Extracting incident data from article {article}...\n") ^ \
        extract_article(article) >> \
        print_gpt_response >> \
        save_article >> \
        save_gpt >> \
        rethrow ^ \
        pure(article)

def display_uncaught_exceptions(failures: Array[ArticleFailures]) -> Run[Unit]:
    """
    Display uncaught exceptions from article failures.
    """
    if failures.length == 0:
        return pure(unit)
    def display_article_failure(failures: ArticleFailures) -> Run[None]:
        return \
            put_line(f"Article {failures.item.record_id} " \
                     f"failed due to {failures.details[0].s}\n")
    displays = display_article_failure & failures
    return array_sequence(displays) ^ pure(unit)

def after_processing(v_process: V[Array[ArticleFailures], Array[Article]]) \
    -> Run[NextStep]:
    """
    Handle the result after processing all articles.
    """
    def is_run_error(article_failures: ArticleFailures) -> bool:
        return article_failures.details.length > 0 and \
            article_failures.details[0].type \
                == ArticleFailureType.UNCAUGHT_EXCEPTION
    match v_process.validity:
        case Invalid(articles_failures):
            run_failures = articles_failures.filter(is_run_error)
            return \
                put_line("Processing completed with " \
                f"{articles_failures.length} articles " \
                "failing validation.\n") ^ \
                put_line(f"{run_failures.length} articles " \
                "failed due to uncaught exceptions.\n") ^ \
                display_uncaught_exceptions(run_failures) ^ \
                pure(NextStep.CONTINUE)
        case Valid(_):
            return \
                put_line("All articles processed:\n") ^ \
                pure(NextStep.CONTINUE)

def process_all_articles(articles: Articles) -> Run[NextStep]:
    """
    Process all articles for which to extract incident information
    using applicative validation.
    """
    validators: Array[Validator[Article]] = Array(())
    return process_all(
        validators=validators,  # No validators needed
        render=render_as_failure,
        happy=extract_single_article,
        items=articles
    ) >> after_processing

def gpt_incidents() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _extract() -> Run[NextStep]:
        return \
            input_number_to_extract() >> \
            retrieve_articles >> \
            process_all_articles

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
