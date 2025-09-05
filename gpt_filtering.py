"""
Secondary filtering of articles using GPtT
"""

from collections.abc import Callable
from typing import cast
from datetime import datetime
import json

from appstate import user_name
from pymonad import Run, with_namespace, to_prompts, Namespace, EnvKey, \
    PromptKey, pure, put_line, sql_query, SQL, SQLParams, Either, Right, \
    GPTModel, with_models, response_with_gpt_prompt, foldm_either_loop_bind, \
    to_gpt_tuple, response_message, to_json, rethrow, from_either, sql_exec, \
    GPTResponseTuple, GPTFullResponse, String, input_number, throw, \
    ErrorPayload, Tuple, resolve_prompt_template, GPTPromptTemplate, wal, ask, \
    view, bind_first
from menuprompts import NextStep
from article import Article, Articles, ArticleAppError, from_rows
from calculations import articles_to_filter_sql, gpt_homicide_class_sql, \
    gpt_victims_sql, insert_gptresults_sql
from state import WashingtonPostArticleAnalysis

GPT_PROMPTS: dict[str, str | tuple[str,]] = {
    "filternumber": "Enter number of articles to further filter via GPT: ",
    "homicide_filter": (
        "pmpt_68b39d0434d88190b0cffa9020bf4d9f0812d04c667840da",),
    "homicide_filter_dc": (
        "pmpt_68b74e67a7348196b249502a0b15992c0d9f2d89c372f7a0",),
    "combined_filter_dc": (
        "pmpt_68b86cea7d9c8190a6c87d22bdeed028041d76441e5cbcee",),
    "analysis_filter_dc": (
        'pmpt_68b8a1beb47081959598ebf1f4a9048804c14be36aa3f9a3',)
}
GPT_MODELS = {
    EnvKey("filter"): GPTModel.GPT_5_NANO
}
FormatType = WashingtonPostArticleAnalysis
PROMPT_KEY_STR = "analysis_filter_dc"
MODEL_KEY_STR = "filter"

print_gpt_response = bind_first(response_message, \
        lambda parsed_output: cast(FormatType, parsed_output).result_str)

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

def filter_article(article: Article) -> Run[GPTFullResponse]:
    """
    Filter a single article using GPT
    """
    variables = variables_dict(article)
    return \
        to_gpt_tuple & response_with_gpt_prompt(
            PromptKey(PROMPT_KEY_STR),
            variables,
            FormatType,
            EnvKey(MODEL_KEY_STR)
        )
type NewGptClass = String | None
type ArticleClassTuple = Tuple[Article, NewGptClass]

def save_article_class(article_class_t: ArticleClassTuple,
                       resp_t: GPTResponseTuple) \
    -> Run[GPTResponseTuple]:
    """
    Save the article classification to the database.
    """
    record_id = article_class_t.fst.record_id
    gpt_class = article_class_t.snd
    return \
        sql_exec(SQL(gpt_homicide_class_sql()),
            SQLParams((gpt_class, record_id))) ^ \
        put_line(f"New GPT class saved: {gpt_class}\n") ^ \
        pure(resp_t)


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
    print(cast(FormatType, resp_t.parsed.output).incidents_json)
    return \
        sql_exec(SQL(gpt_victims_sql()),
            SQLParams((
                String(cast(FormatType, resp_t.parsed.output).incidents_json),
                record_id
            ))) ^ \
        put_line("Saved new GPT Incident JSON:\n") ^ \
        pure(resp_t)

def save_filtered_article(article: Article, resp_t: GPTResponseTuple) \
    -> Run[GPTFullResponse]:
    """
    Save and refresh the article information.
    """
    gpt_class = Article.new_gpt_class(resp_t.parsed.output)
    save_class = bind_first(save_article_class, Tuple(article, gpt_class))
    save_new_json = bind_first(save_json, Tuple(article, gpt_class))
    return \
        save_class(resp_t) >> save_new_json >> (lambda resp_t: \
        pure(Right(resp_t))
        )

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
        ask() >> (lambda env: \
        resolve_prompt_template(env, PromptKey(PROMPT_KEY_STR)) >> \
        (lambda _prompt_template: wal( \
            (prompt_template:=cast(GPTPromptTemplate, _prompt_template)), \
            (prompt_id:=prompt_template.id),
            (prompt_version:=prompt_template.version),
        sql_exec(SQL(insert_gptresults_sql()),
            SQLParams((
                record_id,
                String(user),
                timestamp,
                String(PROMPT_KEY_STR),
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
                pure(Right(resp_t))

def save_gpt_fn(article: Article) -> GPTFullKreisli:
    """
    Adapter to plug save_gpt_result into the monadic chain via from_either
    """
    bind_save = bind_first(save_gpt_result, article)
    return bind_first(from_either, bind_save)

def process_append(articles: Articles, article: Article) \
    -> Run[Either[str, Articles]]:
    """
    Filter the next article and append to accumulated list
    of processed articles
    """
    save_article= save_article_fn(article)
    save_gpt = save_gpt_fn(article)
    return \
        put_line(f"Processing article {article}...\n") ^ \
        filter_article(article) >> \
        print_gpt_response >> \
        save_article >> \
        save_gpt ^ \
        pure(Right(Articles.snoc(articles, article)))

def process_articles(articles: Articles) -> Run[Either]:
    """
    Process the articles to be filtered.
    """
    return \
        foldm_either_loop_bind(articles, Articles(()), process_append) >> \
        rethrow

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _second_filter() -> Run[NextStep]:
        return \
            input_number_to_filter() >> \
            retrieve_articles >> \
            process_articles ^ \
            put_line("All articles filtered.\n") ^ \
            pure(NextStep.CONTINUE)

    return with_models(GPT_MODELS,
            with_namespace(Namespace("gpt"), to_prompts(GPT_PROMPTS),
                _second_filter()))
