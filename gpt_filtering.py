"""
Secondary filtering of articles using GPT
"""

from calculations import articles_to_filter_sql
from pymonad import Run, with_namespace, to_prompts, Namespace, PromptKey, \
    pure, put_line, sql_query, SQL, SQLParams, Either, Left
from menuprompts import input_number, NextStep
from article import Article, Articles, from_rows

GPT_PROMPTS: dict[str, str | tuple[str,]] = {
    "filternumber": "Enter number of articles to further filter via GPT: ",
    "homicide_filter": (
        "pmpt_68b39d0434d88190b0cffa9020bf4d9f0812d04c667840da",)
}

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _second_filter() -> Run[NextStep]:
        def _filter_next(articles: Articles, next_article: Article) \
            -> Run[Either[str, Articles]]:
            variables = {"article_title": next_article.title,
                         "article_text": next_article.full_text}
            return pure(Left("Not implemented"))
                
        return \
            input_number(PromptKey("filternumber")) >> (lambda num:
            pure(NextStep.CONTINUE) if num <= 0
            else
            put_line(f"Filtering {num} articles...") ^
            (from_rows & sql_query(SQL(articles_to_filter_sql()),
                      SQLParams((num,)))) >> (lambda articles:
            put_line(f"Articles to be filtered: {articles}") ^
            pure(NextStep.CONTINUE)
            ))

    return with_namespace(Namespace("gpt"),
                          _second_filter(),
                          prompts = to_prompts(GPT_PROMPTS))
