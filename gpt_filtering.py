"""
Secondary filtering of articles using GPT
"""

from calculations import articles_to_filter_sql
from pymonad import Run, with_namespace, to_prompts, Namespace, PromptKey, \
    pure, put_line, sql_query, SQL, SQLParams
from menuprompts import input_number, NextStep
from article import from_rows

GPT_PROMPTS = {
    "filternumber": "Enter number of articles to further filter via GPT: "
}

def second_filter() -> Run[NextStep]:
    """
    Use GPT to further filter articles that are homicide related
    and in correct location
    """
    def _second_filter() -> Run[NextStep]:
        return \
            input_number(PromptKey("filternumber")) >> (lambda num:
            pure(NextStep.CONTINUE) if num <= 0
            else
            put_line(f"Filtering {num} articles...") ^
            (from_rows & sql_query(SQL(articles_to_filter_sql()),
                      SQLParams((num,)))) >> (lambda articles:
            put_line(f"Filtered articles: {articles}") ^
            pure(NextStep.CONTINUE)
            ))

    return with_namespace(Namespace("gpt"),
                          _second_filter(),
                          prompts = to_prompts(GPT_PROMPTS))
