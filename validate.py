"""
Classes and functions for validating articles before processing with GPT
"""
from dataclasses import dataclass

from pymonad import String, FailureDetail, ItemFailures, \
    Validator, Array, FailureType
from article import Article

class SpecialCaseTerm(String):
    """
    A term that indicates the article is a special case
    """

class ArticleFailureType(FailureType):
    """
    Types of validation errors that can occur
    """
    CONTAINS_SPECIAL_CASE = "Contains terms related to special case"
    UNCAUGHT_EXCEPTION = "Uncaught exception during processing"

class ArticleErrInfo(String):
    """
    Specific additional information about the error
    """

@dataclass(frozen=True)
class ArticleFailureDetail(FailureDetail[Article]):
    """
    Validation error for articles
    """
    type: ArticleFailureType
    s: ArticleErrInfo

type ArticleFailures = ItemFailures[Article]
type ArticlesFailures = Array[ArticleFailures]

type ArticleValidator = Validator[Article]
