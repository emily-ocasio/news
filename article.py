"""
Article class and related functions
"""
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from sqlite3 import Row
from typing import cast

from pydantic import BaseModel

from calculations import display_article, camel_to_snake
from pymonad import Array, String
from state import WashingtonPostArticleAnalysis, Article_Classification, \
    Homicide_Classification

class ArticleAppError(str, Enum):
    """
    Enumeration of possible errors in article processing.
    """
    NO_ARTICLE_FOUND = "No article found with that record ID."
    MULTIPLE_ARTICLES_FOUND = "Unexpected result - more than one article found."
    USER_ABORT = ""


@dataclass(frozen=True)
class Article:
    """
    Represents a single newspaper article.
    """
    row: Row
    current: int = 0
    total: int = 0
    record_id: int | None = None
    pub_date: str | None = None
    gpt_class: str | None = None
    auto_class: str | None = None
    title: str | None = None
    full_text: str | None = None

    def __str__(self) -> str:
        display, _ = display_article(self.total, self.current, self.row, ())
        return display

    def __post_init__(self):
        for col in self.row.keys():
            snake = camel_to_snake(col)
            if snake in (f.name for f in fields(self)):
                current_val = getattr(self, snake)
                # only overwrite when the field is currently None
                if current_val is None:
                    object.__setattr__(self, snake, self.row[col])

    @property
    def full_date(self) -> str:
        """
        Return the publication date including the date of the week
        """
        if self.pub_date:
            date = datetime.strptime(self.pub_date, "%Y%m%d")
            return date.strftime('%A %B %-d, %Y')
        return "Unknown"

    @classmethod
    def new_gpt_class(cls, homicide_class: BaseModel) \
        -> String | None:
        """
        Updates the GPT classification for the article.
        """
        gpt_class: str | None = None
        _homicide_class = cast(WashingtonPostArticleAnalysis, homicide_class)
        match _homicide_class.article_classification, \
            _homicide_class.homicide_classification:
            case (None, None):
                gpt_class = None
            case (None, _):
                gpt_class = "ERR_NONE"
            case (Article_Classification.NO_HOMICIDE_IN_ARTICLE, _):
                gpt_class = "N_NOHOM"
            case (Article_Classification.HOMICIDE_BEFORE_1977, _):
                gpt_class = "E"
            case (Article_Classification.HOMICIDES_OUTSIDE_WASHINGTON_DC, _):
                gpt_class = "O"
            case (Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                  None):
                gpt_class = "ERR_M_NONE"
            case (Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                  Homicide_Classification.VEHICULAR_HOMICIDE):
                gpt_class = "N_VEH"
            case (Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                  Homicide_Classification.FICTIONAL_HOMICIDE):
                gpt_class = "N_FIC"
            case (Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                  Homicide_Classification.MILITARY_KILLINGS):
                gpt_class = "N_MIL"
            case (Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                  Homicide_Classification.OTHER_ACTUAL_HOMICIDE):
                gpt_class = "M"
            case _, _:
                gpt_class = "ERR_OTHER"
        return None if not gpt_class else String(gpt_class)

def from_row(row: Row, current: int = 0, total: int = 0) -> Article:
    """
    Converts a SQLite row to an Article instance.
    """
    return Article( \
        row=row, \
        current=current, \
        total=total \
    )

class Articles(Array[Article]):
    """
    Represents a collection of newspaper articles.
    """
    def __repr__(self) -> str:
        return "\n_____________\n\n".join(repr(article) for article in self)

def from_rows(rows: Array[Row]) -> Articles:
    """
    Converts a list of SQLite rows to a list of Article instances.
    """
    return Articles(tuple(from_row(row, i, len(rows)) \
                          for i, row in enumerate(rows)))
