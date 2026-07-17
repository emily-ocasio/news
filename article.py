"""
Article class and related functions
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from typing import cast, Mapping

from pydantic import BaseModel

from calculations import display_article, camel_to_snake
from pymonad import Array, String
from state import WashingtonPostArticleHomicideClassification, \
    Article_Classification, \
    Homicide_Classification, ArticleIncidentExtraction, \
    ClassificationOutcome, GPTClassificationResult, \
    NewYorkTimesArticleHomicideClassification

class ArticleAppError(str, Enum):
    """
    Enumeration of possible errors in article processing.
    """
    NO_ARTICLE_FOUND = "No article found with that record ID."
    MULTIPLE_ARTICLES_FOUND = "Unexpected result - more than one article found."
    USER_ABORT = ""


@dataclass(frozen=True)
class Article:  # pylint: disable=too-many-instance-attributes
    """
    Represents a single newspaper article.
    """
    row: Mapping
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
    def wp_classification_result(cls, homicide_class: BaseModel) \
        -> GPTClassificationResult:  # pylint: disable=too-many-return-statements
        """Map the unchanged WP response model to neutral domain semantics."""
        response = cast(WashingtonPostArticleHomicideClassification, homicide_class)
        match response.article_classification, response.homicide_classification:
            case (Article_Classification.NO_HOMICIDE_IN_ARTICLE, _):
                return GPTClassificationResult(
                    ClassificationOutcome.NO_HOMICIDE_IN_ARTICLE
                )
            case (Article_Classification.HOMICIDE_BEFORE_1977, _):
                return GPTClassificationResult(
                    ClassificationOutcome.HOMICIDE_BEFORE_TIME_SCOPE
                )
            case (Article_Classification.HOMICIDES_OUTSIDE_WASHINGTON_DC, _):
                return GPTClassificationResult(
                    ClassificationOutcome.HOMICIDES_OUTSIDE_TARGET_LOCATION
                )
            case (
                Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                Homicide_Classification.VEHICULAR_HOMICIDE,
            ):
                return GPTClassificationResult(
                    ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                    response.homicide_classification,
                )
            case (
                Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                Homicide_Classification.FICTIONAL_HOMICIDE,
            ):
                return GPTClassificationResult(
                    ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                    response.homicide_classification,
                )
            case (
                Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                Homicide_Classification.MILITARY_KILLINGS,
            ):
                return GPTClassificationResult(
                    ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                    response.homicide_classification,
                )
            case (
                Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                Homicide_Classification.OTHER_ACTUAL_HOMICIDE,
            ):
                return GPTClassificationResult(
                    ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                    response.homicide_classification,
                )
            case (
                Article_Classification.HOMICIDE_IN_WASHINGTON_DC_SINCE_1977,
                None,
            ):
                return GPTClassificationResult(
                    ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE
                )
            case _:
                raise ValueError("Unrecognized WP article classification response")

    @classmethod
    def nyt_classification_result(cls, homicide_class: BaseModel) \
        -> GPTClassificationResult:
        """Map the NYT response model to neutral domain semantics."""
        response = cast(NewYorkTimesArticleHomicideClassification, homicide_class)
        return GPTClassificationResult(
            response.article_classification,
            response.homicide_classification,
        )

    @classmethod
    def new_gpt_class(cls, classification: GPTClassificationResult) \
        -> String | None:
        """
        Updates the GPT classification for the article.
        """
        gpt_class: str
        match (
            classification.outcome,
            classification.homicide_classification,
        ):
            case (ClassificationOutcome.NO_HOMICIDE_IN_ARTICLE, _):
                gpt_class = "N_NOHOM"
            case (ClassificationOutcome.HOMICIDE_BEFORE_TIME_SCOPE, _):
                gpt_class = "E"
            case (ClassificationOutcome.HOMICIDES_OUTSIDE_TARGET_LOCATION, _):
                gpt_class = "O"
            case (
                ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                Homicide_Classification.VEHICULAR_HOMICIDE,
            ):
                gpt_class = "N_VEH"
            case (
                ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                Homicide_Classification.FICTIONAL_HOMICIDE,
            ):
                gpt_class = "N_FIC"
            case (
                ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                Homicide_Classification.MILITARY_KILLINGS,
            ):
                gpt_class = "N_MIL"
            case (
                ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                None,
            ):
                gpt_class = "ERR_M_NONE"
            case (
                ClassificationOutcome.HOMICIDE_IN_TARGET_LOCATION_IN_TIME_SCOPE,
                _,
            ):
                gpt_class = "M_PRELIM"
            case _:
                gpt_class = "ERR_OTHER"
        return None if not gpt_class else String(gpt_class)

    @classmethod
    def extracted_gpt_class(
        cls, extraction: ArticleIncidentExtraction, incident_start_year: int
    ) \
        -> String | None:
        """
        Return the extracted GPT classification for the active date scope.
        """
        if len(extraction.incidents) == 0:
            return String("N_NOINC")
        if any(
            incident.year >= incident_start_year
            for incident in extraction.incidents
        ):
            return String("M")
        return String("E")

def from_row(row: Mapping, current: int = 0, total: int = 0) -> Article:
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

    @classmethod
    def from_rows(cls, rows: Array[Mapping]) -> Articles:
        """ Creates Articles object from array of database rows """
        return from_rows(rows)

def from_rows(rows: Array[Mapping]) -> Articles:
    """
    Converts a list of SQLite rows to a list of Article instances.
    """
    return Articles(tuple(from_row(row, i, len(rows)) \
                          for i, row in enumerate(rows)))
