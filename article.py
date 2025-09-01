"""
Article class and related functions
"""
from dataclasses import dataclass
from sqlite3 import Row
from calculations import display_article

from pymonad import Array

@dataclass(frozen=True)
class Article:
    """
    Represents a single newspaper article.
    """
    row: Row
    current: int = 0
    total: int = 0

    def __repr__(self) -> str:
        display, _ = display_article(self.total, self.current, self.row, ())
        return display

    @property
    def title(self) -> str:
        """
        Returns the title of the article.
        """
        return self.row['Title']

    @property
    def full_text(self) -> str:
        """
        Returns the full text of the article.
        """
        return self.row['FullText']

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
