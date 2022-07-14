"""
Pure functions with no side effects
"""
import re
from sqlite3 import Row
from collections.abc import Iterable
from typing import Optional, Union
from functools import reduce

from flashtext import KeywordProcessor  # type: ignore
from colorama import Style
from rich.console import Console, Text
from rich.table import Table

from mass_towns import townlist
from state import Rows

absolute_roots = (
    'slain',
    'slaying',
    'murder',
    'homicide',
    'manslaughter',
)

conditional_roots = (
    'shot',
    'shooting',
    'wounded',
    'stabb',
    'kidnap',
    'strangl',
    'beat',
    'bludgeon',
    'asphyxiat'
)

death_roots = (
    'fatal',
    'dead',
    'died',
    'dies',
    'kill',
    'death'
)


def add_spaces(word: str) -> str:
    r"""
    Inserts optional spaces for regex between each letter of a word
    Input: word
    Output: Regex string of the form:
        w\s?o\s?r\s?d
    This is so the regex catches cases in which the words are arbitrarily
        separated by a space anywhere, e.g. "mur der" for "murder"
    """
    return r"\s?".join(word)


def any_root_regex_string(roots: tuple[str, ...]) -> str:
    r"""
    Generates regex string that represents match to
    any word that starts with one of the given roots
    Input: Tuple of word roots, example: ('root1', 'root2', 'root3')
    Output: Regex string in the form:
        \b(?:root1|root2|root3)\w*
    Explanation:
        \b finds strings only at the beginning of a word
        (?: ...  )  limits the effect of the "OR" operator |
                    without creating a group
        \w*  matches remaining letters of the word
    """
    rootstring = '|'.join(add_spaces(root) for root in roots)
    return fr"\b(?:{rootstring})\w*"


def any_word_regex_string(words: tuple[str, ...]) -> str:
    r"""
    Generates regex string that represents match to any
        full word/phrase of the list
    Input: Tuple of words, example: ('word1', 'word2', 'word3')
    Output: Regex string in the form:
        \b(?:word1|word2|word3)\b
    """
    rootstring = '|'.join(words)
    return fr"\b(?:{rootstring})\b"


def distance_match_regex_string(string1: str,
                                string2: str, sep: int = 300) -> str:
    """
    Generates regex string that requires two strings to
    match (in any order) separated by a maximum of sep characters
    Input: strings to match string1 and string2
    Output: Regex string of the form (assuming sep is default of 300):
        string1.{0,300}?string2|string2.{0,300}?string1
    Explanation:
        .{0,300}?  represents any 0-300 characters (? makes it non-greedy)
    """
    def dist(str1: str, str2: str) -> str:
        return fr"{str1}.{{0,{sep}}}?{str2}"
    return fr"{dist(string1, string2)}|{dist(string2, string1)}"


absolute_regex = re.compile(any_root_regex_string(absolute_roots),
                            re.IGNORECASE)
conditional_regex = re.compile(any_root_regex_string(conditional_roots),
                               re.IGNORECASE)
death_regex = re.compile(any_root_regex_string(death_roots), re.IGNORECASE)

conditional_string = distance_match_regex_string(
    any_root_regex_string(conditional_roots),
    any_root_regex_string(death_roots)
)
conditional_death_regex = re.compile(conditional_string, re.IGNORECASE)

single_word_regex = re.compile(
    r"(\S+)", re.IGNORECASE
)

town_regex = re.compile(any_word_regex_string(townlist), re.IGNORECASE)


def high_word(word: str, color=Style.BRIGHT) -> str:
    """
    Changes word or phrase to show as bold in terminal
    """
    return color + word + Style.NORMAL + "\x1b[0m"


kp = KeywordProcessor()
for keyword in townlist:
    kp.add_keyword(keyword, high_word(keyword))


def in_mass(gpe):
    """
    Returns list of matched Massachussets locations
    """
    return kp.extract_keywords(gpe)


def is_in_mass(gpe):
    """
    True is there is at least one Mass. location in text
    """
    return len(in_mass(gpe)) > 0


def unified_prompt(prompts: tuple[str,...],
                    add_quit: bool = True,
                    allow_return: bool = False,
                    width: int = 150) -> tuple[str, tuple]:
    """
    Creates a single prompt including individual choices,
        also adds an option to quit
        and option to press <Return> to continue
    Also returns list of menu letters based on letters
        inside brackets in prompts.
    If all prompts fit inside a single line (based on width parameter)
        print them separated by commas, otherwise break them into
        columns
    """
    regex = r'\[(\w)\]'
    prompt_list = (prompts
                + (("[Q]uit",) if add_quit else tuple())
                + (("<Return> to continue",) if allow_return else tuple())
    )
    full_prompt = ("Select option: "
                   + ", ".join(prompt_list)
                   + " > "
    )
    if len(full_prompt) > width:
        max_len = max(len(prompt) for prompt in prompt_list) + 4
        columns = (width - 5) // max_len
        full_prompt = ('Select option:\n'
            + '\n'.join(4*' '+ ''.join(prompt.ljust(max_len)
                                    for prompt in prompt_list[i:i+columns])
                        for i in range(0, len(prompt_list), columns))
            + "\n> ")
    return full_prompt, tuple(re.findall(regex, full_prompt))
    # return (full_prompt if len(full_prompt) < width
    #         else full_prompt.replace(', ', '\n    ')
    #         .replace(': ', ':\n    ')
    #         .replace(' > ', '\n> '), re.findall(regex, full_prompt))


def unverified_articles_sql() -> tuple[str, str]:
    """
    SQL statement to return articles not yet labeled for given date
    """
    sql = f"""
        FROM (
            {article_type_join_sql()}
            WHERE a.Pubdate = ?
            AND a.Status IS NULL
            GROUP BY a.RecordId
            HAVING GoodTypes > 0
        )
    """
    return "SELECT COUNT(*) " + sql, "SELECT * " + sql


def retrieve_types_sql() -> str:
    """
    SQL statement to return list of types for given article
    """
    return """
        SELECT e.TypeDesc desc 
        FROM articletypes t 
        JOIN articleenum e ON e.TypeId = t.TypeId 
        WHERE t.RecordId= ?
    """


def verify_article_sql() -> str:
    """
    SQL statement to update specific article with given label
    """
    return """
        UPDATE articles
        SET Status = ?
        WHERE RecordId = ?
    """


def article_type_join_sql(index: str = "") -> str:
    """
    Initial portion of SQL statement joining articles with types
        and aggregating via binary encoding all the possible types
    """
    indexed_sql = "" if not index else f"INDEXED BY {index}"
    return f"""
        SELECT 
            a.*, 
            SUM(IIF(t.TypeID IN (7,8,9,10,12,13,19,21), 0, 1)) AS GoodTypes,
            SUM(1 << t.TypeId) as BinaryTypes
        FROM articles a {indexed_sql}
        JOIN articletypes t
        ON a.RecordId = t.RecordId
    """


def verified_articles_sql() -> str:
    """
    SQL Statement to return articles with from a specified dataset and label
    """
    return article_type_join_sql() + """
        WHERE Status = ?
        AND Dataset = ?
        GROUP BY a.RecordId
    """


def all_articles_sql() -> str:
    """
    SQL statement to return all articles from a specific dataset
    """
    return article_type_join_sql() + """
        WHERE Dataset = ?
        GROUP BY a.RecordId
    """


def single_article_sql():
    """
    SQL statement to return a specific article by its recordId
    """
    return article_type_join_sql() + """
        WHERE a.RecordId = ?
        GROUP BY a.RecordId
    """


def articles_to_classify_sql():
    """
    SQL statement to return articles to auto-classify based on date priority
    """
    return article_type_join_sql() + """
        WHERE PubDate IN (
            SELECT PubDate
            FROM dates
            WHERE Complete = 0
            ORDER BY Priority
            LIMIT ?
        )
        AND a.AutoClass IS NULL
        GROUP BY a.RecordId
    """


def articles_to_assign_sql():
    """
    SQL statement to return verified articles for assignment
    """
    return article_type_join_sql() + """
         WHERE a.Dataset = "CLASS"
         AND a.Status = "M"
         AND a.AssignStatus IS NULL
         AND a.PubDate IN (
             SELECT PubDate
             FROM dates
             WHERE PubDate IN (
                 SELECT DISTINCT PubDate
                 FROM articles
                 WHERE Dataset = "CLASS"
                 AND Status = "M"
                 AND AssignStatus IS NULL
             )
             ORDER BY Priority
             LIMIT ?
         )
         GROUP BY a.RecordId
         ORDER BY a.PubDate, a.RecordId
    """


def articles_to_assign_by_year_sql():
    """
    SQL Statement to return verified articles for assignment
        Instead of priority filter by year
    """
    return article_type_join_sql() + """
        WHERE a.Dataset = 'CLASS'
        AND a.Status = 'M'
        AND a.AssignStatus IS NULL
        AND a.PubDate >= ? 
        AND a.PubDate <= ?
        GROUP BY a.RecordId
        ORDER BY a.PubDate, a.RecordId
    """


def articles_to_reclassify_sql():
    """
    SQL statement to return auto-classified articles for reclassification
    Used when given a number of dates to grab auto-classified articles
        in order to verify whether they are in fact true positives
    """
    return """
        SELECT a.*
        FROM articles a
         WHERE a.Dataset = "CLASS"
         AND a.Status IS NULL
         AND a.Autoclass = "M"
         AND a.PubDate IN (
             SELECT PubDate
             FROM dates
             WHERE PubDate IN (
                 SELECT DISTINCT PubDate
                 FROM articles
                 WHERE Dataset = "CLASS"
                 AND Status IS NULL
                 AND Autoclass = "M"
             )
             ORDER BY Priority
             LIMIT ?
         )
         ORDER BY a.PubDate, a.RecordId
    """


def articles_to_reclassify_by_year_sql() -> str:
    """
    SQL statemnet to return auto-classified articles for reclassification
        given a particular year or set of years
    Used when selecting to review auto-classified articles and particularly
        in order to separate groups and have multiple users simultaneously
        reclassifying
    """
    return """
        SELECT a.*
        FROM articles a
        WHERE a.Dataset = 'CLASS'
        AND a.Status IS NULL
        AND a.AutoClass = 'M'
        AND a.PubDate >= ?
        AND a.PubDate <= ?
        ORDER BY a.PubDate, a.RecordId
    """


def classify_sql():
    """
    SQL statement to update auto-classification of a single article
    """
    return """
        UPDATE articles
        SET AutoClass = ?,
        Dataset = "CLASS"
        WHERE RecordId = ?
    """


def assign_status_sql():
    """
    SQL statement to update assignment status for a single article
    """
    return """
        UPDATE articles
        SET AssignStatus = ?
        WHERE RecordId = ?
    """


def update_note_sql() -> str:
    """
    SQL statement to update note in specific article
    """
    return """
        UPDATE articles
        SET Notes = ?
        WHERE RecordID = ?
    """


def cleanup_sql() -> str:
    """
    SQL Statement to update dates that have been completely autoclassified
    """
    return """
        WITH datelist AS MATERIALIZED (
            SELECT DISTINCT PubDate
            FROM articles
            INDEXED BY Dataset
            WHERE Dataset = "CLASS"
            AND Pubdate NOT IN 
            (
                SELECT DISTINCT PubDate
                FROM articles
                INDEXED BY Dataset
                WHERE Dataset != "CLASS"
            )
        )
        UPDATE dates
        SET Complete = 1
        WHERE PubDate IN (SELECT PubDate FROM datelist)
    """


def homicides_by_month_sql() -> str:
    """
    SQL Statement to retrieve homicides based on year-month
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY Agency, Inc) AS n, *
        FROM view_shr
        WHERE YearMonth = ?
    """


def homicides_by_victim_sql() -> str:
    """
    SQL Statement to retrieve homicides based on victim name
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY YearMonth, Agency, Inc) AS n, *
        FROM view_shr
        WHERE Victim LIKE '%' || ? || '%'
    """


def homicides_assigned_by_article_sql() -> str:
    """
    SQL Statement to retrieve homicides already assigned
        to a specific article
    """
    return """
        SELECT ROW_NUMBER() OVER (ORDER BY Agency, Inc) AS k, *
        FROM view_shr
        WHERE Id IN
        (
            SELECT ShrId
            FROM topics
            WHERE RecordId = ?
        )
    """


def assign_homicide_victim_sql() -> str:
    """
    SQL Statement (transation) to add assignment of homicide
        to a specific article and also adjust the victim name
    """
    return """
            INSERT OR IGNORE INTO topics
            (ShrId, RecordId)
            VALUES (?, ?);
            UPDATE shr
            SET Victim = ?
            WHERE "index" = ?
    """


def assign_homicide_sql(repeat:int = 1) -> str:
    """
    SQL Statement to add assignment of homicide
        without changing victim's name
    Allows for multi-row insert (repeat number of rows inserted)
    """
    return """
        INSERT OR IGNORE INTO topics
        (ShrId, RecordId)
        VALUES """ + ' , '.join(('(?, ?)',) * repeat)


def unassign_homicide_sql() -> str:
    """
    SQL Statement to un-assign (delete from assignment list)
        a particular homicide previously assigned to an article
    """
    return """
        DELETE FROM topics
        WHERE ShrId = ?
        AND RecordId = ?
    """


def display_article(total: int,
                    current: int,
                    row: Row,
                    types: Rows,
                    limit_lines=0) -> tuple[str, tuple[str, ...]]:
    """
    Full text to display contents and metadata of one article
    """
    counter = article_counter(current, total)
    label = article_label(row)
    # lines = wrap_lines(
    #     color_text_matches(color_mass_locations(row['FullText']))
    # )
    lines = tuple(rich_text(row['FullText']).splitlines())
    limit = limit_lines - 12
    art_types = article_types(types)
    return ("\n".join(counter
                      + label
                      + art_types
                      + (lines[:limit] if limit_lines > 0 else lines))
            + '\n', lines)


def display_remaining_lines(lines, limit_lines=0) -> str:
    """
    Displays lines after limit_lines - 12
    """
    return "\n".join(lines[limit_lines-12:])


def article_counter(current: int, total: int) -> tuple[str, ...]:
    """
    Returns text showing article counter
    """
    return (f"Article {current+1} of {total}:\n",) if total > 1 else tuple()


def article_label(row: Row) -> tuple[str, ...]:
    """
    Returns label wtih article metadata
    """
    return (f"Title: {rich_text(row['Title'])}    Date: "
            f"{row['PubDate'][4:6]}/{row['PubDate'][6:8]}/{row['PubDate'][0:4]}"
            "\n",
            f"Record ID = {row['RecordId']}",
            f"Verified status = <{row['status']}>\n"
            )


def article_types(types: Rows) -> tuple[str, ...]:
    """
    Return listing of all article types
    """
    return tuple(f"Article Type: {row['desc']}" for row in types)


def article_notes(row: Row) -> str:
    """
    Return display of notes for article
    """
    notes = row['Notes']
    return f"Notes: {notes}" if notes else "No notes"


def words_in(text: str) -> Iterable[str]:
    """
    Separate text into discrete words
    """
    return text.split()


def any_match(regex: re.Pattern, text: str) -> bool:
    """
    Determines whether there is a regex pattern match
    """
    return regex.search(text) is not None


def test_all_filters(text: str) -> bool:
    """
    Determines whether text has required word matches
    """
    return (any_match(absolute_regex, text)
            or any_match(conditional_death_regex, text)
            )


def filter_text(document: Optional[str]) -> bool:
    """
    Determines whether text is nonempty and has a match
    """
    if document is None:
        return False
    return test_all_filters(document)


def rich_text(document: Optional[str]) -> str:
    """
    Apply rich text formatting
    Includes the following:
        - Color highlighting the word matches
        - Color highlighting the location matches
        - Word wrapping long text
    """
    if document is None:
        return ''
    text = Text(document)
    text.highlight_regex(absolute_regex, 'red bold')  # type: ignore
    text.highlight_regex(conditional_death_regex, 'red bold')  # type: ignore
    text.highlight_regex(town_regex, 'blue bold')  # type: ignore
    return rich_to_str(text)


def rich_to_str(text: Union[Text, Table]) -> str:
    """
    Returns directly printable string corresponding to a text
    Applies style formatting and word wrapping automatically
    """
    console = Console()
    with console.capture() as capture:
        console.print(text)
    return capture.get()


def filter_row(row: Row) -> bool:
    """
    Determines whether there are matches in either
        the title or the main text of the article
    """
    return filter_text(row['Title']) or filter_text(row['FullText'])


def partition(rows: tuple[Row, ...], filt=filter_row) -> tuple[tuple[Row, ...],
                                                               tuple[Row, ...]]:
    """
    Returns two tuples, one of the rows that match filt criteria
        (default = filter_row),
    and the other of rows that don't
    Uses reduce as a fold/accumulator
    The last argument of reduce function is tuple of two empty lists
    Each pass through reduce appends the items two either of the two lists
        depending on the value of filter_row
    Since append() returns none, then we use the x.append() or x idiom
        to return the mutated object
    """
    part: tuple = reduce(
        lambda x, y: x[not filt(y)].append(y) or x, rows, ([], []))
    return tuple(part[0]), tuple(part[1])


def mass_divide(rows: tuple[Row, ...]) -> tuple[int, int]:
    """
    Returns count of how hany articles have or do not have a match
    """
    count_mass = len(
        tuple(filter(is_in_mass, (row['FullText'] for row in rows))))
    count_nomass = len(rows) - count_mass
    return count_mass, count_nomass


def color_mass_locations(document: Optional[str]) -> Optional[str]:
    """
    Highlights Massachusetts locations in the text
    """
    if document is None:
        return None
    return kp.replace_keywords(document)


def located_articles(rows: tuple[Row, ...], mass=True) -> tuple[Row, ...]:
    """
    Returns subset of articles that are (or are not) from Massachusetts
    """
    def loc_filter(row: Row) -> bool:
        in_m = is_in_mass(row['FullText'])
        return in_m if mass else not in_m
    return tuple(filter(loc_filter, rows))


def filter_by_type(rows: tuple[Row, ...], good=True) -> tuple[Row, ...]:
    """
    Returns subset of articles that have (or do not have) good types only
    """
    def type_filter(row: Row) -> bool:
        is_good = is_good_type(row)
        return is_good if good else not is_good
    return tuple(filter(type_filter, rows))


def type_divide(rows: tuple[Row, ...]) -> tuple[int, int]:
    """
    Returns counts of articles that have good and bad types
    """
    good_count = len(filter_by_type(rows))
    return good_count, len(rows) - good_count


def classify(row: Row) -> str:
    """
    Determines auto-classification of an article
    """
    if not is_good_type(row) or not filter_row(row):
        return 'N'
    if not is_in_mass(row['FullText']):
        return 'O'
    return 'M'


def is_good_type(row: Row) -> bool:
    """
    Determine whether row is is of a 'good type'
    Takes into consideration that some articles marked as Advertisement
        may not in fact be ads
    Returns true if article has at least one of the accepted ('good') types
    OR
    One of the types is 'Advertisement' AND the word 'ad'
        is not included in the title
    The column 'BinaryTypes' has all the types assigned to this article
        binary encoded, so that each bit represents a possible type
    Advertisement is type 7, so we do a bitwise-and to 2**7 to determine
        whether article is
        categorized as advertisement
    """
    return row['GoodTypes'] > 0 or (row['BinaryTypes'] & 2**7 > 0
                                    and 'ad' not in row['Title'].lower())


def full_pred(row: Row) -> bool:
    """
    Determine wheter row is predicted to be a match using all three criteria:
    - must be of a 'good type' (e.g. no advertisements)
    - must by in Massachusetts
    - must pass the regex filters in filter_row
    Must pass all criteria to return True
    """
    return is_good_type(row) and is_in_mass(row['FullText']) and filter_row(row)


def manual_match(row: Row) -> bool:
    """
    Determine whether row was manually determined to be a match
    """
    return row['Status'] == 'M'


def confusion_matrix(rows: tuple[Row, ...]) -> tuple[tuple[Row, ...],
                                                     tuple[Row, ...],
                                                     tuple[Row, ...],
                                                     tuple[Row, ...]]:
    """
    Computes confusion matrix statistics for a set of articles
    """
    matches, no_matches = partition(rows, full_pred)
    TP, FP = partition(matches, manual_match)
    FN, TN = partition(no_matches, manual_match)
    return TP, TN, FP, FN


def statistic_summary(TP, TN, FP, FN) -> str:
    """
    Displays summary statistics
    """
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    msg = ("Matching statistics:\n"
           f"True Positives: {TP}\n"
           f"True Negatives: {TN}\n"
           f"False Positives: {FP}\n"
           f"False Negatives: {FN}\n"
           f"Sensitivity: {100*sens:.2f}%\n"
           f"Specificity: {100*spec:.2f}%"
           )
    return msg


def year_month_from_article(row: Row) -> str:
    """
    Return year and month in YYYY-MM format
    from an article PubDate
    """
    article_date = row['PubDate']
    return f"{article_date[:4]}-{article_date[4:6]}"


def homicide_table(rows: Rows) -> str:
    """
    Return formatted table of homicide info
    """
    table = Table(row_styles=['','on grey85'])
    for col in rows[0].keys():
        table.add_column(col)
    for row in rows:
        elements = tuple(str(element) for element in row)
        table.add_row(*elements)  # type: ignore
    return rich_to_str(table)


def tuple_replace(tup: tuple, index: int, value) -> tuple:
    """
    Replaces item in tuple and returns newly created tuple
    """
    temp_list = list(tup)
    temp_list[index] = value
    return tuple(temp_list)
