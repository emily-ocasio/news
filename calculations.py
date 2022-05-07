import re
import textwrap
from colorama import Fore, Style
from sqlite3 import Row
from typing import Iterable, Tuple, Optional
from itertools import tee, filterfalse
from functools import reduce
from flashtext import KeywordProcessor #type: ignore
from mass_towns import townlist

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
    return fr"\s?".join(word)

def any_root_regex_string(roots: Tuple[str,...]) -> str:
    """
    Generates regex string that represents match to 
    any word that starts with one of the given roots
    Input: Tuple of word roots, example: ('root1', 'root2', 'root3')
    Output: Regex string in the form:
        \b(?:root1|root2|root3)\w*
    Explanation:
        \b finds strings only at the beginning of a word
        (?: ...  )  limits the effect of the "OR" operator | without creating a group
        \w*  matches remaining letters of the word
    """
    rootstring = '|'.join(add_spaces(root) for root in roots)
    return fr"\b(?:{rootstring})\w*"

def distance_match_regex_string(string1: str, string2: str, sep: int = 300) -> str:
    """
    Generates regex string that requires two strings to
    match (in any order) separated by a maximum of sep characters
    Input: strings to match string1 and string2
    Output: Regex string of the form (assuming sep is default of 300):
        string1.{0,300}?string2|string2.{0,300}?string1
    Explanation:
        .{0,300}?  represents any 0-300 characters (? makes it non-greedy)
    """
    def dist(s1: str, s2: str) -> str:
        return fr"{s1}.{{0,{sep}}}?{s2}"
    return fr"{dist(string1, string2)}|{dist(string2, string1)}"

absolute_regex = re.compile(any_root_regex_string(absolute_roots), re.IGNORECASE)
conditional_regex = re.compile(any_root_regex_string(conditional_roots), re.IGNORECASE)
death_regex = re.compile(any_root_regex_string(death_roots), re.IGNORECASE)

conditional_string = distance_match_regex_string(any_root_regex_string(conditional_roots), any_root_regex_string(death_roots)) 
conditional_death_regex = re.compile(conditional_string, re.IGNORECASE)

def colored_word(word: str, color = Fore.RED) -> str:
    return color + word + Fore.RESET

def high_word(word: str, color = Style.BRIGHT) -> str:
    return color + word + Style.NORMAL

kp = KeywordProcessor()
for keyword in townlist:
    kp.add_keyword(keyword, high_word(keyword))
# kp.add_keywords_from_list(list(townlist))

def in_mass(gpe):
    return kp.extract_keywords(gpe)
    
def is_in_mass(gpe):
    return len(in_mass(gpe))> 0

def unified_prompt(prompts, add_quit = True, allow_return = False):
    """
    Creates a single prompt including individual choices,
    also adds an option to quit.
    Also returns list of menu letters based on letters inside brackets in prompts.
    """
    regex = '\[(\w)\]'
    full_prompt = "Select option: " + ", ".join(prompts) + (", [Q]uit" if add_quit else "") + (", <Return> to continue" if allow_return else "") + " > "
    return full_prompt, re.findall(regex, full_prompt)

def unverified_articles_sql():
    # sql = """
    #     FROM articles a
    #     LEFT JOIN verifications v ON a.RecordId = v.RecordId
    #     WHERE PubDate = ? AND v.Status IS NULL
    # """
    sql = f"""
        FROM (
            {article_type_join_sql()}
            ON a.RecordID = t.RecordId
            WHERE a.Pubdate = ?
            AND a.Status IS NULL
            GROUP BY a.RecordId
            HAVING GoodTypes > 0
        )
    """
    return "SELECT COUNT(*) " + sql, "SELECT * " + sql

def retrieve_types_sql() -> str:
    return """
        SELECT e.TypeDesc desc 
        FROM articletypes t 
        JOIN articleenum e ON e.TypeId = t.TypeId 
        WHERE t.RecordId= ?
    """

def verify_article_sql(row: Row) -> str:
    # if row['status'] is None:
    #     return f"""
    #         INSERT INTO verifications
    #         (Status, RecordId)
    #         VALUES (?, ?)
    #     """
    # return """
    #     UPDATE verifications
    #     SET Status = ?
    #     WHERE RecordId = ?
    # """
    return """
        UPDATE articles
        SET Status = ?
        WHERE RecordId = ?
    """

def article_type_join_sql():
    return """
        SELECT 
            a.*, 
            SUM(IIF(t.TypeID IN (7,8,9,10,13,19,21), 0, 1)) AS GoodTypes,
            SUM(1 << t.TypeId) as BinaryTypes
        FROM articles a
        JOIN articletypes t
        ON a.RecordId = t.RecordId
    """

def verified_articles_sql():
    # return """
    #     SELECT *
    #     FROM verified_articles
    #     WHERE status = ?
    #     AND dataset = ?;
    # """
    return article_type_join_sql() + """
        WHERE Status = ?
        AND Dataset = ?
        GROUP BY a.RecordId
    """

def all_articles_sql():
    return article_type_join_sql() + """
        WHERE Dataset = ?
        GROUP BY a.RecordId
    """

def single_article_sql():
    return article_type_join_sql() + """
        WHERE a.RecordId = ?
        GROUP BY a.RecordId
    """

def articles_to_classify_sql():
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

def classify_sql():
    return """
        UPDATE articles
        SET AutoClass = ?,
        Status = ?,
        Dataset = "CLASS"
        WHERE RecordId = ?
    """

def display_article(total: int, current, row: Row, types) -> Tuple[str, Tuple[str,...]]:
    counter = article_counter(current, total)
    label = article_label(row)
    lines = wrap_lines(color_mass_locations(color_text_matches(row['FullText'])))
    art_types = article_types(types)
    return "\n".join(counter + label + art_types + lines[:35]) + '\n', lines

def display_remaining_lines(lines) -> str:
    return "\n".join(lines[35:])

def wrap_lines(text, width = 140) -> Tuple[str,...]:
    if not text:
        return tuple("No text")
    return tuple(textwrap.wrap(text, width))

def article_counter(current, total) -> Tuple[str,...]:
    return (f"Article {current+1} of {total}:\n",) if total>1 else tuple()

def article_label(row: Row):
    return f"Title: {color_text_matches(row['Title'])}    Date: {row['PubDate'][4:6]}/{row['PubDate'][6:8]}/{row['PubDate'][0:4]}\n", \
        f"Record ID = {row['RecordId']}", \
        f"Verified status = <{row['status']}>\n"

def article_types(types: Tuple[Row,...]) -> Tuple[str,...]:
    return tuple(f"Article Type: {row['desc']}" for row in types)

def words_in(text: str) -> Iterable[str]:
    return text.split()

def any_match(regex: re.Pattern, text: str) -> bool:
    return regex.search(text) is not None

def test_all_filters(text: str) -> bool:
    return any_match(absolute_regex,text) or any_match(conditional_death_regex,text)
    #return any_match(absolute_regex,text) or (any_match(conditional_regex,text) and any_match(death_regex,text))

def filter_text(document: Optional[str]) -> bool:
    if document is None:
        return False
    return test_all_filters(document)

def color_text_matches(document: Optional[str]) -> Optional[str]:
    if document is None:
       return None
    text = document
    text = absolute_regex.sub(lambda match: colored_word(match.group(0)), text)
    # text = conditional_regex.sub(lambda match: colored_word(match.group(0)), text)
    # text = death_regex.sub(lambda match: colored_word(match.group(0)), text)
    text = conditional_death_regex.sub(lambda match: colored_word(match.group(0)), text)
    return text

def filter_row(row: Row) -> bool:
    return filter_text(row['Title']) or filter_text(row['FullText'])

def partition_old(rows: Tuple[Row,...]) -> Tuple[Tuple[Row,...], Tuple[Row,...]]:
    match, nomatch = tee(rows)
    return tuple(filter(filter_row, match)), tuple(filterfalse(filter_row, nomatch))

def partition(rows: Tuple[Row,...], filt = filter_row) -> Tuple[Tuple[Row,...], Tuple[Row,...]]:
    """
    Returns two tuples, one of the rows that match filt criteria (default = filter_row), 
    and the other of rows that don't
    Uses reduce as a fold/accumulator
    The last argument of reduce function is tuple of two empty lists
    Each pass through reduce appends the items two either of the two lists depending on the value of filter_row
    Since append() returns none, then we use the x.append() or x idiom to return the mutated object
    """
    part: tuple = reduce(lambda x,y: x[not filt(y)].append(y) or x, rows, ([],[]))
    return tuple(part[0]), tuple(part[1])

def mass_divide(rows: Tuple[Row,...]) -> Tuple[int, int]:
    count_mass = len(tuple(filter(is_in_mass, (row['FullText'] for row in rows))))
    count_nomass = len(rows) - count_mass
    return count_mass, count_nomass

def color_mass_locations(document: Optional[str]) -> Optional[str]:
    if document is None:
        return None
    return kp.replace_keywords(document)

def located_articles(rows: Tuple[Row,...], mass = True) -> Tuple[Row,...]:
    def loc_filter(row: Row) -> bool:
        in_m = is_in_mass(row['FullText'])
        return in_m if mass else not in_m
    return tuple(filter(loc_filter, rows))

def filter_by_type(rows: Tuple[Row,...], good = True) -> Tuple[Row,...]:
    def type_filter(row: Row) -> bool:
        is_good = is_good_type(row)
        return is_good if good else not is_good
    return tuple(filter(type_filter, rows))

def type_divide(rows: Tuple[Row,...]) -> Tuple[int, int]:
    good_count = len(filter_by_type(rows))
    return good_count, len(rows) - good_count

def classify(row: Row) -> str:
    if not is_good_type(row) == 0 or not filter_row(row):
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
    One of the types is 'Advertisement' AND the word 'ad' is not included in the title
    The column 'BinaryTypes' has all the types assigned to this article 
        binary encoded, so that each bit represents a possible type
    Advertisement is type 7, so we do a bitwise-and to 2**7 to determine whether article is
        categorized as advertisement
    """
    return row['GoodTypes'] > 0 or (row['BinaryTypes'] & 2**7 > 0 and 'ad' not in row['Title'].lower())

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

def confusion_matrix(rows: Tuple[Row,...]) -> Tuple[Tuple[Row,...],Tuple[Row,...],Tuple[Row,...],Tuple[Row,...]]:
    matches, no_matches = partition(rows, full_pred)
    TP, FP = partition(matches, manual_match)
    FN, TN = partition(no_matches, manual_match)
    return TP, TN, FP, FN

def statistic_summary(TP, TN, FP, FN) -> str:
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    msg = f"Matching statistics:\n" + \
        f"True Positives: {TP}\n" + \
        f"True Negatives: {TN}\n" + \
        f"False Positives: {FP}\n" + \
        f"False Negatives: {FN}\n" + \
        f"Sensitivity: {100*sens:.2f}%\n" + \
        f"Specificity: {100*spec:.2f}%"
    return msg
