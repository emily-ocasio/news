"""
This module provides functionality for filtering and labeling homicide-related
news articles.

It includes functions for:
- Reviewing and labeling articles
- Filtering articles based on specific keywords
- Interacting with a SQLite database for article information
- Displaying article content and prompting user input for labeling

The module uses predefined filters to identify potential homicide-related
articles and allows users to manually review and categorize them.
"""
import sqlite3
import textwrap
import os
import re

db: sqlite3.Connection

default_filters = [
    ('slain',),
    ('slaying',),
    ('murder',),
    ('homicide',),
    ('shot to death',),
    ('shot dead',),
    ('stabbed', 'die'),
    ('wounded', 'die'),
    ('strangl', 'die'),
    ('shot', 'die'),
    ('shot', 'kill'),
    ('shooting', 'death'),
    ('manslaughter',),
    ('fatally wounded',),
    ('fatally stabbed',),
    ('fatal stabbing',),
    ('kidnap', 'kill')
]
# Unused filters:
# ('hit by car', 'die'),
# ('struck by car', 'die'),
# ('hit and run', 'die'),

initial_prompts = [
    "[R]eview matches from previous labels",
    "[F]ix errors by Record Id",
    "Enter [N]ew labels"
]

label_prompts = [
    "[M]assachussetts homicides",
    "[O]ther location homicides",
    "[N]ot homicides",
    "[P]ass and label later"
]

review_prompts = [
    "Review articles that [M]atch homicide keywords",
    "Review articles that do [N]ot match keywords",
    "[C]ontinue without reviewing"
]


def main():
    """" """
    while True:
        choice = get_user_choice(initial_prompts)
        if choice == "R":
            review_labels()
        elif choice == "N":
            new_labels()
        elif choice == "F":
            fix_record()


def fix_record():
    """
    This function prompts the user to enter a Record Id to fix.
    It then retrieves the corresponding article from the database 
        and calls the
    `label_articles` function to allow the user to manually review 
        and label the article.

    If the user enters 'Q' to quit, the function will exit.
    If the user enters an empty string, the function will return 
        without performing any action.
    """
    recordid = input(
        "Enter Record Id to fix, <Return> to go back, [Q] to quit > ")
    if recordid.upper() == "Q":
        exit()
    if recordid == "":
        return
    sql = "SELECT * FROM verified_articles WHERE RecordId = ?"
    cur = db.cursor()
    result = cur.execute(sql, (recordid,)).fetchone()
    if result is None:
        print("No record found.")
        return
    label_articles((result,), 1, allow_return=True, use_filter=False)


def new_labels():
    """
    This function is responsible for initiating the labeling process 
    for new articles.
    It prompts the user to select a date for labeling, retrieves 
    the unverified articles for that date,
    and then calls the `label_articles` function to start labeling.

    Parameters:
    None

    Returns:
    None
    """
    print("\n\nSelect date to label articles")
    pubdate = get_date()
    countsql, sql = unverified_articles_sql()
    cur = db.cursor()
    total = cur.execute(countsql, (pubdate,)).fetchone()[0]
    rows = cur.execute(sql, (pubdate,))
    label_articles(rows, total, use_filter=False)


def get_date():
    """
    Prompts the user to enter a date in the format YYYYMMDD or to quit 
    the program.

    Parameters:
    None

    Returns:
    str: The entered date as a string in the format YYYYMMDD. If the user 
    chooses to quit, the program exits.
    """
    answer = input("Enter date in format YYYYMMDD, [Q] to quit > ")
    answer = answer.upper()
    if answer == "Q":
        exit()
    return answer


def review_labels():
    """
    This function initiates the review process for articles 
        based on their labels.
    It prompts the user to select a label and a group of articles to review.
    The function then calls the `review_articles` function to start the review.

    Parameters:
    None

    Returns:
    None
    """
    print("\nWhich label would you like to review?")
    status = get_user_choice(label_prompts)
    match_count, nonmatch_count = count_matches(status)

    print("\nWhich group of articles would you like to review?")
    choice = get_user_choice(review_prompts)
    if choice == "C":
        return
    if choice == "M":
        total = match_count
        show_match = True
    else:
        total = nonmatch_count
        show_match = False
    review_articles(status, total, show_match=show_match)


def review_articles(status, total, show_match=False):
    """
    This function initiates the review process for articles 
        based on their labels.
    It retrieves the articles from the database that match the given status 
        and calls the
    `label_articles` function to start the review.

    Parameters:
    status (str): The label status to review. It can be one of the following:
        - 'M': Massachusetts homicides
        - 'O': Other location homicides
        - 'N': Not homicides
        - 'P': Pass and label later
    total (int): The total number of articles to review.
    show_match (bool, optional): If True, only articles that match 
        the filters will be shown.
        If False, only articles that do not match the filters will be shown.
        Defaults to False.

    Returns:
    None
    """
    sql = verified_articles_sql()
    cur = db.cursor()
    rows = cur.execute(sql, (status,))
    label_articles(rows, total, allow_return=True, show_match=show_match)


def label_articles(rows, total, allow_return=False,
                   show_match=True, use_filter=True):
    """
    This function iterates through a set of articles 
        and allows the user to label them.

    Parameters:
    rows (sqlite3.Cursor): A cursor object containing 
        the articles to be labeled.
    total (int): The total number of articles to be labeled.
    allow_return (bool, optional): If True, allows the user 
        to continue labeling articles by pressing Enter. Defaults to False.
    show_match (bool, optional): If True, only articles 
        that match the filters will be shown. If False, only articles 
        that do not match the filters will be shown. Defaults to True.
    use_filter (bool, optional): If True, uses the filters to determine 
        whether an article should be shown. If False, shows all articles. 
        Defaults to True.

    Returns:
    None
    """
    current = 0
    for row in rows:
        match = (not use_filter) or filter_document(
            row['Title']) or filter_document(row['FullText'])
        show = (not use_filter) or (match if show_match else not match)
        if show:
            current += 1
            os.system('clear')
            print(f"Article #{current} of {total}")
            linesleft = show_single(row)
            prompts = label_prompts.copy()
            if linesleft > 0:
                prompts.append("show e[X]tra lines")
            choice = get_user_choice(prompts, allow_return=allow_return)
            if choice == "X":
                show_single(row, start=35)
                choice = get_user_choice(
                    label_prompts, allow_return=allow_return)
            if choice == "":
                continue
            save_verification(row, choice)


def save_verification(row, choice):
    """
    This function saves the verification status of an article in the database.

    Parameters:
    row (sqlite3.Row): A row object representing an article from the database.
        The row should contain the following columns: 'status' and 'RecordId'.
    choice (str): The verification status to be saved. 
        It should be one of the following:
        - 'M': Massachusetts homicides
        - 'O': Other location homicides
        - 'N': Not homicides
        - 'P': Pass and label later

    Returns:
    None

    The function updates the 'Status' column 
        in the 'verifications' table of the database.
    If the 'status' column in the input row is None, a new row 
        is inserted into the 'verifications' table.
    If the 'status' column is not None, the existing row 
        in the 'verifications' table is updated.
    """
    if row['status'] is None:
        sql = "INSERT INTO verifications (Status, RecordId) VALUES (?, ?)"
    else:
        sql = "UPDATE verifications SET Status = ? WHERE RecordId = ?"
    cur = db.cursor()
    cur.execute(sql, (choice, row['RecordId'],))
    db.commit()


def show_single(row, start=0):
    """
    Displays information from a single row of data and prints wrapped text.

    Args:
        row (dict): A dictionary containing the data for a single row. 
            Expected keys are 
            'Title', 'PubDate', 'RecordId', 'status', and 'FullText'.
        start (int, optional): The starting position for wrapping the text. i
            Defaults to 0.

    Returns:
        str: The wrapped text from the 'FullText' field of the row.
    """
    if start == 0:
        print(
            f"Title: {row['Title']}    "
            + f"Date: {row['PubDate'][4:6]}"
            + f"/{row['PubDate'][6:8]}/{row['PubDate'][0:4]}\n")
        print(f"Record ID = {row['RecordId']}")
        print(f"Verified status = <{row['status']}>\n")
        print_types(row['RecordId'])
        total = 35
    else:
        total = 1000
    return print_wrapped(row['FullText'], start=start, total=total)


def print_types(record_id):
    """
    Prints the types of articles associated with a given record ID.

    Args:
        record_id (int): The ID of the record for which to 
        retrieve and print article types.

    Returns:
        None

    Side Effects:
        Prints the article types to the console.
    """
    cur = db.cursor()
    types = cur.execute(
        "SELECT e.TypeDesc FROM articletypes t JOIN articleenum e "
        +"ON e.TypeId = t.TypeId WHERE t.RecordId=?", (record_id,))
    for row in types:
        print(f"Article Type = {row['TypeDesc']}")


def count_matches(status="N"):
    """
    Count the number of articles that match or do not match certain filters.

    Args:
        status (str): The status of the articles to be examined. Default is "N".

    Returns:
        tuple: A tuple containing the count of matching articles 
            and non-matching articles.

    Prints:
        The number of articles matching the filters, not matching the filters, 
            and the total number of articles examined.
    """
    print("Examining documents...")
    sql = verified_articles_sql()
    cur = db.cursor()
    rows = cur.execute(sql, (status,))
    match_count = 0
    nonmatch_count = 0
    for row in rows:
        if filter_document(row['Title']) or filter_document(row['FullText']):
            match_count += 1
        else:
            nonmatch_count += 1
    print(
        f"# of articles matching filters: {match_count}, "
        +f"Not matching filters: {nonmatch_count}, "
        +f"Total: {match_count+nonmatch_count}")
    return match_count, nonmatch_count


def verified_articles_sql():
    """
    Generates an SQL query to select all columns 
        from the 'verified_articles' table
    where the status matches a specified value and the dataset is 'TRAIN'.

    Returns:
        str: The SQL query string.
    """
    return """
        SELECT *
        FROM verified_articles
        WHERE status = ?
        AND dataset="TRAIN";
    """


def unverified_articles_sql():
    """
    Generates SQL queries to count and retrieve unverified articles.

    The function constructs two SQL queries:
    1. A query to count the number of unverified articles.
    2. A query to retrieve all unverified articles.

    The queries join the 'articles' table with the 
    'verifications' table on the 'RecordId' field
    and filter the results to include only those articles 
    where the publication date matches the
    provided date and the verification status is NULL.

    Returns:
        tuple: A tuple containing two SQL query strings.
            - The first string is the SQL query to count unverified articles.
            - The second string is the SQL query to 
            retrieve unverified articles.
    """
    sql = """
        FROM articles a
        LEFT JOIN verifications v ON a.RecordId = v.RecordId
        WHERE PubDate = ? AND v.Status IS NULL
    """
    return "SELECT COUNT(*) " + sql, "SELECT * " + sql


def filter_document(document, filters=None):
    """
    Filters a document based on the provided filters.

    Args:
        document (str): The document to be filtered.
        filters (list of list of str, optional): 
            A list of lists, where each inner list contains words 
            that must all be present in the document for it to pass the filter. 
            If None, default_filters will be used.

    Returns:
        bool: True if the document passes any of the filters, False otherwise.
    """
    if document is None:
        return False
    if filters is None:
        filters = default_filters
    for filter_word in filters:
        if all(word in document for word in filter_word):
            return True
    return False


def print_wrapped(text, length=140, start=0, total=35):
    """
    Prints the given text wrapped to a specified length 
    and returns the number of remaining lines.

    Args:
        text (str): The text to be wrapped and printed.
        length (int, optional): The maximum length of each line. 
        Defaults to 140.
        start (int, optional): The starting line index to print from. 
        Defaults to 0.
        total (int, optional): The total number of lines to print. 
        Defaults to 35.

    Returns:
        int: The number of remaining lines after the printed lines.
    """
    if not text:
        print("No text")
        return 0
    lines = textwrap.wrap(text, length)
    linecount = len(lines)
    for line in lines[start:start+total]:
        print(line)
    return max(linecount-start-total, 0)


def get_user_choice(prompts, add_quit=True, allow_return=False):
    """
    Prompts the user to make a choice from a list of options.

    Args:
        prompts (list or str): 
            A list of prompt strings or a single prompt string.
        add_quit (bool, optional): If True, adds a 'Q' option to quit. 
            Defaults to True.
        allow_return (bool, optional): If True, allows the user 
        to return by pressing Enter. Defaults to False.

    Returns:
        str: The user's choice from the list of options. 
        Returns an empty string if allow_return is True 
            and the user presses Enter.
    """
    full_prompt, choices = unified_prompt(
        prompts, add_quit=add_quit, allow_return=allow_return)
    answer = ""
    while not answer in choices:
        answer = input(full_prompt).upper()
        if allow_return and answer == "":
            return ""
    if answer == "Q":
        exit()
    return answer


def unified_prompt(prompts, add_quit=True, allow_return=False):
    """
    Creates a single prompt including individual choices,
    also adds an option to quit.
    Also returns list of menu letters based on 
    letters inside brackets in prompts.
    """
    regex = r'\[(\w)\]'
    full_prompt = "Select option: " + ", ".join(prompts) + (
        ", [Q]uit" if add_quit else "") + (
        ", <Return> to continue" if allow_return else "") + " > "
    return full_prompt, re.findall(regex, full_prompt)


if __name__ == "__main__":
    db = sqlite3.connect('newarticles.db')
    db.row_factory = sqlite3.Row
    main()
