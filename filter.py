import sqlite3
import textwrap
import os
import re

filters = [
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
    while True:
        choice = get_user_choice(initial_prompts)
        if choice == "R":
            review_labels()
        elif choice == "N":
            new_labels()
        elif choice == "F":
            fix_record()

def fix_record():
    recordid = input("Enter Record Id to fix, <Return> to go back, [Q] to quit > ")
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
    label_articles((result,), 1, allow_return = True, filter = False)

def new_labels():
    print("\n\nSelect date to label articles")
    pubdate = get_date()
    countsql, sql = unverified_articles_sql()
    cur = db.cursor()
    total = cur.execute(countsql, (pubdate,)).fetchone()[0]
    rows = cur.execute(sql, (pubdate,))
    label_articles(rows, total, filter = False)

def get_date():
    answer = input("Enter date in format YYYYMMDD, [Q] to quit > ")
    answer = answer.upper()
    if answer == "Q":
        exit()
    return answer

def review_labels():
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
    review_articles(status, total, show_match = show_match)

def review_articles(status, total, show_match = False):
    sql = verified_articles_sql()
    cur = db.cursor()
    rows = cur.execute(sql, (status,))
    label_articles(rows, total, allow_return = True, show_match = show_match)

def label_articles(rows, total, allow_return = False, show_match = True, filter = True):
    current = 0
    for row in rows:
        match = (not filter) or filter_document(row['Title']) or filter_document(row['FullText'])
        show = (not filter) or (match if show_match else not match)
        if show:
            current += 1
            os.system('clear')
            print(f"Article #{current} of {total}")
            linesleft = show_single(row)
            prompts = label_prompts.copy()
            if linesleft > 0:
                prompts.append("show e[X]tra lines")
            choice = get_user_choice(prompts, allow_return = allow_return)
            if choice == "X":
                show_single(row, start = 35)
                choice = get_user_choice(label_prompts, allow_return = allow_return)
            if choice == "":
                continue
            save_verification(row, choice)

def save_verification(row, choice):
    if row['status'] is None:
        sql = "INSERT INTO verifications (Status, RecordId) VALUES (?, ?)"
    else:
        sql = "UPDATE verifications SET Status = ? WHERE RecordId = ?"
    cur = db.cursor()
    cur.execute(sql, (choice, row['RecordId'],))
    db.commit()

def show_single(row, start = 0):
    if start == 0:
        print(f"Title: {row['Title']}    Date: {row['PubDate'][4:6]}/{row['PubDate'][6:8]}/{row['PubDate'][0:4]}\n")
        print(f"Record ID = {row['RecordId']}")
        print(f"Verified status = <{row['status']}>\n")
        print_types(row['RecordId'])
        total = 35
    else:
        total = 1000
    return print_wrapped(row['FullText'], start = start, total = total)

def print_types(id):
    cur = db.cursor()
    types = cur.execute("SELECT e.TypeDesc desc FROM articletypes t JOIN articleenum e ON e.TypeId = t.TypeId WHERE t.RecordId=?", (id,))
    for type in types:
        print(f"Article Type = {type['desc']}")

def count_matches(status = "N"):
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
    print(f"# of articles matching filters: {match_count}, Not matching filters: {nonmatch_count}, Total: {match_count+nonmatch_count}")
    return match_count, nonmatch_count

def verified_articles_sql():
    return """
        SELECT *
        FROM verified_articles
        WHERE status = ?
        AND dataset="TRAIN";
    """

def unverified_articles_sql():
    sql = """
        FROM articles a
        LEFT JOIN verifications v ON a.RecordId = v.RecordId
        WHERE PubDate = ? AND v.Status IS NULL
    """
    return "SELECT COUNT(*) " + sql, "SELECT * " + sql

def filter_document(document, filters = filters):
    if document is None:
        return False
    for filter in filters:
        if all(word in document for word in filter):
            return True
    return False

def print_wrapped(text, length = 140, start = 0, total = 35):
    if not text:
        print("No text")
        return 0
    lines = textwrap.wrap(text, length)
    linecount = len(lines)
    for line in lines[start:start+total]:
        print(line)
    return max(linecount-start-total,0)

def get_user_choice(prompts, add_quit = True, allow_return = False):
    full_prompt, choices = unified_prompt(prompts, add_quit = add_quit, allow_return = allow_return)
    answer = ""
    while not answer in choices:
        answer = input(full_prompt).upper()
        if allow_return and answer == "":
            return ""
    if answer == "Q":
        exit()
    return answer

def unified_prompt(prompts, add_quit = True, allow_return = False):
    """
    Creates a single prompt including individual choices,
    also adds an option to quit.
    Also returns list of menu letters based on letters inside brackets in prompts.
    """
    regex = '\[(\w)\]'
    full_prompt = "Select option: " + ", ".join(prompts) + (", [Q]uit" if add_quit else "") + (", <Return> to continue" if allow_return else "") + " > "
    return full_prompt, re.findall(regex, full_prompt)

if __name__ == "__main__":
    db = sqlite3.connect('newarticles.db')
    db.row_factory = sqlite3.Row
    main();