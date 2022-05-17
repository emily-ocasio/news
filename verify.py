# pylint: skip-file
import sqlite3
import os
import textwrap


def main():

    cur = db.cursor()
    # infinite loop
    while True:
        pubdate = input(
            "Enter desired date in format YYYYMMDD, enter q to quit > ")
        if pubdate.lower()[0] == 'q':
            exit()
        variables = (pubdate,)
        fromclause = "FROM articles a"
        joinclause = "LEFT JOIN verifications v ON a.RecordId = v.RecordId"
        whereclause = "WHERE PubDate=? and v.Status IS NULL"
        sqlclause = " ".join((fromclause, joinclause, whereclause))
        count = cur.execute(f"SELECT COUNT(*) {sqlclause}", variables)
        total = count.fetchone()[0]
        articles = cur.execute(f"SELECT * {sqlclause}", variables)
        for i, article in enumerate(articles):
            print(f"Article {i+1} of {total}:")
            print(f"Record ID: {article['RecordId']} Title: {article['Title']}")
            print()
            print_types(article['RecordId'])
            linesleft = print_wrapped(article['FullText'])
            print()
            answer = ""
            while len(answer) == 0 or not answer in "MONP":
                extra = "E[X]tra lines, " if linesleft > 0 else ""
                answer = input(
                    f"Classify article as [M]assachusetts homicide, [O]ther place homicide, {extra}[N]ot homicide, [P]ass, [Q]uit for now > ")
                answer = answer.upper()
                if linesleft > 0 and answer == "X":
                    linesleft = print_wrapped(
                        article['FullText'],
                        start=35, total=200)
                if answer == "Q":
                    exit()
                if answer in "MNOP":
                    verify_record(article['RecordId'], answer)
            os.system('clear')


def print_wrapped(text, length=140, start=0, total=35):
    if not text:
        print("No text")
        return 0
    lines = textwrap.wrap(text, length)
    linecount = len(lines)
    for line in lines[start:start+total]:
        print(line)
    return max(linecount-start-total, 0)


def print_types(id):
    cur = db.cursor()
    types = cur.execute(
        "SELECT e.TypeDesc desc FROM articletypes t JOIN articleenum e ON e.TypeId = t.TypeId WHERE t.RecordId=?",
        (id,))
    for type in types:
        print(f"Article Type = {type['desc']}")


def verify_record(id, status):
    cur = db.cursor()
    cur.execute(
        "INSERT INTO verifications (Status, RecordId) VALUES (?, ?)",
        (status, id))
    db.commit()
    return


if __name__ == "__main__":
    db = sqlite3.connect('newarticles.db')
    db.row_factory = sqlite3.Row
    main()
