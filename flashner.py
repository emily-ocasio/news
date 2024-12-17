#pylint: skip-file
import sqlite3
from flashtext import KeywordProcessor #type: ignore
from mass_towns import townlist
from colorama import Fore

def colored_word(word: str, color = Fore.GREEN) -> str:
    return color + word + Fore.RESET

kp = KeywordProcessor()
for keyword in townlist:
    kp.add_keyword(keyword, colored_word(keyword))
# kp.add_keywords_from_list(list(townlist))

def in_mass(gpe):
    return kp.extract_keywords(gpe)
    
id = 1637276385

sql = """
    SELECT *
    FROM verified_articles
    WHERE status = ?
    AND dataset = "TRAIN"
"""
status = "M"
db = sqlite3.connect("newarticles.db")
db.row_factory = sqlite3.Row

rows = db.execute(sql, (status,))

count_mass = 0
count_other = 0
for article in rows:

    text = article['FullText']
    # print(text)
    #text = "Murder was in San Francisco, also in South Boston, or in Cambridge and Suffolk County"

    places = in_mass(text)
    if len(places) == 0:
        print(text)
        print("Not in Massachusetts")
        count_other += 1
    
    else:
        #print(kp.replace_keywords(text))
        for place in set(places):
            print(place)
        count_mass += 1

    print()
    print()
    print()

print(f"In Mass: {count_mass}")
print(f"Not in Mass: {count_other}")


