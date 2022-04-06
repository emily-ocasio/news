import spacy #type: ignore
import sqlite3
from geopy.geocoders import Nominatim, options #type: ignore
from geopy.adapters import URLLibAdapter, RequestsAdapter #type: ignore

#options.default_adapter_factory = URLLibAdapter
geolocator = Nominatim(user_agent = 'ocasio-app')

def in_mass(gpe):
    box = ((42.7, -73.43),(42.0,-70.6))
    locations = geolocator.geocode(gpe, exactly_one = False, viewbox=box)
    if locations is None:
        return False
    return 'Massachusetts' in locations[0].address

nlp = spacy.load('en_core_web_lg')

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



for article in rows:

    text = article['FullText']
    print(text)
    #text = "Murder was in San Francisco, also in South Boston, or in Cambridge and Suffolk County"

    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "GPE":
            print(ent.text)
            #print(ent.text, ent.label_, "Mass." if in_mass(ent.text) else "Not Mass.")

    print()
    print()
    print()


