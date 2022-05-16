"""
import shr excel database into pandas
"""
import pandas as pd

SHR_FILE = "~/Downloads/SHR76_20.xlsx"

shr = pd.read_excel(SHR_FILE, 'SHR76_20')

print(shr)
