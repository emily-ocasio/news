import numpy as np
import pandas as pd

shr_file = "~/Downloads/SHR76_20.xlsx"

shr = pd.read_excel(shr_file, 'SHR76_20')

print(shr)