
filename = 'towns.txt'

with open(filename, 'r') as fp:
    lines = fp.readlines()

blanks = tuple(line for line in lines if len(line.strip())==0)

towns = tuple(line.strip().replace('*','') for line in lines if len(line.strip())>0 and line.strip() not in ('Florida', 'Lee', 'Washington', 'Berlin', 'Wales'))
print(f"Number of lines: {len(lines)}")
print(f"Number of blank lines: {len(blanks)}")
print(f"Number of towns: {len(towns)}")

filename = 'neighborhoods.txt'
with open(filename, 'r') as fp:
    lines = fp.readlines()

neighborhoods = tuple(line.strip() for line in lines)

towncode = "','".join(towns+neighborhoods+('Massachusetts',))

filename = 'mass_towns.py'

with open(filename, 'w') as fp:
    fp.write(f"townlist = ('{towncode}')")

from mass_towns import townlist

print(townlist)