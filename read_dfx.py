# Wat moet de functie doen?

# Inlezen DXF bestand
# Aangeven hoe veel elementen er zijn van alles
# Afbreken naar individuele lijnstukken
# Lijnstukken naar dict of .txt
# Coordinaten van lijnstuk of lijnstukken exporteren aan de hand van naam
# Lijnstukken binnen range exporteren

# %% Imports
import numpy as np
import ezdxf
from collections import Counter

# %% Input
filename = '../oostende_2d.dxf'

# %% Functions
# Read DXF file and get info
doc = ezdxf.readfile(filename)
msp = doc.modelspace()

# Get the amount of different entities in the file
entities = Counter()
for entity in msp:
    entities[entity.dxftype()] += 1
    print(type(entity))

print(f"Entities in {filename}:")
for entity_type, count in entities.items():
    print(f"{entity_type}: {count}")

# Get the no. of dimensions
is_3d = False
for entity in entities:
    for geometry in msp.query(entity):
        try:
            start = geometry.dxf.start
            if start[2] >= 1E-6:
                is_3d = True
                print(start)
                break  
        except ezdxf.DXFAttributeError:
            continue

if is_3d:
    print("3D DXF file detected.")
else:
    print("2D DXF file detected.")


# %% Split the entities

lines = []
for entity in msp:
    try:
        if entity.dxftype() == 'LINE':
            linepoints = [entity.dxf.start, entity.dxf.end]
            lines.append(linepoints)
        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points('xyz')
            linepoints = [point for point in points]
            lines.append(linepoints)
        elif entity.dxftype() == 'POLYLINE':
            linepoints = [[point[0], point[1], point[2]] for point in entity.points()]
            lines.append(linepoints)
    except Exception as e:
        print(f"Error processing entity {entity}: {e}")


# %%
