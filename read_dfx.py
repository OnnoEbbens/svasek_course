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
import matplotlib.pyplot as plt

# %% Input
filename = '../oostende.dxf'
# filename = '../bridge.dxf'

# %% Functions
# Read DXF file and get info
doc = ezdxf.readfile(filename)
msp = doc.modelspace()

# Get the amount of different entities in the file
entities = Counter()
for entity in msp:
    entities[entity.dxftype()] += 1

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

def get_lines(entity):
    linepoints = [entity.dxf.start, entity.dxf.end]
    return linepoints

def get_lwpolylines(entity):
    points = entity.get_points('xy')
    elevation = entity.dxf.elevation
    linepoints = [list(point) + [elevation] for point in points]
    return linepoints

def get_polylines(entity):
    linepoints = [[point[0], point[1], point[2]] for point in entity.points()]
    return linepoints

def get_insert(entity):
    linepoints = []
    sub_entities = entity.virtual_entities()
    for sub_entity in sub_entities:
        if sub_entity.dxftype() == 'LINE':
            linepoints.append(get_lines(sub_entity))
        elif sub_entity.dxftype() == 'LWPOLYLINE':
            linepoints.append(get_lwpolylines(sub_entity))
        elif sub_entity.dxftype() == 'POLYLINE':
            linepoints.append(get_polylines(sub_entity))
    return linepoints

lines = {
    'LINES': [],
    'LWPOLYLINES': [],
    'POLYLINES': [],
    'BLOCKS': []
}
for entity in msp:
    try:
        if entity.dxftype() == 'LINE':
            linepoints = get_lines(entity)
            lines['LINES'].append(linepoints)

        elif entity.dxftype() == 'LWPOLYLINE':
            linepoints = get_lwpolylines(entity)
            lines['LWPOLYLINES'].append(linepoints)

        elif entity.dxftype() == 'POLYLINE':
            linepoints = get_polylines(entity)
            lines['POLYLINES'].append(linepoints)

        elif entity.dxftype() == 'INSERT':
            linepoints = get_insert(entity)
            lines['BLOCKS'].append(linepoints)

    except Exception as e:
        print(f"Error processing entity {entity}: {e}")


# %%
