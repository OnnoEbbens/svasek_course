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
filename = 'oostende_2d.dxf'

# %% Functions
# Read DXF file and get info
doc = ezdxf.readfile(filename)
msp = doc.modelspace()

# Get the amount of different entities in the file
# entities = Counter()
# for entity in msp:
#     entities[entity.dxftype()] += 1
#     print(type(entity))

# print(f"Entities in {filename}:")
# for entity_type, count in entities.items():
#     print(f"{entity_type}: {count}")

# # Get the no. of dimensions
# is_3d = False
# for entity in entities:
#     for geometry in msp.query(entity):
#         try:
#             start = geometry.dxf.start
#             if start[2] >= 1E-6:
#                 is_3d = True
#                 print(start)
#                 break  
#         except ezdxf.DXFAttributeError:
#             continue

# if is_3d:
#     print("3D DXF file detected.")
# else:
#     print("2D DXF file detected.")



# def split_dxf_into_lines(msp: ezdxf.layouts.layout.Modelspace):
polylines = []

for polyline in msp.query('POLYLINE'):
    i_polyline = []
    # Get the vertices of the polyline
    vertices = polyline.vertices
    # Split the polyline into individual line segments
    for i in range(len(vertices) - 1):
        start_point = vertices[i].format()
        end_point = vertices[i + 1].format()
        i_polyline.append([start_point, end_point])
    polylines.append(i_polyline)

lw_polylines = []
for lwpolyline in msp.query('LWPOLYLINE'):
    i_lwpolyline = [i+345]
    # Get the vertices of the polyline
    for vertex in lwpolyline.vertices_in_wcs():
        i_lwpolyline.append(vertex.format())
    lw_polylines.append(i_lwpolyline)


# %%
