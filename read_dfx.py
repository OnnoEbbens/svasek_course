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
            if start[2] >= 1e-6:
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
    line = LineString(linepoints)
    return line


def get_lwpolylines(entity):
    points = entity.get_points("xy")
    elevation = entity.dxf.elevation
    linepoints = [list(point) + [elevation] for point in points]
    if linepoints[-1] == linepoints[0]:
        line = Polygon(linepoints)
    else:
        line = LineString(linepoints)
    return line


def get_polylines(entity):
    linepoints = [[point[0], point[1], point[2]] for point in entity.points()]
    if linepoints[-1] == linepoints[0]:
        line = Polygon(linepoints)
    else:
        line = LineString(linepoints)
    return line


def get_insert(entity):
    linepoints = []
    sub_entities = entity.virtual_entities()
    for sub_entity in sub_entities:
        if sub_entity.dxftype() == "LINE":
            linepoints.append(get_lines(sub_entity))
        elif sub_entity.dxftype() == "LWPOLYLINE":
            linepoints.append(get_lwpolylines(sub_entity))
        elif sub_entity.dxftype() == "POLYLINE":
            linepoints.append(get_polylines(sub_entity))
    return linepoints

lines = {"LINES": [], "LWPOLYLINES": [], "POLYLINES": [], "BLOCKS": []}
for entity in msp:
    # try:
    if entity.dxftype() == "LINE":
        line = get_lines(entity)
        lines["LINES"].append(line)

    elif entity.dxftype() == "LWPOLYLINE":
        line = get_lwpolylines(entity)
        lines["LWPOLYLINES"].append(line)

    elif entity.dxftype() == "POLYLINE":
        line = get_polylines(entity)
        lines["POLYLINES"].append(line)

    elif entity.dxftype() == "INSERT":
        line = get_insert(entity)
        [lines["BLOCKS"].append(iline) for iline in line]

    # except Exception as e:
        # print(f"Error processing entity {entity}: {e}")

# %% Plot the lines in 2D

def plot_lines(lines, color, x_index=0, y_index=1, **kwargs):
    for line in lines:
        if len(line) == 2:
            x = [line[0][x_index], line[1][x_index]]
            y = [line[0][y_index], line[1][y_index]]
            plt.plot(x, y, color=color, **kwargs)
        elif len(line) > 2:
            x = [point[x_index] for point in line]
            y = [point[y_index] for point in line]
            plt.plot(x, y, color=color, **kwargs)


plt.figure()
plt.subplot(2,1,1, aspect='equal')
for i, shape in enumerate(lines.keys()):
    print(f"Plotting {shape}: {len(lines[shape])}")
    if shape == 'BLOCKS':
        for block in lines[shape]:
            plot_lines(block, 'r', alpha=0.5)
    elif shape == 'LINES':
        plot_lines(lines[shape], 'b', alpha=0.5)
    elif shape == 'LWPOLYLINES':
        plot_lines(lines[shape], 'g', alpha=0.5)
    elif shape == 'POLYLINES':
        plot_lines(lines[shape], 'c', alpha=0.5)

plt.title("DXF Lines")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')
plt.grid()

plt.subplot(2,1,2, aspect='equal')
for i, shape in enumerate(lines.keys()):
    print(f"Plotting {shape}: {len(lines[shape])}")
    if shape == 'BLOCKS':
        for block in lines[shape]:
            plot_lines(block, 'r', 1, 2)
    elif shape == 'LINES':
        plot_lines(lines[shape], 'b', 1, 2)
    elif shape == 'LWPOLYLINES':
        plot_lines(lines[shape], 'g', 1, 2)
    elif shape == 'POLYLINES':
        plot_lines(lines[shape], 'c', 1, 2)

plt.title("DXF Lines")
def change_xy_to_yz(geom):
    if type(geom) == shapely.geometry.linestring.LineString:
        points = [arr for arr in geom.coords]
        coords = [(y, z) for x, y, z in points]
        return LineString(coords)
    elif type(geom) == shapely.geometry.polygon.Polygon:
        pass


gpd_lines = gpd.GeoDataFrame({"geometry": lines["LINES"]})
gpd_lwpolylines = gpd.GeoDataFrame({"geometry": lines["LWPOLYLINES"]})
gpd_polylines = gpd.GeoDataFrame({"geometry": lines["POLYLINES"]})
gpd_blocks = gpd.GeoDataFrame({"geometry": lines["BLOCKS"]})

gpd_data = pd.concat([gpd_lines, gpd_lwpolylines, gpd_polylines, gpd_blocks], ignore_index=True)
gpd_data["geometry"].force_3d()
gpd_data["geometry_yz"] = gpd_data["geometry"].apply(change_xy_to_yz)

# gpd_blocks = gpd.GeoDataFrame({'geometry': lines['BLOCKS']})

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')
plt.grid()
plt.show()

# %%
