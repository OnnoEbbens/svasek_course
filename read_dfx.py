# Wat moet de functie doen?

# Inlezen DXF bestand
# Aangeven hoe veel elementen er zijn van alles
# Afbreken naar individuele lijnstukken
# Lijnstukken naar dict of .txt
# Coordinaten van lijnstuk of lijnstukken exporteren aan de hand van naam
# Lijnstukken binnen range exporteren

# TODO: z-dimensie toevoegen in slices
# TODO: Plotjes netter maken

# %% Imports
import ezdxf
import simplekml
import pandas as pd
import geopandas as gpd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import shapely
from shapely.geometry import Polygon, LineString


# %% Class definitions
class DXFReader:
    def __init__(self, filename):
        print(f"Reading DXF file: {filename}")
        self.filename = filename
        self.doc = ezdxf.readfile(filename)
        self.msp = self.doc.modelspace()
        self.entities = Counter()
        self.lines = {"LINES": [], "LWPOLYLINES": [], "POLYLINES": [], "BLOCKS": []}
        self.is_3d = False
        self.gpd_data = None
        self.get_entities()
        self.detect_3d()

    def get_entities(self):
        """Get a summary of the entities in the DXF file.
        """
        for entity in self.msp:
            self.entities[entity.dxftype()] += 1

        print("")
        print(f"Entities in {self.filename}:")
        for entity_type, count in self.entities.items():
            print(f"{entity_type}: {count}")
        print("")

    def detect_3d(self):
        """Detect if the DXF file is 3D or 2D.
        """
        for entity in self.entities:
            for geometry in self.msp.query(entity):
                try:
                    start = geometry.dxf.start
                    if start[2] >= 1e-6:
                        self.is_3d = True
                        break
                except ezdxf.DXFAttributeError:
                    continue

        if self.is_3d:
            print("3D DXF file detected.")
        else:
            print("2D DXF file detected.")
        print("")

    def entities_to_gpd(self):
        """Transform the DXF entities to a GeoDataFrame.
        Sets self.gpd_data to a GeoDataFrame containing the lines.
        """
        self._entities_to_shapely()
        gpd_lines = gpd.GeoDataFrame({"geometry": self.lines["LINES"]})
        gpd_lwpolylines = gpd.GeoDataFrame({"geometry": self.lines["LWPOLYLINES"]})
        gpd_polylines = gpd.GeoDataFrame({"geometry": self.lines["POLYLINES"]})
        gpd_blocks = gpd.GeoDataFrame({"geometry": self.lines["BLOCKS"]})

        self.gpd_data = pd.concat(
            [gpd_lines, gpd_lwpolylines, gpd_polylines, gpd_blocks], ignore_index=True
        )
        if self.is_3d:
            self.gpd_data["geometry"].force_3d()
            self.get_yz()

    def plot_dxf(self):
        """Plot the DXF file
        """
        fig1, ax1 = self._set_fig()
        plt.title("Top down view")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        self.gpd_data["geometry"].plot(ax=ax1, color="k", alpha=0.5, edgecolor="black")

        if self.is_3d:
            fig2, ax2 = self._set_fig()
            plt.title("Side view")
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
            self.gpd_data["geometry_yz"].plot(
                ax=ax2, color="k", alpha=0.5, edgecolor="black"
            )

        plt.show()

    def get_yz(self):
        """Ge the yz coordinates of the geometry
        """
        self.gpd_data["geometry_yz"] = self.gpd_data["geometry"].apply(
            self._change_xy_to_yz
        )

    def set_crs(self, crs):
        """Set the coordinate reference system of the GeoDataFrame.

        Args:
            crs (string): crs in EPSG format (e.g. "EPSG:31370").
        """
        self.gpd_data.set_crs(crs, inplace=True)

    def transform_crs(self, crs):
        """Transform the coordinate reference system of the GeoDataFrame.

        Args:
            crs (string): crs in EPSG format (e.g. "EPSG:31370").

        Raises:
            ValueError: If the CRS is not set in geodataframe
        """
        if self.gpd_data.crs is None:
            raise ValueError("CRS not set. Please set the CRS before transforming.")
        self.gpd_data.to_crs(crs, inplace=True)
        
    def get_lines_from_fig(self):
        """Click in figure to select lines and save to csv files

        Returns:
            List, List: x and y coordinates of the selected lines
        """
        fig, ax = self._set_fig()
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Select one or multiple lines from the figure")

        # Plot each geometry separately so we can attack a picker
        lines = []
        for geom in self.gpd_data.geometry:
            if geom.geom_type == "LineString":
                x, y = geom.xy
                (line,) = ax.plot(x, y, color="k", alpha=0.5, picker=2)
                lines.append(line)

        # Pick event callback
        x_data = []
        y_data = []

        def on_pick(event):
            line = event.artist
            xdata = line.get_xdata()
            ydata = line.get_ydata()

            print("Selected line coordinates:")
            print(list(zip(xdata, ydata)))

            x_data.append(xdata)
            y_data.append(ydata)

            line.set_color("red")
            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)
        plt.show()

        if not x_data:
            print("No valid lines selected.")
            return

        self.lines_to_csv(x_data, y_data)
        return x_data, y_data

    def lines_to_csv(self, x_lines, y_lines):
        """Save selected lines to csv files.

        Args:
            x_lines (list): x coordinates of the selected lines
            y_lines (list): y coordinates of the selected lines
        """
        # Save x and y to csv file
        for i in range(len(x_lines)):
            ix = x_lines[i]
            iy = y_lines[i]
            df = pd.DataFrame({"x": ix, "y": iy})
            df.to_csv(f"line_{i}.csv", index=False)

    def extract_from_polygon(self, x_p, y_p, z=None):
        """Extract lines from a polygon defined by x_p and y_p.

        Args:
            x_p (list): x coordinates of polygon
            y_p (list): y coordinates of polygon
            z (list, optional): NOT YET ADDED. Defaults to None.

        Returns:
            geopandas.geodataframe: subset of original dataframe that lies inside the polygon.
        """
        poly = Polygon([(x_p, y_p) for x_p, y_p in zip(x_p, y_p)])
        mask = self.gpd_data.geometry.within(poly)
        ser = self.gpd_data[mask]

        # Check for crs in ser
        if self.gpd_data.crs is not None:
            ser.set_crs(self.gpd_data.crs, inplace=True)
        return ser

    def get_lines_from_rectangle(self):
        """Select lines from a rectangle in the figure and save to csv/geojson files.
        """
        fig, ax = self._set_fig()
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # Plot DFX
        self.gpd_data["geometry"].plot(ax=ax, color="k", alpha=0.5, edgecolor="black")
        plt.title("Rectangle selector")

        # Store selected rectangles
        selected_rectangles = []

        def on_select(eclick, erelease):
            # Extract corner coordinates
            x1, y1 = eclick.xdata, eclick.ydata  # First click
            x2, y2 = erelease.xdata, erelease.ydata  # Release point

            # Calculate corners
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            corners = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]

            selected_rectangles.append(corners)

            print("Rectangle drawn with corners:")
            for i, (x, y) in enumerate(corners, 1):
                print(f"  Corner {i}: ({x:.2f}, {y:.2f})")

            # Optional: draw rectangle for visual feedback
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            fig.canvas.draw()

        # Create RectangleSelector
        toggle_selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=0,
            minspany=0,
            spancoords="data",
            interactive=True,
        )

        plt.show()

        # Return if we have no lines
        if not selected_rectangles:
            print("No rectangles selected or no lines in rectangle.")
            return

        # After window closes, print all stored rectangles
        print("\nAll saved rectangles:")
        for i, rect in enumerate(selected_rectangles):
            print(f"Rectangle {i + 1} corners: {rect}")

        x_rect = [rect[0] for rect in selected_rectangles[0]]
        y_rect = [rect[1] for rect in selected_rectangles[0]]

        selected_lines = self.extract_from_polygon(x_rect, y_rect)
        selected_lines["geometry"].to_csv(
            "lines_snippet_from_rectangle.csv", index=False
        )
        selected_lines["geometry"].to_file(
            "lines_snippet_from_rectangle.geojson", driver="GeoJSON"
        )

        fig2, ax2 = self._set_fig()
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        selected_lines["geometry"].plot(ax=ax2, color="k", alpha=0.5, edgecolor="black")
        plt.title("Selected lines")
        plt.show()

    def _entities_to_shapely(self):
        """Transform the DXF entities to shapely geometries.
        """
        for entity in self.msp:
            try:
                if entity.dxftype() == "LINE":
                    line = self._get_lines(entity)
                    self.lines["LINES"].append(line)

                elif entity.dxftype() == "LWPOLYLINE":
                    line = self._get_lwpolylines(entity)
                    self.lines["LWPOLYLINES"].append(line)

                elif entity.dxftype() == "POLYLINE":
                    line = self._get_polylines(entity)
                    self.lines["POLYLINES"].append(line)

                elif entity.dxftype() == "INSERT":
                    line = self._get_insert(entity)
                    [self.lines["BLOCKS"].append(iline) for iline in line]
            except Exception as e:
                print(f"Error processing entity {entity}: {e}")

    def _get_lines(self, entity):
        """Get the 'lines' from the DXF entity.

        Args:
            entity (dxf.entity): considered entity

        Returns:
            shapely.LineString: shapely LineString object
        """
        linepoints = [entity.dxf.start, entity.dxf.end]
        line = LineString(linepoints)
        return line

    def _get_lwpolylines(self, entity):
        """Get the 'lwpolylines' from the DXF entity.

        Args:
            entity (dxf.entity): considered entity

        Returns:
            shapely.LineString or shapely.Polygon: shapely LineString or Polygon object
        """
        points = entity.get_points("xy")
        elevation = entity.dxf.elevation
        linepoints = [list(point) + [elevation] for point in points]
        if linepoints[-1] == linepoints[0]:
            line = Polygon(linepoints)
        else:
            line = LineString(linepoints)
        return line

    def _get_polylines(self, entity):
        """Get the 'polylines' from the DXF entity.

        Args:
            entity (ezdxf.entity): considered entity

        Returns:
            shapely.LineString or shapely.Polygon: shapely LineString or Polygon object
        """
        linepoints = [[point[0], point[1], point[2]] for point in entity.points()]
        if linepoints[-1] == linepoints[0]:
            line = Polygon(linepoints)
        else:
            line = LineString(linepoints)
        return line

    def _get_insert(self, entity):
        linepoints = []
        sub_entities = entity.virtual_entities()
        for sub_entity in sub_entities:
            if sub_entity.dxftype() == "LINE":
                linepoints.append(self._get_lines(sub_entity))
            elif sub_entity.dxftype() == "LWPOLYLINE":
                linepoints.append(self._get_lwpolylines(sub_entity))
            elif sub_entity.dxftype() == "POLYLINE":
                linepoints.append(self._get_polylines(sub_entity))
        return linepoints

    def _change_xy_to_yz(self, geom):
        if isinstance(geom, shapely.geometry.linestring.LineString):
            points = [arr for arr in geom.coords]
            coords = [(y, z) for x, y, z in points]
            return LineString(coords)
        else:
            print(f"Geometry type {type(geom)} not supported for change_xy_to_yz.")

    def _set_fig(self):
        fig, ax = plt.subplots()
        plt.grid(True)
        ax.set_aspect("equal")
        return fig, ax


def geojson_to_kml(file):
    kml = simplekml.Kml()
    gdf = gpd.read_file(file)
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Polygon":
            kml.newpolygon(name=str(i), outerboundaryis=geom.exterior.coords[:])
        elif geom.geom_type == "LineString":
            kml.newlinestring(name=str(i), coords=geom.coords[:])
    kml.save(file.replace(".geojson", ".kml"))


def lines_to_kml(x, y):
    kml = simplekml.Kml()
    for i in range(len(x)):
        line = kml.newlinestring(name=f"Line {i}", coords=list(zip(x[i], y[i])))
    kml.save("lines.kml")


def georeference_from_points(src1, src2, dst1, dst2):
    # Vectors in source and destination coordinate systems
    src_vector = src2-src1
    dst_vector = dst2-dst1

    # Calculate the scale factor & angles
    scale = np.linalg.norm(dst_vector) / np.linalg.norm(src_vector)
    angle_src = np.arctan2(src_vector[1], src_vector[0])
    angle_dst = np.arctan2(dst_vector[1], dst_vector[0])
    theta = angle_dst - angle_src

    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Affine coefficients for Shapely
    a = scale * cos_theta
    b = -scale * sin_theta
    c = scale * sin_theta
    d = scale * cos_theta

    # Translate src1 to origin
    xoff = dst1[0] - (a * src1[0] + b * src1[1])
    yoff = dst1[1] - (c * src1[0] + d * src1[1])

    return [a, b, c, d, xoff, yoff]


# %% Input
if __name__ == "__main__":
    filename = "../oostende.dxf"

    # Read in the DXF file
    dxf = DXFReader(filename)

    # Create GeoDataFrame from the lines
    dxf.entities_to_gpd()

    # Set CRS
    dxf.set_crs("EPSG:31370")

    # Transform CRS
    dxf.transform_crs("EPSG:4326")

    # Plot the figures
    # dxf.plot_dxf()

    # Extract lines from rectangle in figure
    # dxf.get_lines_from_rectangle()

    # Convert GeoJSON to KML
    # geojson_to_kml("lines_snippet_from_rectangle.geojson")

    # Extract lines by clicking in figure
    x_lines, y_lines = dxf.get_lines_from_fig()

    # Lines to kml
    lines_to_kml(x_lines, y_lines)