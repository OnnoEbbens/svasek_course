import xarray as xr
# import uxarray as ux 
import xugrid as xu #deltares
import numpy as np
import scipy.io as sio
import scipy.interpolate as sci
from scipy import sparse
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import glob
import datetime
#TODO: package maken van deze module
# from .plots import _plot_windrose

# loadmat() laadt een .mat bestand in en zet alle data in dictionaries
def loadmat(filename):
    """Improved loadmat (replacement for scipy.io.loadmat)
    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908
    """

    def _has_struct(elem):
        """Determine if elem is an array
        and if first array item is a struct
        """
        return isinstance(elem, np.ndarray) and (
            elem.size > 0) and isinstance(
            elem[0], sio.matlab.mat_struct)

    def _check_keys(d):
        """checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            elem = d[key]
            if isinstance(elem,
                          sio.matlab.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """A recursive function which constructs from
        matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem,
                          sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the
        elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem,
                          sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = sio.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def elem2p(trip,ze):
    npo = trip.max()
    ind = ~np.isnan(ze)
    t = trip[ind,:]
    ntr = t.shape[0]
    zp = np.zeros([npo, 1])

    I = t.transpose() - 1
    J = np.ones([3, 1]).astype(int) * range(ntr)
    A = sparse.coo_matrix((np.ones([ntr*3]), (I.ravel(),J.ravel())), shape=(npo,ntr))
    
    aa = np.ravel(np.sum(A, axis=1))
    at = np.divide(1, aa, where=aa>0)
    As = np.squeeze(np.asarray(at))
    nn = np.linspace(0, npo-1,npo).astype(int)
    B = sparse.coo_matrix((As, (nn,nn)), shape=(npo,npo))
    
    BA = B * A
    zp = BA * ze[ind]
    zp[aa == 0] = np.nan
    return zp

def datenum2datetime(datenum):
    return datetime.datetime.fromordinal(int(datenum)) + datetime.timedelta(days=datenum%1) - datetime.timedelta(days = 366)

class Model:
    def __init__(self, runid):
        
        if os.path.isfile(os.path.join(runid, 'results_merged.nc')):
            self.ds_map = xr.open_dataset(os.path.join(runid, 'results_merged.nc'))

        else:
            mesh  = loadmat(os.path.join(runid, 'Mesh01.mat'))
            flowfiles = sorted(glob.glob(os.path.join(runid, 'Flow*.mat')))
            time = []; U = []; V = []; H = []; Ue = []; Ve = []; He = [];
            for flowfile in flowfiles:
                flow = loadmat(flowfile)
                time.append(datenum2datetime(flow['mattime']))
                Ue.append(xr.DataArray(flow['U'], dims=["element"], attrs={'name': "horizontal current velocity at elements"}))
                Ve.append(xr.DataArray(flow['V'], dims=["element"], attrs={'name': "vertical current velocity at elements"}))
                He.append(xr.DataArray(flow['H'], dims=["element"], attrs={'name': "water level at elements"}))
                U.append(xr.DataArray(elem2p(mesh['tri'], flow['U']), dims=["node"], attrs={'name': "horizontal current velocity at nodes"}))
                V.append(xr.DataArray(elem2p(mesh['tri'], flow['V']), dims=["node"], attrs={'name': "vertical current velocity at nodes"}))
                H.append(xr.DataArray(elem2p(mesh['tri'], flow['H']), dims=["node"], attrs={'name': "water level at nodes"}))
            
            tri = xr.DataArray(mesh['tri'], dims=["element", "vertex"], attrs={'name': "triangles"})
            xe = xr.DataArray(mesh['xe'], dims=["element"], attrs={'name': "x-coordinate at elements"})
            ye = xr.DataArray(mesh['ye'], dims=["element"], attrs={'name': "y-coordinate at elements"})
            x = xr.DataArray(mesh['x'], dims=["node"], attrs={'name': "x-coordinate at nodes"})
            y = xr.DataArray(mesh['y'], dims=["node"], attrs={'name': "y-coordinate at nodes"})
            Ue = xr.concat(Ue, dim='time')
            Ve = xr.concat(Ve, dim='time')
            He = xr.concat(He, dim='time')
            U = xr.concat(U, dim='time')
            V = xr.concat(V, dim='time')
            H = xr.concat(H, dim='time')

            # ds_map = xr.Dataset({
            #     "Ue": Ue, "Ve": Ve, "He": He,
            #     "U": U, "V": V, "H": H
            #     },
            #     coords={
            #         "time": time,
            #         "node": np.arange(len(x)),
            #         "element": np.arange(len(tri)),
            #         "vertex": np.arange(3),
            #         "tri": tri-1, "xe": xe, "ye": ye, "x": x, "y": y,
            #     })
            # ds_map.to_netcdf(os.path.join(runid, 'results_merged.nc'), 'w')
            # self.ds_map = ds_map
            
            ds = xr.Dataset({
                "Mesh2_face_nodes": (["nMesh2_face", "nMaxNodesPerFace"], mesh["tri"]),
                "Mesh2_node_x": (["nMesh2_node"], mesh["x"]),
                "Mesh2_node_y": (["nMesh2_node"], mesh["y"]),
                "U": (["nMesh2_face"], Ue.values[0]),
                "V": (["nMesh2_face"], Ve.values[0]),
                "H": (["nMesh2_face"], He.values[0]),
            })
            # Add required UGRID attributes
            ds["Mesh2_face_nodes"].attrs.update({
                "cf_role": "face_node_connectivity",
                "long_name": "Face to node connectivity",
                "start_index": 1,
                "mesh": "Mesh2"
            })
            ds["Mesh2_node_x"].attrs.update({
                "standard_name": "projection_x_coordinate",   # or "projection_y_coordinate"
                "units": "meters"
            })
            ds["Mesh2_node_y"].attrs.update({
                "standard_name": "projection_y_coordinate",   # or "projection_y_coordinate"
                "units": "meters"
            })
            # Add global mesh_topology variable
            ds["Mesh2"] = xr.DataArray(0)
            ds["Mesh2"].attrs.update({
                "cf_role": "mesh_topology",
                "topology_dimension": 2,
                "node_coordinates": "Mesh2_node_x Mesh2_node_y",
                "face_node_connectivity": "Mesh2_face_nodes"
            })
            ds.attrs["Conventions"] = "UGRID-1.0"

            # Create the dataset
            #optie 1
            # ugrid = ux.open_grid(ds)
            # uds = ux.UxDataset.from_xarray(ds)

            #optie 2
            uds = xu.UgridDataset(ds)

            # Get a cross-section
            


    def interpolate(self, time, x, y, elem2p=True):
        #TODO: interpoleren over de tijd
        #TODO: nearest-optie?
        #TODO: loop over alle variabelen met coordinates "nodes"
        # maak een nieuwe xarray.Dataset met coordinaten x,y
        
        # Create triangulation object
        trimesh = mtri.Triangulation(self.ds['x'], self.ds['y'], self.ds['tri'])
        fU = mtri.LinearTriInterpolator(trimesh, self.ds['U'].isel(time=0))
        fV = mtri.LinearTriInterpolator(trimesh, self.ds['V'].isel(time=0))
        fH = mtri.LinearTriInterpolator(trimesh, self.ds['H'].isel(time=0))
        U_interp = fU(x, y)

        #interpolate with griddata
        U_interp = sci.griddata((self.ds['xe'], self.ds['ye']))

        return U_interp

    def plot(self, time=0):
        self.fig = plt.figure(figsize=(5, 4), dpi=100)
        self.axes = self.fig.add_axes([0.05,0.1,0.77,0.8])
        trimesh = mtri.Triangulation(self.ds_map['x'], self.ds_map['y'], self.ds_map['tri'])
        self.axes.tripcolor(trimesh, self.ds_map['H'].values)
    # def plot_windrose(self):
    #     """test"""
    #     _plot_windrose(self.ds)

if __name__ == '__main__':
    runid = r'T:\Python\SvasekScripts\TPak\FECSM2021_METEO_20250402_1800'
    model = Model(runid)
    xg = np.linspace(model.ds_map.xe.min().values, model.ds_map.xe.max().values, 100)
    yg = np.linspace(model.ds_map.ye.min().values, model.ds_map.ye.max().values, 100)
    # model_klein = model.interpolate(0, xg, yg)
    model.plot()
    # model_klein.plot_windrose()