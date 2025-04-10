import xarray as xr
# import uxarray as ux 
import xugrid as xu #deltares
import numpy as np
import scipy.io as sio
import scipy.interpolate as sci
from scipy import sparse
import os
import sys
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
    def __init__(self, uds_map):
        self.uds_map = uds_map
            
    @classmethod
    def from_runid(cls, runid):
        if os.path.isfile(os.path.join(runid, 'results_merged.nc')):
            uds_map = xu.open_dataset(os.path.join(runid, 'results_merged.nc'))
        else:
            mesh  = loadmat(os.path.join(runid, 'Mesh01.mat'))
            flowfiles = sorted(glob.glob(os.path.join(runid, 'Flow*.mat')))
            time = []; U = []; V = []; H = []; Ue = []; Ve = []; He = [];
            for flowfile in flowfiles:
                flow = loadmat(flowfile)
                time.append(datenum2datetime(flow['mattime']))
                Ue.append(flow['U'])
                Ve.append(flow['V'])
                He.append(flow['H'])   
            Ue = np.array(Ue)
            Ve = np.array(Ve)
            He = np.array(He)
            
            ds = xr.Dataset({
                "Mesh2_face_nodes": (["nMesh2_face", "nMaxNodesPerFace"], mesh["tri"]),
                "Mesh2_node_x": (["nMesh2_node"], mesh["x"]),
                "Mesh2_node_y": (["nMesh2_node"], mesh["y"]),
                "U": (["time","nMesh2_face"], Ue),
                "V": (["time","nMesh2_face"], Ve),
                "H": (["time","nMesh2_face"], He),
            })
            ds = ds.assign_coords(time=("time", time))
            ds["time"].attrs.update({
                "standard_name": "time"
            })
            uds_map = cls._add_ugrid_properties(ds)
            uds_map.ugrid.to_netcdf(os.path.join(runid, 'results_merged.nc'), 'w')
            
        return cls(uds_map)
    
    @classmethod
    def from_dataset(cls, ds):
        #TODO: uitkomst moet een dataset zijn, geen ugrid-dataset
        # uds_map = cls._add_ugrid_properties(ds)
        x = ds['Mesh2_x'].values
        y = ds['Mesh2_y'].values
        nx = np.unique(x).size
        ny = np.unique(y).size
        x = x.reshape(ny, nx)[0,:]
        y = y.reshape(ny, nx)[:,0]
        H_reshaped = ds['H'].values.reshape(ds['H'].shape[0], ny, nx)
        ds_map = xr.DataArray(
            H_reshaped,
            dims=["time", "Mesh2_y", "Mesh2_x"],
            coords={"time":(["time"], ds['time'].values),
                    "Mesh2_x": (["Mesh2_x"], x), 
                    "Mesh2_y": (["Mesh2_y"], y)},
            name="H"
        )
        ds_map = ds_map.to_dataset()
        return cls(ds_map)
    
    @staticmethod
    def _add_ugrid_properties(ds):
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
        #TODO crs
        # ds["crs"] = xr.DataArray(0)
        # ds["crs"].attrs.update({
        #     "grid_mapping_name": "transverse_mercator",
        #     "semi_major_axis": 6378137.0,
        #     "inverse_flattening": 298.257223563,
        #     "latitude_of_projection_origin": 0.0,
        #     "longitude_of_central_meridian": 15.0,
        #     "scale_factor_at_central_meridian": 0.9996,
        #     "false_easting": 500000.0,
        #     "false_northing": 0.0,
        #     "units": "m"
        # })
        # ds["H"].attrs["grid_mapping"] = "crs"
        # ds["U"].attrs["grid_mapping"] = "crs"
        # ds["V"].attrs["grid_mapping"] = "crs"
        uds_map = xu.UgridDataset(ds)
        return uds_map

    def interpolate(self, time, x, y):
        #TODO: nearest-optie?
        #TODO: loop over alle variabelen met coordinates "nodes"
        #TODO interpoleer van een ugrid naar een ander ugrid (e.g.: SWAN-grid naar FINEL)
        # maak een nieuwe xarray.Dataset met coordinaten x,y
        
        # # Create triangulation object
        # trimesh = mtri.Triangulation(self.ds['x'], self.ds['y'], self.ds['tri'])
        # fU = mtri.LinearTriInterpolator(trimesh, self.ds['U'].isel(time=0))
        # fV = mtri.LinearTriInterpolator(trimesh, self.ds['V'].isel(time=0))
        # fH = mtri.LinearTriInterpolator(trimesh, self.ds['H'].isel(time=0))
        # U_interp = fU(x, y)
        # #interpolate with griddata
        # U_interp = sci.griddata((self.ds['xe'], self.ds['ye']))

        #interpolate with sel (uses barycentric interpolation)
        H_interp = self.uds_map['H'].ugrid.sel(x=x, y=y)
        if isinstance(H_interp, xr.DataArray):
            H_interp = H_interp.to_dataset()
        return self.from_dataset(H_interp)
    
    def plot(self, var, time=0):
        # self.fig = plt.figure(figsize=(5, 4), dpi=100)
        # self.axes = self.fig.add_axes([0.05,0.1,0.77,0.8])
        # trimesh = mtri.Triangulation(self.ds_map['x'], self.ds_map['y'], self.ds_map['tri'])
        # self.axes.tripcolor(trimesh, self.ds_map['H'].values)

        plt.figure()
        if isinstance(self.uds_map, xu.UgridDataArray):
            self.uds_map.isel(time=time).ugrid.plot(vmin=-3, vmax=3)
        elif isinstance(self.uds_map, xu.UgridDataset):
            self.uds_map[var].isel(time=time).ugrid.plot(vmin=-3, vmax=3)
        elif isinstance(self.uds_map, xr.DataArray):
            self.uds_map.isel(time=time).plot(vmin=-3, vmax=3)
        elif isinstance(self.uds_map, xr.Dataset):
            self.uds_map[var].isel(time=time).plot(vmin=-3, vmax=3)


    # def plot_windrose(self):
    #     """test"""
    #     _plot_windrose(self.ds)

    # def exceedance_plot(self):
    #     """test"""
    #     pass

    # def history(self):
    #     """test"""
    #     return self.uds_his

if __name__ == '__main__':
    runid = r'T:\Python\SvasekScripts\TPak\FECSM2021_METEO_20250402_1800'
    model = Model.from_runid(runid)
    sys.exit(1)
    model.plot('H', time=4)

    xg = np.linspace(model.uds_map.Mesh2_node_x.min().values, model.uds_map.Mesh2_node_x.max().values, 100)
    yg = np.linspace(model.uds_map.Mesh2_node_y.min().values, model.uds_map.Mesh2_node_y.max().values, 100)
    model_klein = model.interpolate(0, xg, yg)
    model_klein.plot('H', time=4)

    xg = np.linspace(640000, 690000, 200)
    yg = np.linspace(5880000, 5930000, 200)
    model_klein2 = model.interpolate(0, xg, yg)
    model_klein2.plot('H', time=4)

    # model_klein.plot_windrose()