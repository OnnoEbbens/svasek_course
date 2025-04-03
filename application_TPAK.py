import xarray as xr
import numpy as np
import scipy.io as sio
from scipy import sparse
import os
import matplotlib.tri as mtri

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

runid = r'T:\Python\SvasekScripts\TPak\FECSM2021_METEO_20250402_1800'

class Model:
    def __init__(self, runid):
        mesh  = loadmat(os.path.join(runid, 'Mesh01.mat'))
        flow1 = loadmat(os.path.join(runid, 'Flow_20250402_180000.mat'))

        tri = xr.DataArray(mesh['tri'], dims=["element", "vertex"], name="triangles")
        xe = xr.DataArray(mesh['xe'], dims=["element"], name="x-coordinate at elements")
        ye = xr.DataArray(mesh['ye'], dims=["element"], name="y-coordinate at elements")
        x = xr.DataArray(mesh['x'], dims=["node"], name="x-coordinate at nodes")
        y = xr.DataArray(mesh['y'], dims=["node"], name="y-coordinate at nodes")
        Ue = xr.DataArray(flow1['U'], dims=["element"], name="horizontal current velocity at elements")
        Ve = xr.DataArray(flow1['V'], dims=["element"], name="vertical current velocity at elements")
        He = xr.DataArray(flow1['H'], dims=["element"], name="water level at elements")
        U = xr.DataArray(elem2p(mesh['tri'], flow1['U']), dims=["node"], name="horizontal current velocity at nodes")
        V = xr.DataArray(elem2p(mesh['tri'], flow1['V']), dims=["node"], name="vertical current velocity at nodes")
        H = xr.DataArray(elem2p(mesh['tri'], flow1['H']), dims=["node"], name="water level at nodes")

        ds = xr.Dataset({
            "tri": tri-1, "xe": xe, "ye": ye, "x": x, "y": y,
            "U": Ue, "V": Ve, "He": He,
            "U": U, "V": V, "H": H
            },
            coords={
                "node": np.arange(len(x)),
                "element": np.arange(len(tri)),
                "vertex": np.arange(3)
            })
        self.ds = ds

    def interpolate(self, var, xg, yg):
        # Create triangulation object
        triang = mtri.Triangulation(self.ds['x'], self.ds['y'], self.ds['tri'])
        fU = mtri.LinearTriInterpolator(triang, self.ds['U'])
        fV = mtri.LinearTriInterpolator(triang, self.ds['V'])
        fH = mtri.LinearTriInterpolator(triang, self.ds['H'])
        U_interp = fU(xg, yg)