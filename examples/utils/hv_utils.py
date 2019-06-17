import os, sys, time
import numpy as np
import scipy as sp
import pandas as pd
import geopandas as gpd
    
from pathlib import Path
from pprint import pprint

import holoviews as hv
import xarray as xr

from holoviews import opts
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from holoviews.streams import Stream, param
from holoviews import streams

import geoviews as gv
import geoviews.feature as gf
from geoviews import tile_sources as gvts

import cartopy.crs as ccrs
import cartopy.feature as cf


def get_element_plot_mapping(backend='bokeh'):
    """
    shows the mapping of holoviews element and plot 
    for the backend
    - backend (str): eg: 'bokeh','matplotlib'
    """
    pd.set_option('max_colwidth', 100)
    regi = hv.Store.registry[backend]
    df = pd.DataFrame({
            'element': list(map(str, regi.keys())),
            'plot': list(map(str, regi.values()))})
    return hv.Table(df).opts(width=700)

def relabel_elements(ndoverlay, labels):
    """
    ndOverlay is indexed by integer
    labels (str or iterable of strs)
    length of hv elements in the overlay must equal to the length of labels
    """
    import holoviews as hv
    from itertools import cycle
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels, list) and len(labels) != len(ndoverlay):
        raise ValueError('Length of the labels and ndoverlay must be the same')
        
        
    it = cycle(labels) 
    relabeled = hv.NdOverlay({i: ndoverlay[i].relabel(next(it)) for i in range(len(ndoverlay))})
    return relabeled

def ranges2lbrt(x_range, y_range):
    """
    Convert x_range and y_range to a tuple of 4 floats indicating
    l,b,r,t 
    Args:
    - x_range, y_range (tuple)
    Returns
    - lbrt (tuple of 4 floats)
    """
    minx, maxx = x_range
    miny, maxy = y_range
    return (minx, miny, maxx, maxy)

def lbrt2ranges(lbrt):
    """
    Converts a tuple of 4 floats indicating l,b,r,t coordinate
    to x_range and y_range
    """
    minx, miny, maxx, maxy = lbrt
    return ( (minx, maxx), (miny, maxy) )


def test_ranges2lbrt():
    x_range = (-1,1); y_range=(-10,10)
    lbtc = ranges2lbrt(x_range, y_range)
    assert lbtc == (-1,-10,1,10)
def test_lbrt2ranges():
    lbrt = (-1,-10,1,10)
    assert lbrt2ranges(lbrt) == ((-1, 1), (-10, 10))