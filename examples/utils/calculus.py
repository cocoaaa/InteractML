import os, time
import numpy as np
from scipy.signal import correlate2d
from levelset import Kernel

def gradient(grid, switch=True):
    """
    Compute dxp, dxm, dyp, dym.
    Assumes Cartesian Coordinate System's axis direction,
    that is: 
    - xaxis increases as we move to the right
    - yaxis increasea as we move up 
    Note the yaxis's direction is the oppostie of numpy's row indexing order
    In other words, the forward difference in y direction (ie. axis=0) would 
    be implemented with numpy arrays as:
    
    grady[j][i] = M[j-1][i] - M[j][i]
                  ---------
        this is "ahead" in cartisian coordinate system, although the indexing in
         numpy array goes the other way around


    Args:
    - grid (2dim np.array): represents the levelset function
    - switch (bool): Use the forward difference when backword difference 
    does not have valid values (ie. at the edge), and vice-versa. 
    to_switch is True by default

    Returns:
    - dxb, dxf, dyb, dyf (tuple of np.arrays):
    a tuple of four np.arrays of the same shape as the input grid where
    `b` and `f` indicate `backward` and `forward` difference, respectively
    """
    dxb, dxf = gradx(grid, switch)
    dyb, dyf = grady(grid, switch)
    return (dxb, dxf, dyb, dyf)

def gradx(M, switch):
    dxb = correlate2d(M, Kernel.xb, mode='same')
    dxf = correlate2d(M, Kernel.xf, mode='same')

    if switch:
        dxb[:,0] = dxf[:,0]
        dxf[:,-1] = dxb[:,-1]

    return dxb, dxf

def grady(M, switch):
    dyb = correlate2d(M, Kernel.yb, mode='same')
    dyf = correlate2d(M, Kernel.yf, mode='same')

    if switch:
        dyb[-1] = dyf[-1]
        dyf[0] = dyb[0]

    return dyb, dyf

def curvature(grid):
    pass