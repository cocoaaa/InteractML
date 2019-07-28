import os, time
import numpy as np
from scipy.signal import correlate2d
from levelset import Kernel

def diff1_central(grid):
    dy, dx = np.gradient(grid)
    return dx,dy

def diff1_bf(grid, switch=True):
    """
    Compute dxb, dxf, dyb, dyf where `b` and `f` refer to `backward` and 
    `forward` difference, respectively.
    
    It assumes Cartesian Coordinate System's axis direction, that is: 
    - xaxis increases as we move to the right
    - yaxis increasea as we move up 
    Note the yaxis's direction is the oppostie of numpy's row indexing order.
    In other words, the forward difference in y direction (ie. axis=0) would 
    be implemented with numpy arrays as:
    
    grady[j][i] = M[j-1][i] - M[j][i]
                  ---------
        this is "ahead" in cartisian coordinate system, although the indexing 
        to the numpy array is smaller


    Args:
    - grid (2dim np.array): represents the levelset function
    - switch (bool): Whether to use the forward difference when backword 
    difference  does not have valid values (ie. at the edge), and vice-versa. 
    True by default
        - If False, pad the edges with zero and apply the kernels with 
        `scipy.signal.correlated2d`. Refer to `gradx` and `grady` functions 

    Returns:
    - dxb, dxf, dyb, dyf (tuple of np.arrays):
    a tuple of four np.arrays each of which has the same shape as the input. 
    `b` and `f` indicate `backward` and `forward` difference, respectively.
    """
    dxb, dxf = gradx_bf(grid, switch)
    dyb, dyf = grady_bf(grid, switch)
    return (dxb, dxf, dyb, dyf)

def gradx_bf(M, switch):
    dxb = correlate2d(M, Kernel.xb, mode='same')
    dxf = correlate2d(M, Kernel.xf, mode='same')

    if switch:
        dxb[:,0] = dxf[:,0]
        dxf[:,-1] = dxb[:,-1]

    return dxb, dxf

def grady_bf(M, switch):
    dyb = correlate2d(M, Kernel.yb, mode='same')
    dyf = correlate2d(M, Kernel.yf, mode='same')

    if switch:
        dyb[-1] = dyf[-1]
        dyf[0] = dyb[0]

    return dyb, dyf

def diff2_central(grid):
    """Central Second-order difference, accuracy of second order
    """
    # todo: mix and match backford and forward diff1 
    pass


def curvature(grid):
    dx, dy = diff1_central(grid)
    d2x, d2y, dxdy = diff2_cenntrl(grid)
    
    dx2 = dx**2
    dy2 = dy**2
    
    k = (dx2*dy2 + dy2*d2x - 2*dx*dy*dxdy) / (np.sqrt(dx2 + dy2)*(dx2 + dy2))
    return k

    
    