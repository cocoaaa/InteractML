import os, time
import numpy as np
from scipy.signal import correlate2d
from utils import clip_close_values
from kernel import CartesianKernel 
from samples import LSTestSample

import holoviews as hv

def diff1_central(grid, postproc=clip_close_values):
    dy, dx = np.gradient(grid)
    
    if postproc:
        # Note: negate dy since np.gradient 
        dy, dx = postproc(-dy), postproc(dx)
    return dx,dy

def diff1_bf(grid, switch=True, postproc=clip_close_values):
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
    dxb, dxf = gradx_bf(grid, switch, postproc)
    dyb, dyf = grady_bf(grid, switch, postproc)
    
    return (dxb, dxf, dyb, dyf)

def gradx_bf(M, switch, postproc=clip_close_values):
    """
    - M: original array 
    - switch (bool): if True, use forward difference at the invalid backward 
    different regions, and vice versa
    - postproc (callable): Callable to apply to the output of correlate2d 
    before returning the final output
    """
    dxb = correlate2d(M, CartesianKernel.xb, mode='same')
    dxf = correlate2d(M, CartesianKernel.xf, mode='same')

    if switch:
        dxb[:,0] = dxf[:,0]
        dxf[:,-1] = dxb[:,-1]

    if postproc:
        dxb = postproc(dxb)
        dxf = postproc(dxf)
        
    return dxb, dxf

def grady_bf(M, switch, postproc=clip_close_values):
    dyb = correlate2d(M, CartesianKernel.yb, mode='same')
    dyf = correlate2d(M, CartesianKernel.yf, mode='same')

    if switch:
        dyb[-1] = dyf[-1]
        dyf[0] = dyb[0]
        
    if postproc:
        dyb = postproc(dyb)
        dyf = postproc(dyf)
        
    return dyb, dyf

def diff2_central(grid, postproc=clip_close_values):
    """Central Second-order difference, accuracy of second order
    
    - postproc (callable): Callable to apply to the output of the differentiation operations
    before returning the final output
    """
    # todo: mix and match backford and forward diff1 
    pass


def curvature(grid):
    dx, dy = diff1_central(grid)
    d2x, d2y, dxdy = diff2_central(grid)
    
    dx2 = dx**2
    dy2 = dy**2
    
    k = (dx2*dy2 + dy2*d2x - 2*dx*dy*dxdy) / (np.sqrt(dx2 + dy2)*(dx2 + dy2))
    return k

###############################################################################
# Tests
###############################################################################
def generate_grad_test(grid):
    dxb, dxf = gradx_bf(grid, True)
    dyb, dyf = grady_bf(grid, True)
    bounds = (0,0,*grid.shape)
    gridstyle = {
        'grid_line_color': 'black', 
        'minor_grid_line_color':'lightgray',
    }
    base = hv.Image(grid, bounds=bounds, label='original') 
    overlay = (
        base + hv.Image(dxb, bounds=bounds, label='dxb') 
        + base + hv.Image(dxf, bounds=bounds, label='dxf')
        + base + hv.Image(dyb, bounds=bounds, label='dyb') 
        + base + hv.Image(dyf, bounds=bounds, label='dyf')
    ).cols(2)
    
    return overlay.opts(
        hv.opts.Image(cmap='gray', alpha=0.9,
                   show_grid=True, gridstyle=gridstyle)
    )

def test_grad_on_pulse_x():
    grid = LSTestSample.pulse_grid()
    display(generate_grad_test(grid))

def test_grad_on_pulse_y():
    grid = LSTestSample.pulse_grid().T
    display(generate_grad_test(grid))

def test_grad_on_linear_array():
    grid = LSTestSample.linear_array()
    display(generate_grad_test(grid))
    
def run_grad_tests():
    test_grad_on_pulse_x()
    test_grad_on_pulse_y()

    
    
# diff1 tests
def generate_diff1_bf_test(grid):
    dxb, dxf, dyb, dyf = diff1_bf(grid)
    bounds = (0,0,*grid.shape)
    gridstyle = {
        'grid_line_color': 'black', 
        'minor_grid_line_color':'lightgray',
    }
    bounds = (0,0,*grid.shape)
    gridstyle = {
        'grid_line_color': 'black', 
        'minor_grid_line_color':'lightgray',
    }
    base = hv.Image(grid, bounds=bounds, label='original') 
    overlay = (
        base + hv.Image(dxb, bounds=bounds, label='dxb') 
        + base + hv.Image(dxf, bounds=bounds, label='dxf')
        + base + hv.Image(dyb, bounds=bounds, label='dyb') 
        + base + hv.Image(dyf, bounds=bounds, label='dyf')
    ).cols(2)
    
    return overlay.opts(
        hv.opts.Image(cmap='gray', alpha=0.9,
                   show_grid=True, gridstyle=gridstyle)
    )

def test_diff1_bf_on_pulse_x():
    grid = LSTestSample.pulse_grid()
    display(generate_diff1_bf_test(grid))
            
def test_diff1_bf_on_linear_array():
    *_, grid = LSTestSample.linear_array()
    display(generate_diff1_bf_test(grid))
            
def test_diff1_central_on_pulse_x():
    grid = LSTestSample.pulse_grid()
    dx, dy = diff1_central(grid)
    bounds = (0,0,*grid.shape)
    gridstyle = {
        'grid_line_color': 'black', 
        'minor_grid_line_color':'lightgray',
    }
    base = hv.Image(grid, bounds=bounds, label='original') 
    overlay = (
        base + hv.Image(dx, bounds=bounds, label='dx') 
        + base + hv.Image(dy, bounds=bounds, label='dy')
    ).cols(2)
    
    return overlay.opts(
        hv.opts.Image(cmap='gray', alpha=0.9,
                   show_grid=True, gridstyle=gridstyle)
    )

def run_test():
    run_grad_tests()

###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    run_tests()
    
    