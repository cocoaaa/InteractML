import os, sys, time
import numpy as np
import scipy as sp
from scipy.signal import correlate2d
from utils import timeit
from grid import CartesianGrid

################################################################################
# Levelset method
################################################################################
class LevelSet():
    """LevelSet Evolution according to an initial-valued problem given by a PDE
    Args:
    - F (callable): takes a LevelSet object and time index and returns a np array 
    with the same shape as the levelset's grid
    """
    def __init__(self, xs, ys, step=1., t=0):
        self.w = w
        self.h = h
        self.grid = self.init_grid()#np.empty((h,w))
        
        self.step = step # grid step size
        self.t = t# current time        
            
    def init_grid(self):
        return test.astype(np.float) #todo
    
    
    def propagate(self, F, dt):
        """
        Equation 4.8 and 4.20
        For stability in computing the spatial gradients, use Eqn. 4.33
        """
        dxb, dxf, dyb, dyf = self.get_gradient()
        S = np.sign(F)
        
        dx = np.maximum(S*dxb, -S*dxf)
        dy = np.maximum(S*dyb, -S*dyf)
        
        dmag = np.sqrt(dx**2 + dy**2)
        
        # update phi
        self.grid -= dt* dmag * F
        
        # update time
        self.t += dt
    def advect(self, V, dt):
        """
        Args:
        - V (ndarray of shape (w,h,2)): containing x and y component of the vector
        field
        - dt (float): time step size
        """
        pass
    
    def reinit(self, method='sweep'):
        """
        Reset current grid (phi function) to satisfy Eikonal equality
        in Eqn. 4.12
        
        - method 
            - 'pde': solve eqn. 4.37 with current grid, until steady state
            - 'fmm': fast marching method
            - 'sweep' (default): paper [88]
            - 'exact': paper [64]
            
            Default is 'sweep'
        """
        pass

    def get_gradient(self, to_switch=True):
        return gradient(self.grid, to_switch)
    
    def get_curvature(self):
        return curvature(self.grid)

                   