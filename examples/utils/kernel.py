import numpy as np

################################################################################
# Kernels for levelset methods
################################################################################
class CartesianKernel():
    xb = np.atleast_2d([-1,1,0])
    xf = np.atleast_2d([-1,1]) # same as np.atleast_2d([0,-1,1])
   
    yb = np.atleast_2d([1,-1]).T
    yf = np.atleast_2d([1,-1,0]).T
    
    diff2_xx = np.atleast_2d([1,-2,1])
    diff2_yy = diff2_xx.T
    diff2_xy = .25*np.atleast_2d([[-1,0,1],
                             [0,0,0],
                             [1,0,-1]])
    
