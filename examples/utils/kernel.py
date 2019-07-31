import numpy as np

################################################################################
# Kernels for levelset methods
################################################################################
class CartesianKernel():
    xb = np.atleast_2d([-1,1,0])
    xf = np.atleast_2d([-1,1]) # same as np.atleast_2d([0,-1,1])
   
    yb = np.atleast_2d([1,-1]).T
    yf = np.atleast_2d([1,-1,0]).T
    
    
