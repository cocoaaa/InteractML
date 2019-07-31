import holoviews as hv
import math
import pdb
    
###############################################################################
# Cartesian Discrete Grid
###############################################################################
class CartesianGrid():
    """
    Representation of a discrete grid in Cartesian coordinate system
    """
    def __init__(self, xs, ys, data=None):
        """
        Assumes regularly spaced x-coordinates and y-coordinates
        That is, 
        - xcoords in xs increases as we iterate over xs
        - ycoords in ys decreases as we iterate over ys
        This ordering will be validated at initiaization time
        
        - self.data is also in Cartesian coordinate system, which has an 
        important implication that its axis is the opposite of row indexing
        to the numpy array 
        """
        if ys[0] < ys[-1]:
            raise ValueError("ys must be in decreasing order")
        self.xs = xs
        self.ys = ys
        self.xmin, self.xmax = xs[0], xs[-1]
        self.ymin, self.ymax = ys[-1], ys[0]
        
        # levelset sampling resolution in x direction (for discretization)
        self.dx = xs[1] - xs[0] 
        self.dy = -(ys[1] - ys[0])
        
        # Underlying data storage as numpy array
        self.height = len(ys)
        self.width = len(xs)
        self.data = data
        
    ###############################################################################
    # Coordinate conversions across three possible spaces
    # 1. continuous cartesian (x,y): most "mathematical" domain
    # 2. discrete cartesian [i,j]: indexing into xs and ys to retrieve sampled x,y coord
    # 3. Index [r,c] to numpy array: indexing into the storage which is a numpy array
    ###############################################################################
    def xy2ij(self,x,y, do_better=True):
        """
        Choose the right bin index to xs and ys.
        Assume xs and ys are both Cartesian coordinate's axes. That is, 
        xs's values increase as we iterate over it, but 
        ys's values decreases as we iterate over it.
        
        First we get the bin index as if ys is also in the same ordering as xs, 
        and then flip it to compensate for our assumption
        """
        ys = self.ys[::-1]

        if x < self.xmin or x > self.xmax:
            raise ValueError(f'x out of bound of [{self.xmin, self.xmax}]: {x}')
        if y < self.ymin or y > self.ymax:
            raise ValueError(f'y out of bound of [{self.ymin, self.ymax}]: {y}')
        i = math.floor( (x - self.xmin) / self.dx)
        j = math.floor( (y - self.ymin) / self.dy )
        
        # To be more accurate compute distance two both sides of the elements
        # and assign to the bin that has a closer value of x-coord (or y-coord)
        # In other words, check if taking `floor` to get the bin index actually
        # resulted in the best assignment 
        if i < self.width-1:
            diff_left = abs(x - self.xs[i])
            diff_right = abs(self.xs[i+1] - x)
            i = i if diff_left < diff_right else i+1
            
        if j < self.height-1:
            diff_left = abs(y - ys[j])
            diff_right = abs(ys[j+1] - y)
            j = j if diff_left < diff_right else j+1
        
        return i, self.height-1-j
    
    def ij2cr(self, i, j):
        """
        Convert indices to xs and ys to column and row into underlying numpy data storage
        """
        assert isinstance(i, int) and isinstance(j,int)
#         return i, self.height-1-j
        return i,j
    
    def xy2cr(self,x,y):
        i,j = self.xy2ij(x,y)
        return self.ij2cr(i,j)
    
    def get_value(self, x, y):
        """
        Retrieve value at the discrete location of (continuous cartesian coordinate) x,y
        The goal is to abstract away the discrete vs continous domain access, as well as
        cartesian vs. numpy array indexing conversions
        """
        c, r = self.xy2cr(x,y)
        return self.data[r,c]
    
    def set_value(self, x,y, val):
        c, r = self.xy2cr(x,y)
        self.data[r,c] = val

    def set_values(self, xs, ys, vals):
        pass
    
    def reevaluate(self, zfunc):
        self.data = zfunc(self.xs, self.ys)
        
    def hvplot(self, **opts):
        return hv.Image((self.xs, self.ys, self.data)).opts(**opts)
        
    def __repr__(self):
        desc = f"""
        CartesianGrid
        - xlim: ({self.xmin},{self.xmax})
        - dx: {self.dx:.3f}
        - ylim: ({self.ymin},{self.ymax})
        - dy: {self.dy:.3f}
        - storage height, width: {self.height}, {self.width}
        """
        return desc
    
    
        
def test_grid_constructor():
    xs = np.linspace(-1,1,10)
    ys = np.linspace(-1,1,10)
    g = CartesianGrid(xs, ys)
    print(g)
    
def test_grid_conversion():
    xs = np.linspace(-1,1,10)
    ys = np.linspace(-1,1,10)
    g = CartesianGrid(xs, ys)
    x, y = 0.5, 1
    i,j = g.xy2ij(x,y)

    print(f'x,y: {x}, {y}')
    print(f'i,j: {i}, {j}')
    
def test_grid_reevaluate():
    from sdfs import sdUnitHline, eval_sdf
    from functools import partial

    xs = np.linspace(-1,1,10)
    ys = np.linspace(-1,1,10)
    g = CartesianGrid(xs, ys)
    zfunc = partial(eval_sdf, sdFunc=sdUnitHline)
    g.reevaluate(zfunc)

    display(g.reevaluate(zfunc))
