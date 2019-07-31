import numpy as np
import sdfs
import pdb
    
###############################################################################
# Sample Images for sanity checking 
###############################################################################
class LSTestSample():
    
    @staticmethod
    def unit_circle():
        n_points = 100
        xlim = (-2,2)
        ylim = (-2,2)
        sdf = sdfs.sdUnitCircle
        xs = np.linspace(*xlim,n_points)
        ys = np.linspace(*ylim,n_points)[::-1]
        zz = sdfs.eval_sdf(xs,ys,sdf)
        return (xs,ys,zz)
        
    @staticmethod
    def unit_hline():
        n_points = 100
        xlim = (-2,2)
        ylim = (-2,2)
        sdf = sdfs.sdUnitHline
        xs = np.linspace(*xlim,n_points)
        ys = np.linspace(*ylim,n_points)[::-1]
        zz = sdfs.eval_sdf(xs,ys,sdf)
        return (xs,ys,zz)

        
    @staticmethod
    def unit_star1():
        n_points = 100
        xlim = (-2,2)
        ylim = (-2,2)
        sdf = sdfs.sdStar1
        xs = np.linspace(*xlim,n_points)
        ys = np.linspace(*ylim,n_points)[::-1]
        zz = sdfs.eval_sdf(xs,ys,sdf)
        return (xs,ys,zz)

        
        
    @staticmethod
    def unit_star2():
        n_points = 100
        xlim = (-2,2)
        ylim = (-2,2)
        sdf = sdfs.sdStar2
        xs = np.linspace(*xlim,n_points)
        ys = np.linspace(*ylim,n_points)[::-1]
        zz = sdfs.eval_sdf(xs,ys,sdf)
        return (xs,ys,zz)


    @staticmethod
    def linear_array():
        h,w = 100,100
        xs = np.linspace(-1,1,num=w)
        ys = np.linspace(-1,1,num=h)[::-1]
        zz = np.empty((w,h))
        for i in range(len(xs)):
            for j in range(len(ys)):
                zz[j,i] = ys[j] 
        return (xs,ys,zz)

    @staticmethod
    def generate_pattern(pattern, repeats):
        """
        Generate a new 2D np.array by repeating the pattern in repeats[0] times columnwise, 
        and repeats[1] times rowwise
        - Eg: If pattern = [0, 0, 1, 1], generate_pattern(pattern, (2,1)) returns:
        
        Args:
        - pattern (numpy.array)
        - repreats (tuple of length 2): specifies the number of times to repeat columnwise and rowwise
        """
        return np.tile(pattern, repeats)
    
    @staticmethod
    def pulse_grid():
        """
        Returns 12x12 pulse arrays
        """
        pattern = np.array([0,0,0,1,1,1])
        return np.tile(pattern, (12,2))
        
            
    @staticmethod
    def manual_grid1():
        return np.array([[1, 2, 5, 10, 100],
                         [0, -1, 10, -3, 9],
                        [100, -20, 8, 10,-10]], dtype = np.float)