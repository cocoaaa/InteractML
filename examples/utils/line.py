## 2D line
from vector import Vector as vec
import holoviews as hv
import pdb
    
###############################################################################
# 2 dimensional Line class
###############################################################################
class Line2d():
    """ 
    Representation of a line in two dimsional space defined by two end points
    """
    
    def __init__(self, p0, p1):
        """"
        Args:
        - p0 (2dim vec)
        - p1 (2dim vec)
        """
        assert p0.ndim == 2 and p1.ndim == 2
        self.p0 = p0
        self.p1 = p1
        
    def length(self):
        return (p0-p1).norm
    
    def slope(self):
        return (p1[1]-p0[1])/(p1[0]-p0[0])
    
    def unit_tangent(self):
        """Unit tangent(ie. velocity vector)
        """
        return (p1-p0).normalize()
        
    def unit_normal(self):
        x,y = self.unit_tangent()
        return vec(-y,x)
    
    def get_normal_band(self, distance):
        """
        Use the normal vector as the directional vector to construct a bbox around this line
        - distance (float)
        """
        return self.get_band(self.unit_normal(), distance)
    
    def get_band(self, direction, distance):
        """
        Args:
        - direction (2d vec): Doesn't have to be of unit length as we will do the normalization again.
        If the angle from self.tangent to direction vector is not in range [0,np.pi], 
        we negate the direction vector in order to preserve the ordering of returned band box
        - distance (positive float)
        
        Returns:
        - (top-left, bottom-left, bottom-right, top-right): a list of Vector that represents
        the band bbox
        """
        assert direction.ndim == 2 and distance >= 0
        direction = direction.normalize() 
        
        if self.unit_tangent().inner(direction) < 0:
            print("direction vector is flipped")
            direction = i
        b0 = self.p0 + direction
        b1 = self.p0 - direction
        b2 = self.p1 - direction
        b3 = self.p1 + direction
        
        return (b0, b1, b2, b3)
    
    def hvplot(self, **opts):
        return hv.Curve([self.p0, self.p1]).opts(**opts).opts(padding=0.1, aspect='equal')#,data_aspect=1))
    
    def __repr__(self):
        return f"Line2d({self.p0.values}, {self.p1.values})"
        
        
################################################################################
# Tests
################################################################################
def test_Line2d_constructor_1():
    p0 = vec(0,0)
    p1 = vec(2,1)
    l = Line2d(p0,p1)
    overlay = (
        l.hvplot() 
        * p0.hvplot(color='b', size=10) 
        * p1.hvplot(color='r', size=10) *
        p1.rotate(90).hvplot(color='g', size=10)
    )
    display(overlay)
    
def test_Line2d_constructor_2():
    p0 = vec(0,0)
    p1 = vec(2,1)
    l = Line2d(p0, p1.rotate(90))
    diaply(
        * p0.hvplot(color='b', size=10) 
        * p1.hvplot(color='r', size=10) 
        * p1.rotate(90).hvplot(color='g', size=10)
        * l.hvplot()
    )
    
def test_Line2d_tangent_and_normal():
    p0 = vec(0,0)
    p1 = vec(2,1)
    l = Line2d(p0, p1.rotate(90))
    t = l.unit_tangent()
    n = l.unit_normal()
    (
        l.hvplot() 
        * t.hvplot(color='r')
        * n.hvplot(color='g')

    )
def test_Line2d_get_normal_band_1():
    from shapely.geometry import Polygon

    p0 = vec(0,0)
    p1 = vec(0,1)

    l = Line2d(p0,p1)
    band = l.get_normal_band(distance=1)
    overlay = (
        l.hvplot() 
        * l.unit_normal().hvplot(color='r', size=10)
        * hv.Polygons([Polygon(band)]).opts(alpha=0.1)
    )
    display(overlay)
    
def test_Line2d_get_normal_band_2():
    from shapely.geometry import Polygon

    p0 = vec(0,0)
    p1 = vec(2,1)

    l = Line2d(p0,p1)
    band = l.get_normal_band(1)
    overlay = (
        l.hvplot() 
        * l.unit_normal().hvplot(color='r', size=10)
        * hv.Polygons([Polygon(band)]).opts(alpha=0.1)
    )
    display(overlay)
    
def run_tests():
    pass
    
    
###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    run_tests()