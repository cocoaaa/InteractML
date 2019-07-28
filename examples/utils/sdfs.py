# A collection of Signed Distance Functions in 2D
# src: https://is.gd/yLR7tF

from vector import Vector as vec

import numpy as np
from functools import partial
import functools

import pdb
###############################################################################
# Evaluate SDF on a grid 
###############################################################################
def eval_sdf(xs, ys, sdFunc):
    zz = np.empty( (len(ys), len(xs)) )
    
    for j in range(len(ys)):
        for i in range(len(xs)):
            q = vec(xs[i],ys[j])
            zz[j,i] = sdFunc(q)
    return zz


###############################################################################
# SDFs
###############################################################################
def sdCircle( query, radius):
    """
    Args:
    - query (vector.Vector): point to be evaluated at
    - radius (float) : radius of the circle
    """
    return query.norm() - radius

def sdLine( query, a, b):
    """
    Args:
    - query (vec): query point
    - a (vec): one endpoint of the line
    - b (vec): the other endpoint of the line
    """
    qa = query - a
    ba = b - a
    h = np.clip( qa.inner(ba)/ba.inner(ba), 0.0, 1.0)
    return (qa - ba*h).norm()


###############################################################################
# Triangles
###############################################################################
def sdEquilateralTriangle(query):
    """
    Signed distance function for a unit length equilateral triangle centered at the origin
    """
    if len(query) != 2:
        raise ValueError(f"Input vector must be 2-dimensional: {len(query)}")
    k = 3**.5
    query = vec(np.abs(query[0]) - 1.0, query[1] + 1.0/k)
    
    if query[0] + k*query[1] > 0.:
        query = vec(query[0] - k*query[1], -k*query[0]-query[1])/2.
    query[0] -= np.clip(query[0], -2., 2.)
    return - query.norm()*np.sign(query[1])

def sdTriangle(query, v0, v1, v2):
    """
    Args:
    - query (vec): query point
    - v0, v1, v2 (vec): vertices of the triangle
    """
    e0 = v1-v0
    e1 = v2-v1
    e2 = v0-v2
    
    d0 = query - v0
    d1 = query - v1
    d2 = query - v2
    
    qv0 = d0 - e0*np.clip( e0.inner(d0) / e0.inner(e0), 0., 1.0)
    qv1 = d1 - e1*np.clip( e1.inner(d1) / e1.inner(e1), 0., 1.0)
    qv2 = d2 - e2*np.clip( e2.inner(d2) / e2.inner(e2), 0., 1.0)
    
    s = np.sign( e0.cross(e2) )
    
    dist2_x = min(qv0.inner(qv0), qv1.inner(qv1), qv2.inner(qv2))
    dist2_y = min(s*d0.cross(e0), s*d1.cross(e1), s*d2.cross(e2))
    
    return -np.sqrt(dist2_x) * np.sign(dist2_y)

def sdStar(query, radius, n, m):
    """
    Args:
    - query (vec)
    - radius (float)
    - n (int)
    - m (float)
    """
    # Next 4 lines can be precomputed for a given shape
    an = np.pi/float(n)
    en = 2*np.pi/m
    acs = vec(np.cos(an), np.sin(an))
    ecs = vec(np.cos(en), np.sin(en))
    
    bn = np.mod(np.arctan2(*query.values), 2.0*an) - an
    query = query.norm() * vec(np.cos(bn), np.abs(np.sin(bn)))
    query = query - radius*acs
    query = query + ecs * np.clip(- query.inner(ecs), 0.0, radius*acs[1]/ecs[1])
    return query.norm() * np.sign(query[0])


###############################################################################
# Useful, wrapped sdfs as partial functions
###############################################################################
# Unit horizontal line from the origin (ie. line between (0,0) and (1,0)
sdUnitHline = partial(sdLine, a=vec(0.,0.), b=vec(1.0, 0.0))

# Unit circle centered at the origin
sdUnitCircle = partial(sdCircle, radius=1.0)

# Variants of stars
sdStar1 = partial(sdStar, radius=1, n=5, m=5.)
sdStar2 = partial(sdStar, radius=1, n=10, m=3.)








###############################################################################
# Test
###############################################################################
def test_sdCircle():
    q = vec(1.,0.)
    r = 1
    assert np.isclose([sdCircle(q,r)], [0])
    
def test_sdLine():
    q = vec(1.,0.)
    a = vec(0., 0.)
    b = vec(1.0, 0.)
    
    assert np.isclose([sdLine(q,a,b)], [0])
def test_sdTriangle():
    q = vec(0.,-1.)
    v0, v1, v2 = vec(0,0), vec(1,0), vec(0.5, 0.5)
    assert np.isclose( [sdTriangle(q, v0, v1, v2)], [-1.])

def run_tests():
    test_sdCircle()
    test_sdLine()
    
    
###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    run_tests()
# query = vec(1.,0.)
# a = vec(0., 0.)
# b = vec(1.0, 0.)

# qa = query - a
# ba = b - a
# h = np.clip( qa.inner(ba)/ba.inner(ba), 0.0, 1.0)
# val = (qa - ba*h).norm()
# print(val)
# # return val