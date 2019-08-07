import os, sys, time
import numpy as np
import scipy as sp
from scipy.signal import correlate2d
import holoviews as hv

from utils import timeit
from grid import CartesianGrid
import calculus as calc
################################################################################
# Levelset method
################################################################################
################################################################################
class LSEvolver(CartesianGrid):
    """
    Levelset propagator: A levelSet evolution solved by discrete time integration
    of a PDE with a given initial levelset function, \phi(t=0). 
    """
    def __init__(self, xs, ys, data=None, t=0):
        super().__init__(xs, ys, data)
        
        self.time = t #current time
        self.delta = np.inf # average change of LS function values between consecutive time stamps
        
    @timeit        
    def run(self, F, dt, pde_class, threshold=1e-3, maxIter=1e2, collect_every=50, sacred_run=None):
        """
        Args:
        - F (2d ndarray): same shape as self.data or broadcastable to self.data's shape(eg. a constant)
        - dt (flaot): time sampling period in continuous levelset space
        - pde_class (str): 'hyperbolic' or 'parabolic'
        - threshold (float): propagation stopping criterion based on the average change in consecutive phi values 
        - maxIter (int): maximum number of iterations for the propagate steps. This takes precedence over the threshold
        - collect_every (int): interval between collecting the levelset values for visualizing the process
        """
        count = 0
        deltas = {}
        phis = {}
        while self.delta > threshold:
            if count > maxIter: 
                print("MaxIter reached: ", count)
                break
            self.propagate(F, dt, pde_class, debug=False)
            deltas[self.time] = self.delta
            count += 1
            if count%collect_every == 0:
                print(f"Running {count}th iteration")
                phis[self.time] = self.data.copy()
                
        print(f"Ran for {count} steps, for total {self.time} periods")
        print(f"\taverage delta phi: {self.delta}")
        return deltas, phis

    def propagate(self, F, dt, pde_class, debug=False):
        """
        Equation 4.8 and 4.20
        1. Spatial gradient computation
        For stability in computing the spatial gradients, use Eqn. 4.33
        
        2. Temporal discretization
        To propagate the front over time, we neet to update the levelset values over time 
        according to a process. We express the process as a partial differential of the levelset 
        function (\phi) wrt time:
        
        \frac
        
        Our goal is to find \phi(t1), \phi(t2), ... given a initial \phi(t0)
        equatation
        This process is called "time integration" 
        
        Args:
        - pde_class (str): 'hyperbolic', 'parabolic'
            * (1) if F depends on at most order 1 derivatives of the levelset function phi 
            wrt space and time, the information propagation has a specific direction 
            (ie. "characteristics"), and we need to be careful about which gradient to 
            take -- backward, forward.  In this case, the levelset equation is 'hyperbolic', 
            which is a subclass of Hamilton-Jacobian equation. 
            
            * (2) if F depends on derivatives of order >= 2 (eg. F = alpha*curvature),
            then the information propagates from all directions, and we can use the 
            central finite difference method to compute the spatial gradients.
        """
        assert pde_class in ['hyperbolic','parabolic'], f"pde_class must be either 1 or 2: {pde_class}"
        
        if pde_class == 'hyperbolic':
            dxb, dxf, dyb, dyf = self.get_diff1_bf()
            
            if debug:
                overlay = (
                    hv.Image(self.data, label='phi') + hv.Image([])
                    + hv.Image(dxb, label='dx back') + hv.Image(dxf, label='dx forward')
                    + hv.Image(dyb, label='dy back') + hv.Image(dyf, label='dy forward')
                ).cols(2)
                display(overlay)

            S = np.sign(F)
            dx = np.maximum(S*dxb, -S*dxf)
            dy = np.maximum(S*dyb, -S*dyf)
            
        else: #pde_class == 'parabolic':
            dx,dy = self.get_diff1_central()

        dmag = np.sqrt(dx**2 + dy**2)
    
        # update phi
        dphi = dt * dmag * F
        self.data -= dphi 
        self.delta = abs(dphi.sum() / dphi.size)

        # update time
        self.time += dt

        
    def advect(self, V, dt):
        """
        Args:
        - V (ndarray of shape (w,h,2)): containing x and y component of the vector field
        - dt (float): time step size
        """
        if dt > min(self.dx, self.dy):
            #todo: print error but then make dt smaller smartly
            raise ValueError('dt should be smaller than x and y sample resolutions: ', dt)
        pass
    
    def reinit(self, method='sweep'):
        """
        Reset current phi values (in self.data) to satisfy Eikonal equality
        in Eqn. 4.12
        
        - method 
            - 'pde': solve eqn. 4.37 with current phi data, until steady state
            - 'fmm': fast marching method
            - 'sweep' (default): paper [88]
            - 'exact': paper [64]
            
            Default is 'sweep'
        """
        pass
    
    def get_diff1_bf(self, switch=True):
        dxb, dxf, dyb, dyf = calc.diff1_bf(self.data, switch)
        return dxb/self.dx, dxf/self.dx, dyb/self.dy, dyf/self.dy
    
    def get_diff1_central(self):
        dx, dy= calc.diff1_central(self.data)
        return dx/(2*self.dx), dy/(2*self.dy)
    
    def get_curvature(self):
        return curvature(self.data)
    
    def satisfies_cfl(self, V, dt, method="euler1"):
        """
        todo: rename to validate_dt?
        
        Note: this is applicable only when you choose to 'advect' your front. 
        - irrelevant to 'propagate' method.
        
        Check if the given dt satisifes the CFL condition for the stability of Euler forward
        time integration (ie. explicit method). In other words, this is to ensure the time 
        steps are small enough in comparison to the spatial sampling size (dx and dy) so that 
        the errors do not grow over time.
        
         $$ \frac{V \dot \Delta t}{\Delta x} < c $$
         
         where $c$ is the CFL-number which depends on the time integration method.
         
         In case of the explicit time integration method (eg. forward Euler and RK-schemes), 
         c = 1. Other cases (ie. for higher order methods), the c value can be very restrictive, 
         and it may be better to use an implicit time integration (eg. backward Euler)
         
         Args:
         - V (3 dim np.ndarray): 
             - first channel has the x component of the external velocity field
             - second channel has the y component of the external velocity field
         - method (str): time integration method and order. 
             - currently supports only "euler1" which indicates "first-order forward Euler" 
        """
        if method is not "euler1":
            raise ValueError("Currently only first-order forward euler method is supported")
        assert V.ndim == 3, f"V should be three-dimensional: {V.ndim}"
        Vx, Vy = V[...,0], V[...,1]
        
        Vx_satisfies = np.all(Vx < dt/self.dx)
        Vy_satisfies = np.all(Vy < dt/self.dx)
        is_valid = Vx_satisfies and Vy_satisfies
        if not is_valid:
            Vmax = V.max()
            dt = (min(self.dx, self.dy) / Vmax) - 1e-5
        return (is_valid, dt)
    
    # visualization helpers
    def visualize(self, deltas, phis, to_save, outname=None):
        hv.extension('matplotlib')

        tsteps= list(phis.keys())

        # hmap*contours 
        hmap = hv.HoloMap({t: hv.Image((self.xs, self.ys, phis[t])) for t in tsteps})
        contour_hmap =hv.operation.contours(hmap, levels=5)
        phi_hmap = hmap * contour_hmap.opts(framewise=True, fig_inches=10)
        hmap_point = hv.HoloMap({t: hv.Points([(t,deltas[t])])  for t in tsteps}).opts(color='r', s=15)

        # delta curve
        try: 
            deltas.pop(0.)
        except KeyError:
            print("deltas don't contain time=0 value")
            pass
        curve_hmap = (hv.Curve(deltas) *hmap_point).redim.label(x='time', y='delta')
        overlay_hmap = phi_hmap  + curve_hmap

        # save hmap overlay as gif animation
        if to_save:
            outname = outname or utils.get_temp_fname(prefix='ls_propagate', suffix='.gif')
            hv.save((overlay_hmap.opts(framewise=True, fig_inches=10)), outname, fps=1)
            print("Saved simulation as gif: ", outname)
        return overlay_hmap

    def get_dmap(self, deltas, phis):
        tsteps= list(phis.keys())

        # phi dmap
        dmap = hv.DynamicMap(lambda t: hv.Image((self.xs,self.ys,phis[t])), kdims='t').redim.values(t=tsteps)
        contour_dmap = hv.operation.contours(phi_dmap,levels=5).redim.values(t=list(phis.keys()))
        phi_dmap = (phi_dmap * contour_dmap).opts(img_opts, contour_opts)

        # delta curve
        try: 
            deltas.pop(0.)
        except KeyError:
            print("deltas don't contain time=0 value")
            pass

        curve = hv.Curve(deltas)
        dmap_point = hv.DynamicMap(lambda t: hv.Points([(t, deltas[t])]), kdims='t').redim.values(t=tsteps)
        delta_dmap = (curve * dmap_point.opts(color='red', size=5)).redim.label(x='time', y='delta')

        return phi_dmap + delta_dmap

