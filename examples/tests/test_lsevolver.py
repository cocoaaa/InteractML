import os, sys
from pathlib import Path
import numpy as np
import joblib
import holoviews as hv

# Set path to the utils directory
CURR_DIR = Path(__file__).resolve().parent
print('Current file loc: ', CURR_DIR)
UTILS_DIR = (CURR_DIR.parent/'utils').resolve()
# UTILS_DIR = Path('../utils').resolve()# within interactive sheel
assert UTILS_DIR.exists()
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
    print(f"Added {str(UTILS_DIR)} to sys.path")

import utils
from levelset import LSEvolver
import sdfs
from sacred import Experiment
ex = Experiment('HelloWorld')

@ex.config
def cfg():
    # Levelset discretization parameters
    n_points = 100
    xlim = (-2,2)
    ylim = (-2,2)
    sdf = sdfs.sdUnitCircle
#     sdf = sdfs.sdStar1
    
    # Propagation parameters
    F = 1
    dt = 1e-2
    pde_class = 'hyperbolic'
    collect_every = 1
    maxIter = 10
    threshold = 1e-6

@ex.capture
def run_test(n_points, xlim, ylim, sdf, 
             F, dt, pde_class, collect_every, maxIter, threshold,
             to_save_results, to_save_gif):
    
    xs = np.linspace(*xlim, n_points)
    ys = np.linspace(*ylim, n_points)[::-1]
    zz = sdfs.eval_sdf(xs, ys, sdf)
    sdf_name = sdf.__name__
    
    ls = LSEvolver(xs,ys,zz)
    deltas, phis = ls.run(F, dt, pde_class, 
                          threshold=threshold, maxIter=maxIter, collect_every=collect_every)
    
    
    # save result
    if to_save_results:
        out_dir = Path(f'../data/intrim/{utils.get_timestamp()}')
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
            print("Created a new ouput folder: ", out_dir)
        
        out_pkl = str( (out_dir/ f'{sdf_name}_f_{F}_dt_{dt}_t_0_{ls.time:.1f}.pkl').absolute())
        joblib.dump((deltas,phis), out_pkl)

    # visualization 
    gif_dir = Path(f'../outputs/levelset/{utils.get_timestamp()}')
    if not gif_dir.exists():
        gif_dir.mkdir(parents=True)
        print("Created a new ouput folder for gif: ", gif_dir)

                                          
    gif_name = f'{sdf_name}_f_{F}_dt_{dt}_t_0_{ls.time:.1f}.gif'
    gif_name = str( (gif_dir/gif_name).absolute())
    overlay = ls.visualize(deltas, phis, to_save_gif, gif_name)
#     display(overlay)

@ex.automain
def main():
    run_test(to_save_results=True, to_save_gif=True)