import os, time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

import imageio
import pdb

################################################################################
# IO Helpers
################################################################################
def tensor_from_ndarr(arr):
    """
    Reorder the axes of numpy array to torch's convention: (C,H,W)
    Note this function handles a single arr, not a list of arrays

    Args:
    - arr (np.ndarray of 2dim or 3dim):
        - if 2dim, H,W
        - if 3dim, H,W,C
    Returns:
    - 3dim tensor
        - if input is 2dim array, return (1, H,W) size tensor
        - if input is 3dim, return (3,H,W)
    """
    
    if arr.ndim not in [2,3]:
        raise ValueError(f'Input array must of 2 or 3 dim: {arr.dim}')
    if arr.ndim == 2:
        return torch.from_numpy(np.expand_dims(arr,0))
    else:
        return torch.from_numpy(np.transpose(arr, (2,0,1)))
        
def tensor_from_video(fname):
    reader = imageio.get_reader(fname, 'ffmpeg')
    imgs = np.transpose( np.stack(list(reader.iter_data()), axis=0), (0,3,1,2))
    imgs = np.expand_dims(imgs, 0)
    assert imgs.ndim == 5, \
    f'Video data must be 5 dimensional: batchsize, timesteps, C,H,W: {imgs.dim}'
    
    return torch.from_numpy(imgs)

################################################################################
# Tests
################################################################################
def test_tensor_from_video():
    fname = '../outputs/levelset/2019-08-02/sdStar1_f_-1_dt_0.001_t_0_0.3.gif'
    t = tensor_from_video(fname)
          
def test_all():
    test_tensor_from_video()
    
if __name__ == '__main__':
    test_all()
