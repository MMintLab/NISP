from functools import partial
import time
import os
import numpy as np

from absl import logging

from flax.training import checkpoints
from mmint_tools.camera_tools.pointcloud_utils import save_pointcloud

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.tree_util import tree_map

import scipy.io
import ml_collections

import wandb

import models

from jaxpi.utils import restore_checkpoint

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from utils import get_dataset
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    (
        coords,
        wall_coords,
        E,
        P,
        t, 
        nu,
        a, 
        b 
    ) = get_dataset(10000)

    if config.nondim == True:
        L = 15. # Scalling factor 
        E = E / (L **2) # rubber 0.1 GPa (N/m**2)
        P =  P / (L **2)
        t = t * L# 0.3 mm
        # geom = dde.geometry.Interval(-1, 1)
        a, b = a*L, b*L
        wall_coords = wall_coords * L
        coords = coords * L
    
    # Initialize model
    model = models.Membrane(
        config,
        wall_coords,
        E, P, t, nu, a, b, L,
    )

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    print(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params


    w_pred = model.w_pred_fn(params, coords[:,0] ,coords[:,1])
    u_pred = model.u_pred_fn(params, coords[:,0], coords[:,1])
    v_pred = model.v_pred_fn(params, coords[:,0], coords[:,1])

    D =  E * t **3 / (12 * (1 - nu**2))
    w_gt = P/ (64 * D) * (coords[:,0]**2 + coords[:,1]**2  - a**2) **2


    w_error = np.array(abs(w_pred-w_gt))
    w_pred = np.array(w_pred)
    # yhat = model.predict(x)/L
    print(max(u_pred/L * 1e3) , max(v_pred/L * 1e3 ))

    x_hat = (coords[:, 0] + u_pred)/L #+ yhat[:, 0]
    y_hat = (coords[:, 1] + v_pred)/L #+ yhat[:, 1]
    z_hat = w_pred/L


    error_max = (0.001 * L)

    color = w_error / error_max
    import matplotlib.cm as cm
    color = cm.cool(color).squeeze()[..., :3]
    # print(color.shape)   
    
    # yhat = model.predict(x)/L
    x_hat = coords[:, 0]/L #+ yhat[:, 0]
    y_hat = coords[:, 1]/L #+ yhat[:, 1]
    z_hat = w_pred/L

    print("z range",  np.amax(z_hat), np.amin(z_hat) )
    print("error",  np.mean(w_error/L), np.std(w_error/L) )


    pcd = np.concatenate([x_hat.reshape(-1,1), 
                        y_hat.reshape(-1,1) , 
                        z_hat.reshape(-1,1) , color], axis = -1)
    save_pointcloud(pcd, filename=f'pred_best', save_path='.')



    # GT test 
    gt_test = np.concatenate([coords, w_gt.reshape(-1,1)], axis = -1)
    gt_test = gt_test/L
    color =  np.zeros(gt_test.shape)
    color[:,0] = np.ones_like(color[:,0])
    save_pointcloud( np.concatenate([gt_test, np.zeros_like(gt_test)], axis = -1), 
                    filename='gt_best', save_path='.')
    

    ## Visualize result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(gt_test[..., 0], gt_test[..., 1], gt_test[..., 2], s = 0.5, c = 'r')

    ax.scatter(x_hat, y_hat, z_hat, s = 0.8)
    ellipse_points = load_ellipse(2*a, 2*b, num_points=1000)

    ax.scatter(ellipse_points[...,0], ellipse_points[...,1], ellipse_points[...,2], s = 2)
    ax.view_init(elev=20, azim=60, roll=0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_zlim(0, 0.14)
    plt.xlim(-0.07, 0.07)
    plt.ylim(-0.07, 0.07)
    plt.show()
    plt.savefig(f"result_eval.png")
