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

from nisp.utils import restore_checkpoint

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm

from utils import get_dataset
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    (
        obs_pcd,
        wall_coords,
        inner_coords,
        outer_coords, 
        deformation_3d,
        test_points_subset,
        E,
        P,
        t, 
        nu,
        a, 
        b,
        L,
        L_f,
        etc,
    ) = get_dataset(N=100000)
    if config.nondim == True:

        E = E / (L **2)   # rubber 0.1 GPa (N/m**2)
        P = P / (L **2)
        t = t * L# 0.3 mm
        # geom = dde.geometry.Interval(-1, 1)
        a, b = a*L, b*L
        wall_coords = wall_coords * L
        test_coords = test_points_subset * L
        obs_pcd = obs_pcd * L
        inner_coords = inner_coords * L
        outer_coords = outer_coords * L
        deformation_3d = deformation_3d * L
    
    # Initialize model
    model = models.Membrane(
        config,
        wall_coords,
        obs_pcd,
        test_coords,
        E, P, t, nu, a, b, L, L_f,
    )

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params


    # yhat = model.predict(x)/L
    # f_pred = model.f_pred_fn(params, test_coords[:,0], test_coords[:,1])
    u_pred = model.u_pred_fn(params, test_coords[:,0], test_coords[:,1])
    v_pred = model.v_pred_fn(params, test_coords[:,0], test_coords[:,1])
    w_pred = model.w_pred_fn(params, test_coords[:,0], test_coords[:,1])

    f_pred = model.f_pred_fn(params, test_coords[:,0], test_coords[:,1])

    color = abs(f_pred)
    color = color / max(color)
    color = cm.viridis(color).squeeze()[..., :3]


    x_hat = test_coords[:, 0]/L + u_pred/L
    y_hat = test_coords[:, 1]/L + v_pred/L
    z_hat = w_pred/L


    pcd = np.concatenate([x_hat.reshape(-1,1), 
                        y_hat.reshape(-1,1) , 
                        z_hat.reshape(-1,1) , color], axis = -1)
    save_pointcloud(pcd, filename=f'pred_best_force', save_path='result')

    pcd = np.concatenate([obs_pcd[:, 0:1]/L, 
                        obs_pcd[:, 1:2]/L , 
                        obs_pcd[:, 2:3]/L, np.zeros_like(obs_pcd)],axis = -1)
    save_pointcloud(pcd, filename=f'gt_best_force', save_path='result')




    ## Visualize result
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(test_coords[..., 0], 
               test_coords[..., 1], 
               z_hat, s = 0.5, c = 'r')

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
    plt.savefig(f"result/result_eval_force.png")
    plt.close()


    # Contact force 2D plot
    f_pred = f_pred * (L ** 2)

    s = x_hat[2] - x_hat[1] # pointcloud should be ordered
    cnt_fc = np.sum(abs(f_pred) * s**2)
    cnt_fc_gt = np.linalg.norm(etc["wrench_w"][:3])

    sc = plt.scatter(x_hat, y_hat, c = f_pred, 
                vmin = min(f_pred), vmax = max(f_pred),
                  s = 7, cmap='viridis')
    plt.colorbar(sc)
    plt.xlim(-0.065, 0.065)
    plt.ylim(-0.065, 0.065)
    plt.title(f"Contact force estimate {np.round(cnt_fc, 3)}[N], GT  { round(cnt_fc_gt, 3)}[N], [N/m^2]")
    plt.savefig("contact_force.png")
    plt.close()



    u_pred = model.u_pred_fn(params, deformation_3d[:,3], deformation_3d[:,4])
    v_pred = model.v_pred_fn(params, deformation_3d[:,3], deformation_3d[:,4])
    w_pred = model.w_pred_fn(params, deformation_3d[:,3], deformation_3d[:,4])
    f_pred = model.f_pred_fn(params, deformation_3d[:,3], deformation_3d[:,4])

    x_hat = deformation_3d[:, 3]/L + u_pred/L
    y_hat = deformation_3d[:, 4]/L + v_pred/L
    f_pred = f_pred * L ** 2 


    CONTACT_THRES = 2000
    DEPTH_THRES = 0.008
    w, h = etc["delta_depth"].shape[:2]
    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,4,1)
    ax1.imshow(etc["obs_color"].reshape(w, h, 3))


    ax1 = fig.add_subplot(1,4,2)
    mask_gt = np.repeat(etc["contact_mask_r"][..., np.newaxis], 3, axis = 2) * 255.
    mask_gt[..., [0,2]] = 0
    ax1.imshow(mask_gt)



    ax1 = fig.add_subplot(1,4,3)
    mask_thres_idx = np.where( etc["delta_depth"].reshape(-1) > DEPTH_THRES)[0]
    color_thres = np.zeros((len(pcd), 3))
    color_thres[mask_thres_idx,1] = 1.
    ax1.imshow(color_thres.reshape(etc["delta_depth"].shape[0], etc["delta_depth"].shape[1], -1))
    ax1.set_title(f"Depth Thresholding: {DEPTH_THRES}[m]")


    ax2 = fig.add_subplot(1,4,4)
    f_pred =abs(f_pred)
    color = cm.viridis(f_pred/max(f_pred))[..., :3]
    ax2.imshow(color.reshape(etc["delta_depth"].shape[0], 
                                   etc["delta_depth"].shape[1],
                                     -1))
    ax2.set_title("Contact Pressure")


    plt.savefig("mask prediction.png")
    plt.close()


