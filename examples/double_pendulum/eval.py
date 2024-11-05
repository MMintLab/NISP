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
import copy
import wandb

import models

from nisp.utils import restore_checkpoint

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm

from utils import get_dataset, right_hand_side_custom
from utils import *
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse


def evaluate(config: ml_collections.ConfigDict, workdir: str):

    errors = []
    q_errors = []
    psi_errors = []
    phi_errors = []
    for a, b, c, d in config.data_idx:
        # Initialize W&B
        wandb_config = config.wandb
        # wandb_config.name = f"{a}_{b}_{c}_{d}_noise3_2_64"
        wandb_config.name = f"{a}_{b}_{c}_{d}_noise_1"

        # Get dataset
        (L, t, t_dense, g, m0, m1, m2, l, q0, q1, q2, u_control, x_without_noise) = (
            get_dataset(N=151, a=a, b=b, c=c, d=d)
        )  # 100000

        # Initialize model
        model = models.VehicleSuspension(
            config,
            m0,
            m1,
            m2,
            l,
            g,
            L,
            u_control,
        )

        # Restore checkpoint
        ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params
        t_ori = copy.copy(t)
        t = t[: len(u_control), :]
        t_dense = 0.5 * (t[:-1, :] + t[1:, :])[:, 0]

        gt_forcing_term = u_control[:, 0] / L
        f_pred_dense = model.f_pred_fn(params, t_dense) / L
        q0_pred_dense = model.q0_pred_fn(params, t_dense) / L
        q1_pred_dense = model.q1_pred_fn(params, t_dense) / L
        q2_pred_dense = model.q2_pred_fn(params, t_dense) / L

        f_pred = model.f_pred_fn(params, t[:, 0]) / L
        q0_pred = model.q0_pred_fn(params, t[:, 0]) / L
        q1_pred = model.q1_pred_fn(params, t[:, 0]) / L
        q2_pred = model.q2_pred_fn(params, t[:, 0]) / L

        fig = plt.figure()
        error = abs((gt_forcing_term[1:] - np.array(f_pred[1:])))
        errors.append(error)

        q_error = abs(x_without_noise[1:, 0] - np.array(q0_pred_dense))
        psi_error = abs(x_without_noise[1:, 1] - np.array(q1_pred_dense))
        phi_error = abs(x_without_noise[1:, 2] - np.array(q2_pred_dense))

        q_errors.append(q_error)
        psi_errors.append(psi_error)
        phi_errors.append(phi_error)

        # error = jnp.mean( (gt_forcing_term[:-1] - np.array(f_pred_dense))**2)

        plt.plot(t[1:], gt_forcing_term[1:])
        plt.plot(t[1:], np.array(f_pred)[1:])
        plt.xlabel("time [s]")
        plt.ylabel("force [N]")
        plt.title(f"force estimation error: {error}[m]")
        plt.legend(["ground truth", "estimated"])

        plt.show()
        plt.savefig(f"result/force_est_{wandb_config.name}.png")
        plt.close()

        fig = plt.figure()
        q_gt = np.concatenate([q0, q1, q2], axis=1)
        q_est = np.stack([q0_pred, q1_pred, q2_pred], axis=1)
        # error = jnp.mean( (q_gt - q_est)**2, axis = 0)

        plt.plot(t_dense, q0_pred_dense, color="orange")
        plt.plot(t_dense, q1_pred_dense, color="cornflowerblue")
        plt.plot(t_dense, q2_pred_dense, color="limegreen")
        plt.scatter(t_ori, q_gt[:, 0], color="red", s=3.0)
        plt.scatter(t_ori, q_gt[:, 1], color="mediumblue", s=3.0)
        plt.scatter(t_ori, q_gt[:, 2], color="darkgreen", s=3.0)

        # plt.plot(t_dense, q0_pred_dense, color='orange')

        # plt.scatter(t, q_est, '--')
        plt.xlabel("time [s]")
        plt.ylabel("state [m]")
        # plt.title(f"regression error: {erroqr}[m]")
        plt.legend(["q[m]", "phi", "psi"])
        plt.show()
        plt.savefig(f"result/v_{wandb_config.name}.png")
        plt.close()
        # # task performance
        # control_fn_est = lambda t : model.f_pred_fn(params, t)
        # x0 = np.array([0,
        #             np.pi / 2 - 0.3,
        #             np.pi / 2 - 0.3,
        #             1.,
        #             1. ,
        #             1.])
        # x = odeint(right_hand_side_custom, x0, t[:,0], args=(parameter_vals, control_fn_est))
        # animate_pendulum(np.array(t[:,0]),
        #                 x, 0.5, filename="behavior_clone.gif")
