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
import matplotlib.cm as cm

from utils import get_dataset
from utils import *
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    (L, t, g, m0, m1, l, q0, q1, u_control) = get_dataset(N=2000)  # 100000

    # Initialize model
    model = models.VehicleSuspension(
        config,
        m0,
        m1,
        l,
        g,
        L,
        u_control,
    )

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    gt_forcing_term = u_control / L
    f_pred = model.f_pred_fn(params, t[:, 0]) / L
    q0_pred = model.q0_pred_fn(params, t[:, 0]) / L
    q1_pred = model.q1_pred_fn(params, t[:, 0]) / L

    fig = plt.figure()
    error = jnp.mean((gt_forcing_term - np.array(f_pred)) ** 2)
    plt.plot(t, gt_forcing_term)
    plt.plot(t, np.array(f_pred))
    plt.xlabel("time [s]")
    plt.ylabel("force [N]")
    plt.title(f"force estimation error: {error}[m]")
    plt.legend(["ground Truth", "estimated"])

    plt.show()
    plt.savefig(f"result/force_est.png")
    plt.close()

    fig = plt.figure()
    q_gt = np.concatenate([q0, q1], axis=1)
    q_est = np.stack([q0_pred, q1_pred], axis=1)
    error = jnp.mean((q_gt - q_est) ** 2, axis=0)

    plt.plot(t, q_gt)
    plt.plot(t, q_est, "--")
    plt.xlabel("time [s]")
    plt.ylabel("state [m]")
    plt.title(f"regression error: {error}[m]")
    plt.legend(["q0_gt", "q1_gt", "q2_gt", "q0_est", "q1_est", "q2_est"])
    plt.show()
    plt.savefig(f"result/v.png")
    plt.close()

    # task performance
    control_fn_est = lambda t: model.f_pred_fn(params, t)
    x0 = np.array([0, np.pi / 2 - 0.3, 0.0, 0.0])
    x = odeint(
        right_hand_side_custom, x0, t[:, 0], args=(parameter_vals, control_fn_est)
    )
    animate_pendulum(np.array(t[:, 0]), x, 0.5, filename="behavior_clone.gif")
