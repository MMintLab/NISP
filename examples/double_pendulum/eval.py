import os
import copy
import ml_collections

import matplotlib.pyplot as plt
import numpy as np

from nisp.utils import restore_checkpoint
import models
from utils import *


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    errors = []
    q_errors = []
    psi_errors = []
    phi_errors = []

    for a, b, c, d in config.data_idx:
        # Initialize W&B
        wandb_config = config.wandb
        wandb_config.name = f"{a}_{b}_{c}_{d}_{config.noise}"

        # Get dataset
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        with open( os.path.join(save_path, f"{a}_{b}_{c}_{d}_{config.noise}.pkl"), "rb") as f:
            data = pickle.load(f)

        # Initialize model
        L = jnp.array(data["L"])
        u_control = jnp.array(data["u_control"])
        x_without_noise = jnp.array(data["x_without_noise"])
        t = jnp.array(data["t"])
        q0 = jnp.array(data["q0"])
        q1 = jnp.array(data["q1"])
        q2 = jnp.array(data["q2"])
        m0, m1, m2, l, g = jnp.array(data["m0"]), jnp.array(data["m1"]), jnp.array(data["m2"]),jnp.array(data["l"]),  jnp.array(data["g"])

        model = models.DoublePendulum(
            config,
            m0, m1, m2, l, g,
            L, u_control,
        )

        # Restore checkpoint
        ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # unseen timestamps
        t_ori = copy.copy(t)
        t = t[: len(u_control), :]
        t_dense = 0.5 * (t[:-1, :] + t[1:, :])[:, 0]

        # unseen time stamps - external source estimations
        gt_forcing_term = u_control[:, 0] / L
        f_pred_dense = model.f_pred_fn(params, t_dense) / L
        q0_pred_dense = model.q0_pred_fn(params, t_dense) / L
        q1_pred_dense = model.q1_pred_fn(params, t_dense) / L
        q2_pred_dense = model.q2_pred_fn(params, t_dense) / L

        # training time stamps - external source estimations
        f_pred = model.f_pred_fn(params, t[:, 0]) / L
        q0_pred = model.q0_pred_fn(params, t[:, 0]) / L
        q1_pred = model.q1_pred_fn(params, t[:, 0]) / L
        q2_pred = model.q2_pred_fn(params, t[:, 0]) / L

        # external source errors
        fig = plt.figure()
        error = abs((gt_forcing_term[1:] - np.array(f_pred[1:])))
        errors.append(error)

        # state estimation errors
        q_error = abs(x_without_noise[1:, 0] - np.array(q0_pred_dense))
        psi_error = abs(x_without_noise[1:, 1] - np.array(q1_pred_dense))
        phi_error = abs(x_without_noise[1:, 2] - np.array(q2_pred_dense))

        q_errors.append(q_error)
        psi_errors.append(psi_error)
        phi_errors.append(phi_error)


        plt.plot(t[1:], gt_forcing_term[1:])
        plt.plot(t[1:], np.array(f_pred)[1:])
        plt.xlabel("time [s]")
        plt.ylabel("force [N]")
        # plt.title(f"force estimation error: {error}[m]")
        plt.legend(["ground truth", "estimated"])

        plt.show()
        plt.savefig(f"result/force_est_{wandb_config.name}.png")
        plt.close()

        # state estimation error
        q_gt = np.concatenate([q0, q1, q2], axis=1)
        q_est = np.stack([q0_pred, q1_pred, q2_pred], axis=1)
        # error = jnp.mean( (q_gt - q_est)**2, axis = 0)

        # Plot: measurement - full state estiation
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
        plt.legend(["q[m]", "phi", "psi"])
        plt.show()
        plt.savefig(f"result/v_{wandb_config.name}.png")
        plt.close()

