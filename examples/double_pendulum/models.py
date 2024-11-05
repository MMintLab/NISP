from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian
from jax.experimental.jet import jet
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from nisp.models import ForwardBVP
from nisp.evaluator import BaseEvaluator
from nisp.utils import ntk_fn
from jax.experimental.host_callback import call

import matplotlib.cm as cm
from mmint_tools.camera_tools.pointcloud_utils import save_pointcloud
from nisp.chamfer_distance import chamfer_distance_jit


class VehicleSuspension(ForwardBVP):
    def __init__(
        self,
        config,
        m0,
        m1,
        m2,
        l,
        g,
        L,
        u_control,
    ):
        super().__init__(config)
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g
        self.L = L
        self.u_control = u_control

        # Predict functions over batch
        self.q0_pred_fn = vmap(self.q0_net, (None, 0))
        self.q1_pred_fn = vmap(self.q1_net, (None, 0))
        self.q2_pred_fn = vmap(self.q2_net, (None, 0))
        self.f_pred_fn = vmap(self.f_net, (None, 0))

        self.r_pred_fn = vmap(self.r_net, (None, 0))

        # self.x_iv_fn = vmap(self.x_iv, (None, 0))
        # self.f_iv_fn = vmap(self.f_iv, (None, 0))

    def neural_net(self, params, t):
        t = jnp.stack([t])
        outputs = self.state.apply_fn(params, t)
        q0 = outputs[0]
        q1 = outputs[1]
        q2 = outputs[2]
        f = outputs[3]
        return q0, q1, q2, f

    def q0_net(self, params, t):
        q0, _, _, _ = self.neural_net(params, t)
        return q0

    def q1_net(self, params, t):
        _, q1, _, _ = self.neural_net(params, t)
        return q1

    def q2_net(self, params, t):
        _, _, q2, _ = self.neural_net(params, t)
        return q2

    def f_net(self, params, t):
        _, _, _, f = self.neural_net(params, t)
        return f

    def get_residual(self, params, t):
        f = self.f_net(params, t)
        q0 = self.q0_net(params, t)
        q1 = self.q1_net(params, t)
        q2 = self.q2_net(params, t)

        M11 = self.m0 + self.m1 + self.m2
        M12 = -self.l * (self.m1 + self.m2) * jnp.sin(q1)
        M13 = -self.l * self.m2 * jnp.sin(q2)
        M21 = -self.l * (self.m1 + self.m2) * jnp.sin(q1)
        M22 = self.l**2 * (self.m1 + self.m2)
        M23 = self.l**2 * self.m2 * jnp.cos(q1 - q2)
        M31 = -self.l * self.m2 * jnp.sin(q2)
        M32 = self.l**2 * self.m2 * jnp.cos(q1 - q2)
        M33 = self.l**2 * self.m2
        mass_matrix = jnp.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]])

        q0_fn = lambda t: self.q0_net(params, t)
        _, (q0_t, q0_tt) = jet(q0_fn, (t,), [[1.0, 0.0]])

        q1_fn = lambda t: self.q1_net(params, t)
        _, (q1_t, q1_tt) = jet(q1_fn, (t,), [[1.0, 0.0]])

        q2_fn = lambda t: self.q2_net(params, t)
        _, (q2_t, q2_tt) = jet(q2_fn, (t,), [[1.0, 0.0]])

        f1 = (
            self.l * q1_t**2 * jnp.cos(q1) * (self.m1 + self.m2)
            + self.l * self.m2 * q2_t**2 * jnp.cos(q2)
            + f
        )
        f2 = -self.l * (
            self.g * self.m1 * jnp.cos(q1)
            + self.g * self.m2 * jnp.cos(q1)
            + self.l * self.m2 * q2_t**2 * jnp.sin(q1 - q2)
        )
        f3 = (
            self.l
            * self.m2
            * (-self.g * jnp.cos(q2) + self.l * q1_t**2 * jnp.sin(q1 - q2))
        )
        forcing_term = jnp.array([[f1], [f2], [f3]])

        u_dot = jnp.array([[q0_tt], [q1_tt], [q2_tt]])
        residual = mass_matrix @ u_dot - forcing_term
        return residual[0, 0], residual[1, 0], residual[2, 0]

    def r_net(self, params, t):
        residual_1, residual_2, residual_3 = self.get_residual(params, t)
        residual = residual_1**2 + residual_2**2 + residual_3**2
        return residual

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # coords = batch["coords"]
        t = batch["time"]
        t_dense = batch["query"]

        q0 = batch["q0"]
        q1 = batch["q1"]
        q2 = batch["q2"]

        q0_pred_bc = self.q0_pred_fn(params, t[:, 0])
        q0_loss = jnp.mean((q0.squeeze() - q0_pred_bc) ** 2)

        q1_pred_bc = self.q1_pred_fn(params, t[:, 0])
        q1_loss = jnp.mean((q1.squeeze() - q1_pred_bc) ** 2)

        q2_pred_bc = self.q2_pred_fn(params, t[:, 0])
        q2_loss = jnp.mean((q2.squeeze() - q2_pred_bc) ** 2)

        # PDE residuals
        r_in_pred = self.r_pred_fn(params, t_dense[:, 0])
        r_loss = jnp.mean(r_in_pred)

        loss_dict = {
            "q0_loss": q0_loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "r_loss": r_loss,
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        u_in_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.wall_coords[:, 0], self.inflow_coords[:, 1]
        )
        v_in_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_net, params, self.wall_coords[:, 1], self.inflow_coords[:, 1]
        )

        w_in_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.w_net, params, self.wall_coords[:, 1], self.inflow_coords[:, 1]
        )

        r_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.r_net, params, batch[:, 0], batch[:, 1]
        )

        ntk_dict = {
            "u_in": u_in_ntk,
            "v_in": v_in_ntk,
            "w_in": w_in_ntk,
            "r": jnp.zeros_like(r_ntk),
        }

        return ntk_dict


class VehicleSuspensionEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, t, gt):
        # gt_forcing_term  = 18 * jnp.exp(-3 * t)
        f_pred = self.model.f_pred_fn(params, t[:, 0])
        return jnp.mean((gt - f_pred) ** 2)

    def __call__(self, state, batch, t):
        self.log_dict = super().__call__(state, batch)
        gt = batch["u_control"]
        self.log_errors(state.params, t, gt)
        self.log_dict

        return self.log_dict
