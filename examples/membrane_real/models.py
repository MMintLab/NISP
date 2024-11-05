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

# from mmint_tools.camera_tools.pointcloud_utils import save_pointcloud
from nisp.chamfer_distance import chamfer_distance_jit, chamfer_distance_directional_jit


class Membrane(ForwardBVP):
    def __init__(
        self,
        config,
        wall_coords,
        bd_coords,
        coords,
        E,
        P,
        t,
        nu,
        a,
        b,
        L,
        L_f,
    ):
        super().__init__(config)

        self.E = E  # 1.0 # rubber 0.1 GPa
        self.nu = nu  # 0.5 # Latex behaves like an incompressible
        self.P = P  # 2.757
        self.t = t  # 0.0003
        self.D = self.E * self.t**3 / (12 * (1 - self.nu**2))
        self.nu = nu
        self.a = a
        self.b = b
        self.L = L
        self.L_f = L_f

        # Initialize coordinates
        self.wall_coords = wall_coords
        self.bd_coords = bd_coords
        self.coords = coords

        # Predict functions over batch
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.w_pred_fn = vmap(self.w_net, (None, 0, 0))
        self.f_pred_fn = vmap(self.f_net, (None, 0, 0))

        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))
        self.rp_pred_fn = vmap(self.rp_net, (None, 0, 0))
        self.r_nf_pred_fn = vmap(self.r_net_nf, (None, 0, 0))
        self.rp_nf_pred_fn = vmap(self.rp_net_nf, (None, 0, 0))
        self.f_hessian_fn = vmap(self.f_hessian, (None, 0, 0))

        self.gelu = jax.nn.gelu

    def neural_net(self, params, x, y):

        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        w = outputs[2]
        f = outputs[3]

        return u, v, w, f

    def u_net(self, params, x, y):
        u, _, _, _ = self.neural_net(params, x, y)
        return u

    def v_net(self, params, x, y):
        _, v, _, _ = self.neural_net(params, x, y)
        return v

    def w_net(self, params, x, y):
        _, _, w, _ = self.neural_net(params, x, y)
        return w

    def f_net(self, params, x, y):
        _, _, _, f = self.neural_net(params, x, y)

        return f * self.L_f / (self.L**2)

    @partial(jit, static_argnums=(0,))
    def f_hessian(self, params, x, y):
        f_hessian = hessian(self.f_net, argnums=(1, 2))(params, x, y)
        return f_hessian[0][1] ** 2 + f_hessian[0][1] ** 2 + f_hessian[1][1] ** 2

    @partial(jit, static_argnums=(0,))
    def get_residual(self, params, x, y):
        # return function
        f = lambda params, x, y: self.f_net(params, x, y)

        w_a = jit(grad(self.w_net, argnums=1))
        w_b = jit(grad(self.w_net, argnums=2))
        u_a = jit(grad(self.u_net, argnums=1))
        u_b = jit(grad(self.u_net, argnums=2))
        v_a = jit(grad(self.v_net, argnums=1))
        v_b = jit(grad(self.v_net, argnums=2))

        C = self.E * self.t / (1 - self.nu**2)

        Exx = lambda params, x, y: u_a(params, x, y) + 0.5 * w_a(params, x, y) ** 2
        Eyy = lambda params, x, y: v_b(params, x, y) + 0.5 * w_b(params, x, y) ** 2
        Exy = lambda params, x, y: 0.5 * (
            u_b(params, x, y)
            + v_a(params, x, y)
            + w_a(params, x, y) * w_b(params, x, y)
        )

        Nx = lambda params, x, y: C * (Exx(params, x, y) + self.nu * Eyy(params, x, y))
        Ny = lambda params, x, y: C * (Eyy(params, x, y) + self.nu * Exx(params, x, y))
        Nxy = lambda params, x, y: C * (1 - self.nu) * Exy(params, x, y)

        w_aa = jit(grad(w_a, argnums=1))
        w_ab = jit(grad(w_a, argnums=2))
        w_bb = jit(grad(w_b, argnums=2))
        stress_term = (
            lambda params, x, y: Nx(params, x, y) * w_aa(params, x, y)
            + 2 * Nxy(params, x, y) * w_ab(params, x, y)
            + Ny(params, x, y) * w_bb(params, x, y)
        )

        w_aabb = jit(grad(grad(w_aa, argnums=2), argnums=2))
        w_aaaa = jit(grad(grad(w_aa, argnums=1), argnums=1))
        w_bbbb = jit(grad(grad(w_bb, argnums=1), argnums=2))

        r_net_f_without_f = (
            self.D
            * (w_aaaa(params, x, y) + w_bbbb(params, x, y) + 2 * w_aabb(params, x, y))
            - self.P
            - stress_term(params, x, y)
        )
        r_net_output = r_net_f_without_f + f(params, x, y)
        return (
            r_net_f_without_f,
            r_net_output,
            r_net_f_without_f,
            r_net_output,
        )

    def r_net(self, params, x, y):
        _, residual, _, _ = self.get_residual(params, x, y)
        return residual

    def r_net_nf(self, params, x, y):
        residual, _, _, _ = self.get_residual(params, x, y)
        return residual

    def rp_net_nf(self, params, x, y):
        _, _, residual, _ = self.get_residual(params, x, y)
        return residual

    def rp_net(self, params, x, y):
        _, _, _, residual = self.get_residual(params, x, y)
        return residual

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):

        # Inflow boundary conditions
        u_in_pred = self.u_pred_fn(
            params, self.wall_coords[:, 0], self.wall_coords[:, 1]
        )
        v_in_pred = self.v_pred_fn(
            params, self.wall_coords[:, 0], self.wall_coords[:, 1]
        )
        w_in_pred = self.w_pred_fn(
            params, self.wall_coords[:, 0], self.wall_coords[:, 1]
        )

        u_in_loss = jnp.mean(u_in_pred**2)
        v_in_loss = jnp.mean(v_in_pred**2)
        w_in_loss = jnp.mean(w_in_pred**2)

        u_bc_pred = self.u_pred_fn(params, self.coords[:, 0], self.coords[:, 1])
        v_bc_pred = self.v_pred_fn(params, self.coords[:, 0], self.coords[:, 1])
        w_bc_pred = self.w_pred_fn(params, self.coords[:, 0], self.coords[:, 1])
        f_bc_pred = self.f_pred_fn(params, self.coords[:, 0], self.coords[:, 1])
        bc_pred = jnp.stack(
            [u_bc_pred + self.coords[:, 0], v_bc_pred + self.coords[:, 1], w_bc_pred],
            axis=1,
        )
        w_bc_loss = chamfer_distance_directional_jit(self.bd_coords, bc_pred)

        # Residual losses
        r_pred = self.r_pred_fn(params, self.coords[:, 0], self.coords[:, 1])
        r_loss = jnp.mean(r_pred**2)

        f_in_pred = self.f_pred_fn(
            params, self.wall_coords[:, 0], self.wall_coords[:, 1]
        )
        f_in_loss = jnp.mean(f_in_pred**2)

        eps = 0.0
        push_only = jnp.where(
            f_in_pred < eps, f_in_pred, jnp.zeros_like(f_in_pred)
        )  # if un smaller than eps, than no contact.
        stick_loss = jnp.mean(push_only**2)

        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "w_in": w_in_loss,
            "w_bc": w_bc_loss,
            "r": r_loss,
            "stick": stick_loss,
            "f": f_in_loss,
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

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords):
        w_gt = (
            self.P
            / (64 * self.D)
            * (coords[:, 0] ** 2 + coords[:, 1] ** 2 - self.a**2) ** 2
        )
        w_pred = self.w_pred_fn(params, coords[:, 0], coords[:, 1])

        coords = jnp.array(coords)
        w_error = w_gt / self.L - w_pred / self.L

        return w_error, w_pred


class MembraneEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, coords):
        w_pred = self.model.w_pred_fn(params, coords[:, 0], coords[:, 1])
        u_pred = self.model.u_pred_fn(params, coords[:, 0], coords[:, 1])
        v_pred = self.model.v_pred_fn(params, coords[:, 0], coords[:, 1])
        f_pred = self.model.f_pred_fn(params, coords[:, 0], coords[:, 1])

        w_pred = np.array(w_pred)
        u_pred = np.array(u_pred)
        v_pred = np.array(v_pred)

        color = abs(f_pred)
        color = color / max(color)
        color = cm.viridis(color).squeeze()[..., :3]
        pcd = np.concatenate(
            [
                (coords[:, 0:1] + u_pred.reshape(-1, 1)) / self.model.L,
                (coords[:, 1:2] + v_pred.reshape(-1, 1)) / self.model.L,
                w_pred.reshape(-1, 1) / self.model.L,
                color,
            ],
            axis=-1,
        )
        # save_pointcloud(pcd, filename=f'pred_train_{self.model.config.wandb.name}.ply', save_path='.')

    def __call__(self, state, coords):
        self.log_dict = super().__call__(state, None)

        if self.config.logging.log_errors:
            self.log_errors(state.params, coords)

        if self.config.logging.log_preds:
            self.log_preds(state.params, coords)

        for k, i in self.log_dict.items():
            if "_loss" in k:
                self.log_dict[k] = i / self.config.weighting.init_weights[k[:-5]]

        return self.log_dict
