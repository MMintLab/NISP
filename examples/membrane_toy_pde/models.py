from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian
from jax.experimental.jet import jet
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn
from jax.experimental.host_callback import call

import matplotlib.cm as cm
from mmint_tools.camera_tools.pointcloud_utils import save_pointcloud

class Membrane(ForwardBVP):
    def __init__(
        self,
        config,
        wall_coords,
        E, P, t, nu, a, b, L,
    ):
        super().__init__(config)

        self.E = E  # 1.0 # rubber 0.1 GPa
        self.nu = nu #0.5 # Latex behaves like an incompressible
        self.P = P #2.757
        self.t = t #0.0003
        self.D =  self.E * self.t **3 / (12 * (1 - self.nu**2))
        self.nu = nu
        self.a =  a
        self.b = b
        self.L = L
        

        # Initialize coordinates
        self.wall_coords = wall_coords


        # Predict functions over batch
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0)) # basically showing the index of the batch
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.w_pred_fn = vmap(self.w_net, (None, 0, 0))

        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))

    def neural_net(self, params, x, y):
        # x = x / self.L + self.L/2# rescale x into [0, 1]
        # y = y / self.W  + self.W/2 # rescale y into [0, 1]

        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        w = outputs[2]
        return u, v, w

    def u_net(self, params, x, y):
        u, _, _ = self.neural_net(params, x, y)
        return u #* 5e-1 # scale factor

    def v_net(self, params, x, y):
        _, v, _ = self.neural_net(params, x, y)
        return v #* 5e-1

    def w_net(self, params, x, y):
        _, _, w = self.neural_net(params, x, y)
        return w

    def r_net(self, params, x, y):
        # u, v, w = self.neural_net(params, x, y)

        w_fn_x = lambda x: self.w_net(params, x, y)
        _, (_, _, _, w_aaaa) = jet(w_fn_x, (x,), [[1.0, 0.0, 0.0, 0.0]])

        w_fn_y = lambda y: self.w_net(params, x, y)
        _, (_, _, _, w_bbbb) = jet(w_fn_y, (y,), [[1.0, 0.0, 0.0, 0.0]])
        # jax.debug.print("🤯 w_a  {x} 🤯", x= w_a)



    
        w_a = grad(self.w_net, argnums=1)
        w_b = grad(self.w_net, argnums=2)
        u_a = grad(self.u_net, argnums=1)
        u_b = grad(self.u_net, argnums=2)
        v_a = grad(self.v_net, argnums=1)
        v_b = grad(self.v_net, argnums=2)

        C = self.E * self.t / (1 - self.nu **2)
        # jax.debug.print("🤯 w_a {x} 🤯", x=w_a(params, x, y))

        # Wiki version formulation
        # N_11 = lambda params, x, y: C * 0.5  * (2 * u_a (params, x, y) + 2 * self.nu * v_b (params, x, y) + w_a (params, x, y) **2 + self.nu * w_b (params, x, y)** 2)
        # N_22 =  lambda params, x, y: C * 0.5 * (2 * self.nu * u_a (params, x, y)  + 2 * v_b (params, x, y) + self.nu * w_a (params, x, y) **2 +  w_b (params, x, y) ** 2)
        # N_12 = lambda params, x, y:  C * ( 1 - self.nu) * 0.5  * (u_b (params, x, y) + v_a (params, x, y) + w_a  (params, x, y)* w_b (params, x, y))

        # stress_term1 = lambda params, x, y: N_11 (params, x, y) * w_a (params, x, y)+ N_12 (params, x, y) * w_b (params, x, y)
        # stress_term2 = lambda params, x, y: N_12 (params, x, y) * w_a (params, x, y)+ N_22 (params, x, y) * w_b (params, x, y)
        # stress_term =  grad(stress_term1, argnums=1)(params, x, y) + grad(stress_term2, argnums=2)(params, x, y)


        # IIT lecture note version formulation
        Exx = lambda params, x, y: u_a(params, x, y) + 0.5 * w_a(params, x, y) **2
        Eyy = lambda params, x, y: v_b(params, x, y) + 0.5 * w_b(params, x, y) **2
        Exy = lambda params, x, y: 0.5 * ( u_b(params, x, y) + v_a(params, x, y) + w_a(params, x, y) * w_b(params, x, y))

        Nx = lambda params, x, y:  C * (Exx(params, x, y) + self.nu * Eyy(params, x, y))
        Ny = lambda params, x, y:  C * (Eyy(params, x, y) + self.nu * Exx(params, x, y))
        Nxy = lambda params, x, y:  C * (1 - self.nu) * Exy(params, x, y)

        w_aa = grad(w_a, argnums=1)
        w_ab = grad(w_a, argnums=2)
        w_bb = grad(w_b, argnums=2)

        stress_term = Nx(params, x, y) * w_aa(params, x, y) + 2 * Nxy(params, x, y) * w_ab(params, x, y) + Ny (params, x, y)* w_bb(params, x, y)

        w_aabb = grad(grad(grad(grad(self.w_net, argnums=1), argnums=1), argnums=2), argnums=2)(params, x, y)
        pure_bending_term = self.D  * (w_aaaa + w_bbbb + 2*w_aabb)

        pde_result =  pure_bending_term - self.P - stress_term

        # jax.debug.print("🤯 D/C {x}  🤯", x=self.D / C)

        # w_a = grad(self.w_net, argnums=1)(params, x, y)
        # u_a = grad(self.u_net, argnums=1)(params, x, y)
        # v_a = grad(self.v_net, argnums=1)(params, x, y)
        # jax.debug.print("🤯 w_a{x}  🤯", x=w_a)
        # jax.debug.print("🤯 u_a{x}  🤯", x=w_q/u_a)

        return pde_result


    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # x = batch[:,1]
        # y = batch[:,0]

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

        # jax.debug.print("🤯 {x} 🤯", x=batch)

        # Residual losses
        r_pred = self.r_pred_fn(
            params, batch[:, 0], batch[:, 1]
        )

        r_loss = jnp.mean(r_pred**2)

        # jax.debug.print("🤯 bc {x} 🤯", x=u_in_loss+v_in_loss+w_in_loss)
        # jax.debug.print("🤯 loss {x} 🤯", x=r_loss)


        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "w_in": w_in_loss,

            "r": r_loss,
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
        w_gt = self.P/ (64 * self.D) * (coords[:,0]**2 + coords[:,1]**2  - self.a**2) **2
        w_pred = self.w_pred_fn(params, coords[:, 0], coords[:, 1])
        u_pred = self.u_pred_fn(params, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, coords[:, 0], coords[:, 1])

        jax.debug.print("🤯 gt w max {x} [mm] 🤯", x=jnp.max(w_gt)/self.L*1e3)
        jax.debug.print("🤯 w max {x} [mm] 🤯", x=jnp.max(w_pred)/self.L*1e3)
        jax.debug.print("🤯 w min {x} [mm]  🤯", x=jnp.min(w_pred)/self.L*1e3)
        jax.debug.print("🤯 u max {x} [mm] 🤯", x=jnp.max(u_pred)/self.L*1e3)
        jax.debug.print("🤯 u min {x} [mm]  🤯", x=jnp.min(u_pred)/self.L*1e3)
        jax.debug.print("🤯 v max {x} [mm] 🤯", x=jnp.max(v_pred)/self.L*1e3)
        jax.debug.print("🤯 v min {x} [mm]  🤯", x=jnp.min(v_pred)/self.L*1e3)



        coords = jnp.array(coords)
        w_error = abs(w_gt/self.L - w_pred/self.L)


        

        return w_error, w_pred


class MembraneEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, coords):
        w_error, w_pred = self.model.compute_l2_error(
            params, coords
        )

        w_error = np.array(abs(w_error))
        w_pred = np.array(w_pred)
        coords = np.array(coords)

        color_max = 0.01
        color = cm.hot(w_error/color_max).squeeze()[..., :3]

        pcd = np.concatenate([coords[:,0:1]/self.model.L, coords[:,1:2]/self.model.L, 
                            w_pred.reshape(-1,1)/self.model.L , color], axis = -1)
        save_pointcloud(pcd, filename=f'pred_train', save_path='.')

        w_error = abs(w_error)
        self.log_dict["w_error"] = w_error

    def __call__(self, state, batch, coords):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, coords)

        if self.config.logging.log_preds:
            self.log_preds(state.params, coords)
        self.log_dict

        return self.log_dict
