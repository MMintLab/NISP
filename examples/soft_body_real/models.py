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
from jaxpi.chamfer_distance import chamfer_distance_jit, chamfer_distance_directional
from mmint_tools.camera_tools.pointcloud_utils import tr_pointcloud


class VehicleSuspension(ForwardBVP):
    def __init__(
        self,
        config,
        rho,
        g,
        L,
        nu,
        E,
        surface_points,
        combined_pcd_world,
        normals,
        wrist_to_world,
        gt_wrench,
        tri_indexes,
        bottom_indices,
        non_contact_index,
    ):
        super().__init__(config)

        self.g = g
        self.L = L
        self.nu = nu
        self.E = E
        self.rho = rho
        self.surface_points = surface_points
        self.combined_pcd_world = combined_pcd_world
        self.surface_normals = normals
        self.wrist_to_world = wrist_to_world

        self.gt_wrench = gt_wrench
        self.tri_indexes = tri_indexes
        self.bottom_indices = bottom_indices
        self.non_contact_index = non_contact_index

        # Predict functions over batch
        self.q0_pred_fn = vmap(self.q0_net, (None, 0, 0, 0))
        self.q1_pred_fn = vmap(self.q1_net, (None, 0, 0, 0))
        self.q2_pred_fn = vmap(self.q2_net, (None, 0, 0, 0))

        self.f_pred_fn = vmap(self.f_net, (None, 0, 0, 0))
        self.stress_pred_fn = vmap(self.stress_net, (None, 0, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, x, y, z):
        inputs = jnp.stack([x, y, z])
        outputs = self.state.apply_fn(params, inputs)
        q0 = outputs[0]
        q1 = outputs[1]
        q2 = outputs[2]
        f = outputs[3]
        return q0, q1, q2, f

    def q0_net(self, params, x, y, z):
        q0, _, _, _ = self.neural_net(params, x, y, z)
        return q0

    def q1_net(self, params, x, y, z):
        _, q1, _, _ = self.neural_net(params, x, y, z)
        return q1

    def q2_net(self, params, x, y, z):
        _, _, q2, _ = self.neural_net(params, x, y, z)
        return q2

    def f_net(self, params, x, y, z):
        _, _, _, f = self.neural_net(params, x, y, z)
        return jax.nn.gelu(f) * 1e3 / self.L**2  # Inductive bias. push only.

    def stress(self, params, x, y, z):
        # uxx, uxy, uxz =  lambda params, x, y, z:grad(self.q0_net, argnums=[1,2,3])(params, x, y, z)
        # uyx, uyy, uyz =  lambda params, x, y, z:grad(self.q1_net, argnums=[1,2,3])(params, x, y, z)
        # uzx, uzy, uzz =  lambda params, x, y, z:grad(self.q2_net, argnums=[1,2,3])(params, x, y, z)

        uxx = grad(self.q0_net, argnums=1)
        uxy = grad(self.q0_net, argnums=2)
        uxz = grad(self.q0_net, argnums=3)
        uyx = grad(self.q1_net, argnums=1)
        uyy = grad(self.q1_net, argnums=2)
        uyz = grad(self.q1_net, argnums=3)
        uzx = grad(self.q2_net, argnums=1)
        uzy = grad(self.q2_net, argnums=2)
        uzz = grad(self.q2_net, argnums=3)

        exx = lambda params, x, y, z: uxx(params, x, y, z)
        eyy = lambda params, x, y, z: uyy(params, x, y, z)
        ezz = lambda params, x, y, z: uzz(params, x, y, z)
        exy = lambda params, x, y, z: 0.5 * (
            uxy(params, x, y, z) + uyx(params, x, y, z)
        )
        exz = lambda params, x, y, z: 0.5 * (
            uxz(params, x, y, z) + uzx(params, x, y, z)
        )
        eyz = lambda params, x, y, z: 0.5 * (
            uyz(params, x, y, z) + uzy(params, x, y, z)
        )

        C = jnp.array(
            [
                [1 - self.nu, self.nu, self.nu, 0, 0, 0],
                [self.nu, 1 - self.nu, self.nu, 0, 0, 0],
                [self.nu, self.nu, 1 - self.nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * self.nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * self.nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * self.nu) / 2],
            ]
        )
        C = C * self.E / (1 + self.nu) / (1 - 2 * self.nu)

        strain = lambda params, x, y, z: jnp.array(
            [
                exx(params, x, y, z),
                eyy(params, x, y, z),
                ezz(params, x, y, z),
                exy(params, x, y, z),
                exz(params, x, y, z),
                eyz(params, x, y, z),
            ]
        )  # yz\
        sig0 = lambda params, x, y, z: jnp.dot(C[0], strain(params, x, y, z))
        sig1 = lambda params, x, y, z: jnp.dot(C[1], strain(params, x, y, z))
        sig2 = lambda params, x, y, z: jnp.dot(C[2], strain(params, x, y, z))
        sig3 = lambda params, x, y, z: jnp.dot(C[3], strain(params, x, y, z))
        sig4 = lambda params, x, y, z: jnp.dot(C[4], strain(params, x, y, z))
        sig5 = lambda params, x, y, z: jnp.dot(C[5], strain(params, x, y, z))

        return [
            sig0(params, x, y, z),
            sig1(params, x, y, z),
            sig2(params, x, y, z),
            sig3(params, x, y, z),
            sig4(params, x, y, z),
            sig5(params, x, y, z),
        ]

    def stress_net(self, params, x, y, z):
        stress_vec = self.stress(params, x, y, z)  # xx, yy, zz, xy, xz, yz order
        stress_mat_row1 = jnp.array([stress_vec[0], stress_vec[3], stress_vec[4]])
        stress_mat_row2 = jnp.array([stress_vec[3], stress_vec[1], stress_vec[5]])
        stress_mat_row3 = jnp.array([stress_vec[4], stress_vec[5], stress_vec[2]])
        return stress_mat_row1, stress_mat_row2, stress_mat_row3

    def r_net(self, params, x, y, z):
        uxx = grad(self.q0_net, argnums=1)
        uxy = grad(self.q0_net, argnums=2)
        uxz = grad(self.q0_net, argnums=3)
        uyx = grad(self.q1_net, argnums=1)
        uyy = grad(self.q1_net, argnums=2)
        uyz = grad(self.q1_net, argnums=3)
        uzx = grad(self.q2_net, argnums=1)
        uzy = grad(self.q2_net, argnums=2)
        uzz = grad(self.q2_net, argnums=3)

        exx = lambda params, x, y, z: uxx(params, x, y, z)
        eyy = lambda params, x, y, z: uyy(params, x, y, z)
        ezz = lambda params, x, y, z: uzz(params, x, y, z)
        exy = lambda params, x, y, z: 0.5 * (
            uxy(params, x, y, z) + uyx(params, x, y, z)
        )
        exz = lambda params, x, y, z: 0.5 * (
            uxz(params, x, y, z) + uzx(params, x, y, z)
        )
        eyz = lambda params, x, y, z: 0.5 * (
            uyz(params, x, y, z) + uzy(params, x, y, z)
        )

        C = jnp.array(
            [
                [1 - self.nu, self.nu, self.nu, 0, 0, 0],
                [self.nu, 1 - self.nu, self.nu, 0, 0, 0],
                [self.nu, self.nu, 1 - self.nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * self.nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * self.nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * self.nu) / 2],
            ]
        )
        C = C * self.E / (1 + self.nu) / (1 - 2 * self.nu)

        strain = lambda params, x, y, z: jnp.array(
            [
                exx(params, x, y, z),
                eyy(params, x, y, z),
                ezz(params, x, y, z),
                exy(params, x, y, z),
                exz(params, x, y, z),
                eyz(params, x, y, z),
            ]
        )  # yz\
        sigxx = lambda params, x, y, z: jnp.dot(C[0], strain(params, x, y, z))
        sigyy = lambda params, x, y, z: jnp.dot(C[1], strain(params, x, y, z))
        sigzz = lambda params, x, y, z: jnp.dot(C[2], strain(params, x, y, z))
        sigxy = lambda params, x, y, z: jnp.dot(C[3], strain(params, x, y, z))
        sigxz = lambda params, x, y, z: jnp.dot(C[4], strain(params, x, y, z))
        sigyz = lambda params, x, y, z: jnp.dot(C[5], strain(params, x, y, z))

        sig_xxx = lambda params, x, y, z: grad(sigxx, argnums=1)(params, x, y, z)
        sig_yyy = lambda params, x, y, z: grad(sigyy, argnums=2)(params, x, y, z)
        sig_zzz = lambda params, x, y, z: grad(sigzz, argnums=3)(params, x, y, z)

        sig_xyx = lambda params, x, y, z: grad(sigxy, argnums=1)(params, x, y, z)
        sig_xyy = lambda params, x, y, z: grad(sigxy, argnums=2)(params, x, y, z)
        sig_xzx = lambda params, x, y, z: grad(sigxz, argnums=1)(params, x, y, z)
        sig_xzz = lambda params, x, y, z: grad(sigxz, argnums=3)(params, x, y, z)
        sig_yzy = lambda params, x, y, z: grad(sigyz, argnums=2)(params, x, y, z)
        sig_yzz = lambda params, x, y, z: grad(sigyz, argnums=3)(params, x, y, z)

        div_sig_x = (
            sig_xxx(params, x, y, z)
            + sig_xyy(params, x, y, z)
            + sig_xzz(params, x, y, z)
        )
        div_sig_y = (
            sig_xyx(params, x, y, z)
            + sig_yyy(params, x, y, z)
            + sig_yzz(params, x, y, z)
        )
        div_sig_z = (
            sig_xzx(params, x, y, z)
            + sig_yzy(params, x, y, z)
            + sig_zzz(params, x, y, z)
        )

        residual = [div_sig_x, div_sig_y, div_sig_z]  # - self.g * self.rho
        return residual[0], residual[1], residual[2]

    def tr_pointcloud(self, pc, R=None, t=None, T=None):
        """
        Transform a point cloud given the homogeneous transformation represented by R,t
        R and t can be seen as a tranformation new_frame_X_old_frame where pc is in the old_frame and we want it to be in the new_frame
        Args:
            pc: (..., N, 6) pointcloud or (..., N, 3) pointcloud
            R: (3,3) numpy array i.e. target_R_source
            t: (3,) numpy array i.e. target_t_source
            T: (4,4) numpy array representing the tranformation i.e. target_T_source
        Returns:
            pc_tr: (..., N, 6) pointcloud or (..., N, 3) pointcloud transformed in the
        """
        if T is not None:
            R = T[:3, :3]
            t = T[:3, 3]
        pc_xyz = pc[..., :3]
        # pc_xyz_tr = pc_xyz@R.T + t
        pc_xyz_tr = jnp.einsum("ij,...lj->...li", R, pc_xyz) + t

        # handle RGB info held in the other columns
        if pc.shape[-1] > 3:
            pc_rgb = pc[..., 3:7]
            pc_tr = jnp.concatenate([pc_xyz_tr, pc_rgb], axis=-1)
        else:
            pc_tr = pc_xyz_tr
        return pc_tr

    def get_contact_force(self, surface_coords, surface_pressure):
        forces = jnp.zeros(3)
        torque = jnp.zeros(3)
        # bottom_plane = surface_coords[self.bottom_indices]

        # breakpoint()
        for tri_idx in range(len(self.tri_indexes)):
            i, j, k = self.tri_indexes[tri_idx]
            v1, v2, v3 = surface_coords[
                jnp.array([i, j, k])
            ]  # + deformation[[i, j, k]]

            area_v = jnp.cross(v2 - v1, v3 - v1) / 2.0
            avg_pressure = -jnp.mean(
                surface_pressure[jnp.array([i, j, k])]
            )  # normal inwards to the surface
            normal_force = avg_pressure * area_v

            # torque

            # add normal force to each vertex of the triangle
            forces += normal_force

        # jax.debug.print("🤯 {x} 🤯", x=forces)
        return forces

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        query = batch["query"]

        # Deformation prediction.
        s_x = self.surface_points[:, 0]
        s_y = self.surface_points[:, 1]
        s_z = self.surface_points[:, 2]

        s0_pred = self.q0_pred_fn(
            params,
            s_x,
            s_y,
            s_z,
        )

        s1_pred = self.q1_pred_fn(
            params,
            s_x,
            s_y,
            s_z,
        )

        s2_pred = self.q2_pred_fn(
            params,
            s_x,
            s_y,
            s_z,
        )

        # Deformed sponge prediction.
        q0_def = s0_pred + s_x
        q1_def = s1_pred + s_y
        q2_def = s2_pred + s_z

        # chamfer distance.
        pcd_est = jnp.stack([q0_def, q1_def, q2_def], axis=1)
        pcd_world = self.tr_pointcloud(
            pcd_est, R=self.wrist_to_world, t=jnp.array([0, 0, 0])
        )
        cd_loss = chamfer_distance_directional(self.combined_pcd_world, pcd_world)

        # Traction Estimation.
        surface_normal_wrist = self.wrist_to_world.T @ self.surface_normals.T
        surface_normal_wrist = surface_normal_wrist.T

        stress_pred = self.stress_pred_fn(params, s_x, s_y, s_z)
        f_pred = self.f_pred_fn(
            params,
            s_x,
            s_y,
            s_z,
        )

        # Boundary Condition: Traction balance at the contact patch from the wrist (=sponge origin) frame.
        f_x = (
            jnp.einsum("ij, ij -> i", stress_pred[0], surface_normal_wrist)
            - f_pred * self.wrist_to_world.T[0, 2]
        )
        f_y = (
            jnp.einsum("ij, ij -> i", stress_pred[1], surface_normal_wrist)
            - f_pred * self.wrist_to_world.T[1, 2]
        )
        f_z = (
            jnp.einsum("ij, ij -> i", stress_pred[2], surface_normal_wrist)
            - f_pred * self.wrist_to_world.T[2, 2]
        )
        loss_bc = jnp.mean(f_x**2) + jnp.mean(f_y**2) + jnp.mean(f_z**2)

        # deformation along the normal direction. "un"
        normal_deformation = (
            s0_pred * self.surface_normals[:, 0]
            + s1_pred * self.surface_normals[:, 1]
            + s2_pred * self.surface_normals[:, 2]
        )

        # signorini_loss [partial]: Contact force > 0 iff normal deformation > 0.
        # signorini_loss_ = jnp.mean(f_pred[self.non_contact_index] ** 2)

        # zero contact force @ no normal force  
        signorini_loss_no_force = jnp.where(
            normal_deformation <= 0, f_pred, jnp.zeros_like(f_pred)
        ) 

        # zero contact force @ sides 
        signorini_loss_no_force = jnp.where(self.non_contact_index == True, 
                                                  normal_deformation, 
                                                  signorini_loss_no_force)


        # zero normal deformation @ no contact force
        signorini_loss_no_deformation = jnp.where(
            f_pred <= 0, normal_deformation, jnp.zeros_like(f_pred)
        )

        # zero normal deformation @ sides
        signorini_loss_no_deformation = jnp.where(self.non_contact_index == True, 
                                                  normal_deformation, 
                                                  signorini_loss_no_deformation)


        signorini_loss = jnp.mean(signorini_loss_no_force**2) + jnp.mean(signorini_loss_no_deformation**2)

        # no_pull_loss: Contact force >= 0
        no_pull_loss = jnp.mean(
            jnp.where(f_pred < 0, f_pred, jnp.zeros_like(normal_deformation)) ** 2
        )

        # contact_force_loss: Match wrist contact force
        f_contact = jnp.where(
            jnp.all(self.surface_normals == jnp.array([0, 0, -1.0]), axis=1),
            f_pred,
            jnp.zeros_like(f_pred),
        )   
        f_fixture = jnp.where(
            jnp.all(self.surface_normals == jnp.array([0, 0, +1.0]), axis=1),
            f_pred,
            jnp.zeros_like(f_pred),
        )

        net_contact_forces = self.get_contact_force(pcd_est, f_contact)
        net_fixture_forces = self.get_contact_force(pcd_est, f_fixture)

        contact_force_loss = jnp.mean(
            (net_contact_forces.squeeze() - self.gt_wrench[:3]) ** 2
        )
        fixture_force_loss = jnp.mean(
            (net_fixture_forces.squeeze() + self.gt_wrench[:3]) ** 2
        )

        # PDE residuals
        q_x = query[:, 0]
        q_y = query[:, 1]
        q_z = query[:, 2]

        stress_pred = self.r_pred_fn(params, q_x, q_y, q_z)
        r_loss = jnp.mean(
            stress_pred[0] ** 2 + stress_pred[1] ** 2 + (stress_pred[2]) ** 2
        )

        loss_dict = {
            "cd_loss": cd_loss,
            "r_loss": r_loss,
            "loss_bc": loss_bc,
            "signorini_loss": signorini_loss,
            "stick_loss": no_pull_loss,
            "contact_force_loss": contact_force_loss + fixture_force_loss,
            # "non_penetration_loss": non_penetration_loss,
            # "min_def": def_loss,
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

    def __call__(self, state, batch):
        self.log_dict = super().__call__(state, batch)
        # gt = batch["u_control"]
        # self.log_errors(state.params, t, gt)
        self.log_dict

        return self.log_dict
