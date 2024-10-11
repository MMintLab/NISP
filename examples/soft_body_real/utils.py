import os
import torch
import copy

import numpy as np
import open3d as o3d
import jax.numpy as jnp
import scipy.io
import deepxde as dde
import matplotlib.cm as cm

from tactileshadow.nominal_dataset import gen_nominal_data
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse
from bubble_tools.bubble_datasets.implemented_datasets.bubble_calibration_dataset import (
    BubbleCalibrationDataset,
)
from bubble_tools.bubble_tools.bubble_pc_tools import tf_camera_to_bubble_frame
from bubble_tools.bubble_tools.bubble_constants import (
    get_bubble_frame_from_camera_frame,
)
from mmint_tools import tr, pose_to_matrix, matrix_to_pose, transform_matrix_inverse
from bubble_tools.bubble_datasets.implemented_datasets.bubble_calibration_board_dataset import (
    BubbleCalibrationBoardDataset,
)
from bubble_tools.bubble_shape_tools.shape_tools import (
    get_bubble_tool_contact_points_sdf,
)
import mmint_tools.data_utils.loading_utils as load_utils

from bubble_tools.bubble_tools.bubble_img_tools import (
    process_bubble_img,
    unprocess_bubble_img,
    filter_depth_map,
    project_processed_bubble,
    process_bubble_img_coordinates,
    unprocess_bubble_img_coordinates,
)
from bubble_tools.bubble_tools.bubble_pc_tools import tf_camera_to_bubble_frame
from mmint_tools.camera_tools.img_utils import project_depth_image
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse
from bubble_tools.bubble_tools.bubble_shear_tools import (
    load_deformation_field,
    load_deformation_field_advanced,
    compute_deformation_field_advanced,
)
import scipy.signal
import matplotlib.pyplot as plt
from tool_dataset import ToolDataset
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import deepxde as dde

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Original problem : https://github.com/pydy/pydy_examples/blob/master/npendulum/n-pendulum-control.ipynb and https://x-engineer.org/quarter-car-suspension-transfer-function/
# from __future__ import division, print_function
import sympy as sm
import sympy.physics.mechanics as me
from numpy.linalg import matrix_rank
from scipy.linalg import solve_continuous_are
from numpy.linalg import solve
import mmint_utils
from skspatial.objects import Plane, Points
from mmint_tools.camera_tools.pointcloud_utils import tr_pointcloud
from scipy.spatial import Delaunay


def get_dataset(N=1000, IDX=4):
    L = 50
    g = 9.81
    rho = 0.2450 / (0.046 * 0.046 * 0.046)
    nu, E = 0.1, 1.1e4  # 1.1e4

    dataset_dir = "data/press_2_2_23_test_proc"

    data_fns = sorted(
        [
            f
            for f in os.listdir(dataset_dir)
            if "out" in f and ".pkl.gzip" in f and "contact" not in f
        ],
        key=lambda x: int(x.split(".")[0].split("_")[-1]),
    )
    data_fn = data_fns[IDX]
    example_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, data_fn))

    combined_pcd = example_dict["input"]["combined_pointcloud"]
    contact_pcd = example_dict["test"]["contact_patch"]
    combined_pcd[:, 2] -= 0.080
    contact_pcd[:, 2] -= 0.080

    plane = Plane.best_fit(contact_pcd[:, :3])

    def rotation_matrix_from_vectors(vec1, vec2):
        """Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
            vec2 / np.linalg.norm(vec2)
        ).reshape(3)
        v = np.cross(a, b)
        if any(v):  # if not all zeros then
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

        else:
            return np.eye(3)  # cross of all zeros only occurs on identical directions

    # Infer Sponge Rotation
    direction = plane.normal
    direction = direction * np.sign(direction[2]) * (-1)
    tf_rot = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)
    tf_rot = np.linalg.inv(tf_rot)

    # # Put sponge into the world frame
    # contact_world = tr_pointcloud(contact_pcd[:,:3], R=tf_rot, t= np.array([0, 0,  -plane.point[2]]))
    # combined_pcd_world = tr_pointcloud(combined_pcd[:,:3], R=tf_rot, t= np.array([0, 0, -plane.point[2]]))

    geom = dde.geometry.Cuboid(
        [-0.0235, -0.0235, -0.046],
        [0.0235, 0.0235, 0.0],
    )
    geom = dde.geometry.Cuboid(
        [-0.023, -0.023, -0.046],
        [0.023, 0.023, 0.0],
    )
    surface_points = geom.uniform_boundary_points(0.6 * N)
    surface_normals = geom.get_surface_normals(surface_points)
    query = geom.uniform_points(
        n=None, dx=abs(surface_points[1, 1] - surface_points[0, 1])
    )

    plane = Plane.best_fit(contact_pcd[:, :3])
    direction = plane.normal
    direction = direction * np.sign(direction[2]) * (-1)

    # T: wrist -> world
    world_to_wrist = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)
    wrist_to_world = np.linalg.inv(world_to_wrist)

    # wrist pcd -> world pcd
    contact_world = tr_pointcloud(
        contact_pcd[:, :3], R=wrist_to_world, t=np.array([0, 0, 0])
    )
    combined_pcd_world = tr_pointcloud(
        combined_pcd[:, :3], R=wrist_to_world, t=np.array([0, 0, 0])
    )
    def_surface_points = tr_pointcloud(
        surface_points[:, :3], R=wrist_to_world, t=np.array([0, 0, 0.0])
    )

    gt_wrench = example_dict["input"]["wrist_wrench"]

    # For inverse problem
    surface_index = np.where(surface_normals[:, 2] == -1)[0]
    # non_contact_index = np.where(surface_normals[:, 2] == 0)[0]
    non_contact_index = surface_normals[:, 2] == 0

    bottom_plane = surface_points[surface_index]
    tri = Delaunay(bottom_plane[:, :2])
    tri_indexes = surface_index[tri.simplices]

    L = jnp.array(L)
    g = jnp.array(g)
    rho = jnp.array(rho)
    nu = jnp.array(nu)
    E = jnp.array(E)

    query = jnp.array(query)
    query_in = jnp.array(query)

    surface_points = jnp.array(surface_points)
    def_surface_points = jnp.array(def_surface_points)
    wrist_to_world = jnp.array(wrist_to_world)

    normals = jnp.array(surface_normals)
    gt_wrench = jnp.array(gt_wrench)
    tri_indexes = jnp.array(tri_indexes)
    surface_index = jnp.array(surface_index)

    etc = {
        "idx": IDX,
        "combined_pcd": combined_pcd,
        "contact_pcd": contact_pcd,
        "tri_indices": tri_indexes,
        "bottom_indices": surface_index,
        "non_contact_index": non_contact_index,
        "combined_pcd_world": combined_pcd_world,
    }
    return (
        L,
        g,
        rho,
        nu,
        E,
        query_in,
        contact_world,
        surface_points,
        def_surface_points,
        normals,
        wrist_to_world,
        gt_wrench,
        etc,
    )


if __name__ == "__main__":
    get_dataset()
