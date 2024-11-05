import os
import torch
import copy

import numpy as np
import open3d as o3d
import jax.numpy as jnp
import scipy.io
import deepxde as dde
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tactileshadow.nominal_dataset import gen_nominal_data
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse
from bubble_tools.bubble_datasets.implemented_datasets.bubble_calibration_dataset import BubbleCalibrationDataset
from bubble_tools.bubble_tools.bubble_pc_tools import tf_camera_to_bubble_frame
from bubble_tools.bubble_tools.bubble_constants import get_bubble_frame_from_camera_frame
from bubble_tools.bubble_datasets.implemented_datasets.bubble_calibration_board_dataset import BubbleCalibrationBoardDataset
from bubble_tools.bubble_shape_tools.shape_tools import get_bubble_tool_contact_points_sdf
from bubble_tools.bubble_tools.bubble_img_tools import process_bubble_img, unprocess_bubble_img, filter_depth_map, project_processed_bubble, process_bubble_img_coordinates, unprocess_bubble_img_coordinates
from bubble_tools.bubble_tools.bubble_ellipsoid_tools import load_ellipse
from bubble_tools.bubble_tools.bubble_shear_tools import load_deformation_field, load_deformation_field_advanced,  compute_deformation_field_advanced
from bubble_tools.bubble_datasets.implemented_datasets.bubble_v2_dataset import BubbleV2Dataset
# import mmint_tools.data_utils.loading_utils as load_utils
# from mmint_tools import tr, pose_to_matrix, matrix_to_pose, transform_matrix_inverse
# from mmint_tools.camera_tools.img_utils import project_depth_image
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"





def get_dataset(N=100000, SEQ_IDX=0, data_name = None, noise_lev = 0):
    data_dir = 'data'# Replace this with the path to the data
    data_name = 'bubble_v2_tactile_shadowing_data' if data_name is None else data_name
    # data_name = 'bubble_v2_tactile_shadowing_data_only_o_45'

    data_path = os.path.join(data_dir, data_name)
    dataset = BubbleV2Dataset(data_name=data_path, 
                                        smoothed=True, 
                                        load_cache=False, 
                                        load_depth_filtered = True)

    sample = dataset[SEQ_IDX]
    wrench_w = torch.tensor(sample['wrench_w']) # (N, 6)

    L = 15. # Scalling factor 
    L_e = 1e3 #e6
    a, b = 0.06, 0.04
    nominal_pressure = 97650
    E, P, t, nu = 341260, sample["pressure_final"]-nominal_pressure, 0.00045, 0.5
    ellipse_points = load_ellipse(2*a, 2*b, num_points= int(jnp.sqrt(N)))
    
    
    outer = dde.geometry.Ellipse([0,0], a, b)


    # depth observation
    data_i = sample
    obs_pcd = data_i['bubble_pointcloud_bf'][...,:3].reshape(-1,3)

    noise = np.random.normal(0, noise_lev, size = obs_pcd[:,0].shape)
    obs_pcd[:,2] = obs_pcd[:,2] + noise

    bubble_pcd_noise = copy.copy(data_i['bubble_pointcloud_bf'])
    bubble_pcd_noise[:,:3] = obs_pcd
    sample["bubble_pointcloud_bf"] = bubble_pcd_noise
    
     
    obs_pcd = jnp.array(obs_pcd)
    outer_coords = jnp.array(outer.uniform_points(N))
    wall_coords = jnp.array(ellipse_points)

    E = jnp.array(E)
    P = jnp.array(P)
    t = jnp.array(t)
    nu = jnp.array(nu)
    a = jnp.array(a)
    b = jnp.array(b)


    plt.scatter(outer_coords[:,0], outer_coords[:,1], s = 0.02)
    plt.savefig("result/query.png")
    plt.close()



    # # generate ground truth contact using sdf
    contact_mask_r = get_bubble_tool_contact_points_sdf(data_i['bubble_pc_bcf'][...,:3], 
                                                        data_i['shape_id'], 
                                                        data_i['w_X_bcf'], 
                                                        data_i['w_X_sf'], 
                                                        contact_threshold=0.003)


    delta_depth = process_bubble_img( data_i['depth_ref'] - data_i['depth'], extended=True)
    delta_depth = filter_depth_map(delta_depth)
    etc = {"wrench_w": data_i['wrench_w'],
           "delta_depth": delta_depth,
           "bubble_pc_bcf": data_i['bubble_pc_bcf'][...,:3], 
           "contact_mask_r": contact_mask_r,
           "obs_color": data_i['bubble_pc_bcf'][...,3:],
           "bubble_pointcloud_cf": data_i["bubble_pointcloud_cf"],
           "SEQ_IDX":SEQ_IDX,
           "data_name":data_name}

    return     (
        obs_pcd,
        wall_coords,
        outer_coords, 
        E,
        P,
        t, 
        nu,
        a, 
        b,
        L,
        L_e,
        etc,
    )
if __name__ == '__main__':
    get_dataset()
