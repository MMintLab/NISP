import time
import os
import jax
import ml_collections
import wandb
import matplotlib.pyplot as plt

import jax.numpy as jnp
from absl import logging
from jax import vmap, jacrev
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import models
from nisp.samplers import SpaceSampler_dict
from nisp.utils import save_checkpoint
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    for data_idx in config.data_idx:
        # Initialize W&B
        wandb_config = config.wandb
        wandb_config.name = str(data_idx)  # + "_3"
        wandb.init(project=wandb_config.project, name=wandb_config.name)

        # Get dataset
        (
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
        ) = get_dataset(
            N=5000, IDX=data_idx
        )  # 100000

        if config.nondim == True:
            rho = rho / L**3
            g = g * L
            E = E / (L**2)

            query_in = query_in * L
            surface_points = surface_points * L
            def_surface_points = def_surface_points * L
            combined_pcd_world = etc["combined_pcd_world"] * L
            # gt_wrench = gt_wrench
            # contact_pts = contact_pts * L

        # Initialize model (TODO: implement non dimensionalization)
        model = models.VehicleSuspension(
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
            etc["tri_indices"],
            etc["bottom_indices"],
            etc["non_contact_index"],
        )

        evaluator = models.VehicleSuspensionEvaluator(config, model)

        # Initialize residual sampler
        data_dict = {
            "query": query_in,
            #  "query_surf_idx": query_surf_idx,
            "surface_points": def_surface_points,
            #  "contact_idx": contact_idx,
            "normals": normals,
        }

        res_sampler = iter(
            SpaceSampler_dict(data_dict, config.training.batch_size_per_device)
        )

        print("Waiting for JIT...")
        for step in range(config.training.max_steps):
            start_time = time.time()
            batch = next(res_sampler)
            model.state = model.step(model.state, batch)

            # if step ==0:
            #     print(model.compute_loss_gradients(model.state, batch))
            #     breakpoint()

            # # Update weights if necessary
            # if config.weighting.scheme in ["grad_norm", "ntk"]:
            #     if step % config.weighting.update_every_steps == 0 and  step != 0:
            #         model.state = model.update_weights(model.state, batch)

            # Log training metrics, only use host 0 to record results
            if jax.process_index() == 0:
                if step % config.logging.log_every_steps == 0:
                    print(step)
                    # Get the first replica of the state and batch
                    state = jax.device_get(tree_map(lambda x: x[0], model.state))
                    batch = jax.device_get(tree_map(lambda x: x[0], batch))

                    log_dict = evaluator(state, batch)
                    wandb.log(log_dict)

            #         end_time = time.time()
            # Report training metrics
            # logger.log_iter(step, start_time, end_time, log_dict)
            # if step == 0:
            #     print(log_dict)
            #     breakpoint()

            # Save checkpoint
            if config.saving.save_every_steps is not None:
                if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
                ) == config.training.max_steps:
                    path = os.path.join(workdir, "ckpt", wandb_config.name)
                    print(path)

                    save_checkpoint(
                        model.state, path, keep=config.saving.num_keep_ckpts
                    )
