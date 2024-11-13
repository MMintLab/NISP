import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax import vmap, jacrev
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from nisp.samplers import SpaceSampler_dict
from nisp.utils import save_checkpoint

import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    for data_idx in config.data_idx:
        wandb_config = config.wandb
        wandb_config.name = str(data_idx)

        # Initialize W&B
        wandb.init(project=wandb_config.project, name=wandb_config.name)

        # Get dataset
        (
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
        ) = get_dataset(N=3000, SEQ_IDX=data_idx, noise_lev=config.noise_lev)

        if config.nondim == True:

            E = E / (L**2)  # rubber 0.1 GPa (N/m**2)
            P = P / (L**2)
            t = t * L  # 0.3 mm
            a, b = a * L, b * L
            wall_coords = wall_coords * L
            obs_pcd = obs_pcd * L
            outer_coords = outer_coords * L

        # test_coords = jnp.concatenate([inner_coords, outer_coords ], axis = 0)
        model = models.Membrane(
            config,
            wall_coords,
            obs_pcd,
            outer_coords,
            E,
            P,
            t,
            nu,
            a,
            b,
            L,
            L_e,
        )

        evaluator = models.MembraneEvaluator(config, model)

        # Initialize residual sampler
        data_dict = {"outer_coords": outer_coords}
        res_sampler = iter(
            SpaceSampler_dict(data_dict, config.training.batch_size_per_device)
        )

        print("Waiting for JIT...")
        for step in range(config.training.max_steps):
            start_time = time.time()
            batch = next(res_sampler)

            model.state = model.step(model.state, batch)

            # Log training metrics, only use host 0 to record results
            if jax.process_index() == 0:
                if step % config.logging.log_every_steps == 0:
                    # Get the first replica of the state and batch
                    state = jax.device_get(tree_map(lambda x: x[0], model.state))
                    # batch = jax.device_get(tree_map(lambda x: x[0], batch))

                    log_dict = evaluator(state, outer_coords)
                    wandb.log(log_dict)

            # if step == 0:
            #     print(model.compute_loss_gradients(model.state, batch))
            #     print(log_dict)
            #     breakpoint()

            # Save checkpoint
            if config.saving.save_every_steps is not None:
                if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
                ) == config.training.max_steps:
                    path = os.path.join(workdir, "ckpt", config.wandb.name)
                    save_checkpoint(
                        model.state, path, keep=config.saving.num_keep_ckpts
                    )
