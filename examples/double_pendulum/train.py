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

    for a, b, c, d in config.data_idx:
        # Initialize W&B
        wandb_config = config.wandb
        wandb_config.name = f"{a}_{b}_{c}_{d}_train"
        wandb.init(project=wandb_config.project, name=wandb_config.name)

        # Get dataset
        (L, t, t_dense, g, m0, m1, m2, l, q0, q1, q2, u_control, etc) = get_dataset(
            N=200,
            a=a,
            b=b,
            c=c,
            d=d,
            noise=config.noise,
        )

        if config.nondim == True:
            q0 = q0 * L
            q1 = q1 * L
            q2 = q2 * L
            g = g * L
            l = l * L
            u_control = u_control * L

        # Initialize model (TODO: implement non dimensionalization)
        model = models.VehicleSuspension(
            config,
            m0,
            m1,
            m2,
            l,
            g,
            L,
            u_control,
        )

        evaluator = models.VehicleSuspensionEvaluator(config, model)

        # Initialize residual sampler
        data_dict = {
            "time": t,
            "query": t_dense,
            "q0": q0,
            "q1": q1,
            "q2": q2,
            "u_control": u_control,
        }

        res_sampler = iter(
            SpaceSampler_dict(data_dict, config.training.batch_size_per_device)
        )

        print("Waiting for JIT...")
        for step in range(config.training.max_steps):
            start_time = time.time()
            batch = next(res_sampler)
            model.state = model.step(model.state, batch)

            if step == 0:
                print(model.compute_loss_gradients(model.state, batch))
            # Log training metrics, only use host 0 to record results
            if jax.process_index() == 0:
                if step % config.logging.log_every_steps == 0:
                    print(step)
                    # Get the first replica of the state and batch
                    state = jax.device_get(tree_map(lambda x: x[0], model.state))
                    batch = jax.device_get(tree_map(lambda x: x[0], batch))

                    log_dict = evaluator(state, batch, t)
                    wandb.log(log_dict)

            # Save checkpoint
            if config.saving.save_every_steps is not None:
                if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
                ) == config.training.max_steps:
                    path = os.path.join(workdir, "ckpt", config.wandb.name)
                    save_checkpoint(
                        model.state, path, keep=config.saving.num_keep_ckpts
                    )
