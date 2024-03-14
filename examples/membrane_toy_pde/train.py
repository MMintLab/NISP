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

from jaxpi.samplers import SpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset



    
def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    logger = Logger()

    # Get dataset
    (
        coords,
        wall_coords,
        E,
        P,
        t, 
        nu,
        a, 
        b 
    ) = get_dataset()



    if config.nondim == True:
        L = 20. # Scalling factor 
        E = E / (L **2) # rubber 0.1 GPa (N/m**2)
        P = P / (L **2)
        t = t * L# 0.3 mm
        # geom = dde.geometry.Interval(-1, 1)
        a, b = 0.04*L, 0.04*L
        wall_coords = wall_coords * L
        coords = coords * L
 
    # Initialize model (TODO: implement non dimensionalization)
    model = models.Membrane(
        config,
        wall_coords,
        E, P, t, nu,         
        a, 
        b,
        L     
    )

    evaluator = models.MembraneEvaluator(config, model)

    # Initialize residual sampler
    res_sampler = iter(SpaceSampler(coords, config.training.batch_size_per_device))

    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)
        print("sampled")
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, coords)
                wandb.log(log_dict, step)

                end_time = time.time()
                # Report training metrics

                # logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model
