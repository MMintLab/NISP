import ml_collections
import math
import jax.numpy as jnp
from configs.train_cfg import get_config as get_train_config

train_config = get_train_config()

def get_config():
    """Get the default hyperparameter configuration."""
    config = train_config # ml_collections.ConfigDict()
    config.mode = "eval"
    config.wandb.name = train_config.wandb.name
    

    return config
